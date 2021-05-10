import numpy as np
import torch
import torch.nn.functional as F

def update_one_bests(one_best, batch_range, idx, one_bests):
    one_best -= batch_range[0]
    if one_bests["batch_range"][idx][0] is None:
        one_bests["batch_range"][idx] = [batch_range]
        one_bests["best_batch_idx"][idx] = [one_best]
    else:
        one_bests["batch_range"][idx].append(batch_range)
        one_bests["best_batch_idx"][idx].append(one_best)
    return one_bests

def repeat(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
    return torch.index_select(a, dim, order_index)

def expand(inputs, start_beam, beam_width):
    for f_l in start_beam:
        f, l = f_l
        if isinstance(inputs, list):
            if isinstance(inputs[0], int):
                inputs = torch.tensor(inputs)
                inputs = torch.cat((inputs[:f], inputs[f].repeat(beam_width), inputs[f + 1:]), 0)
                inputs = list(map(int, inputs))
            else:
                for i in range(len(inputs)):
                    inputs[i] = torch.cat((inputs[i][:f], repeat(inputs[i][f].unsqueeze(0), 0, beam_width), inputs[i][f + 1:]), 0)
        else:
            if inputs.dim() > 1:
                inputs = torch.cat((inputs[:f], repeat(inputs[f].unsqueeze(0), 0, beam_width), inputs[f+1:]), 0)
            else:
                inputs = torch.cat((inputs[:f], inputs[f].repeat(beam_width), inputs[f + 1:]), 0)
    return inputs

def shrink(inputs, one_best, flrange):
    f, l = flrange
    if isinstance(inputs, list):
        if isinstance(inputs[0], int):
            inputs = torch.tensor(inputs)
            inputs = torch.cat((inputs[:f], inputs[one_best].unsqueeze(0), inputs[l:]), 0)
            inputs = list(map(int, inputs))
        else:
            for i in range(len(inputs)):
                inputs[i] = torch.cat((inputs[i][:f], inputs[i][one_best].unsqueeze(0), inputs[i][l:]), 0)
    else:
        if inputs.dim() > 1:
            inputs = torch.cat((inputs[:f], inputs[one_best].unsqueeze(0), inputs[l:]), 0)
        else:
            inputs = torch.cat((inputs[:f], inputs[one_best], inputs[l:]), 0)
    return inputs

def pick_one_best(z_out, flrange, score):
    f, l = flrange
    one_best = np.argmax(score[f:l].numpy()) + f
    if isinstance(z_out, list):
        for i in range(len(z_out)):
            z_out[i] = torch.cat((z_out[i][:f], z_out[i][one_best].unsqueeze(0), z_out[i][l:]), 0)
    else:
        z_out = torch.cat((z_out[:f], z_out[one_best].unsqueeze(0), z_out[l:]), 0)
    score = torch.cat((score[:f], torch.zeros(1), score[l:]), 0)
    return z_out, score, one_best

def untile(needs_filter, skip, beam_width):
    """
    :param needs_filter: torch tensor
    :param skip: tuples in list
    :param beam_width: int
    """
    t = 0
    for f_l in skip:
        f = f_l[0] - t
        l = f_l[1] - t
        needs_filter[f] = 0
        needs_filter = torch.cat((needs_filter[:(f+1)], needs_filter[l:]))
        t += (beam_width - 1)
    return needs_filter

def tile(z_out, pred_prob, beam_width, start_beam):
    """
    :param z_out: (Beam Batch, odim) hidden state
    :param pred_prob: (Beam_Batch, 1) Confidence
    :param beam_width: int Beam size
    :param start_beam: Beam Search start index (batch needs to be expanded)
    :return:
    """
    z_out = np.argmax(z_out, axis=1)
    t = 0
    for f_l in start_beam:
        f, l = f_l
        org_f = f - t
        n_best_values, n_best_idx = torch.topk(pred_prob[org_f], beam_width)   # (beam_width)
        z_out = torch.cat((z_out[:f], n_best_idx, z_out[(f+1):]), 0)
        t += beam_width
        
    return z_out

def make_partial_beamset(z_out, score, z_list, c_list, ys_in_pad, hs_pad, hlens, att_w, att_c, olength, th_beta, skip,
                 current_time_step, beam_processed_idx, current_beam_length, beam_width, beam_length, one_bests):

    z_out = z_out.detach().cpu()
    th_beta = th_beta.detach().cpu()

    # get output distribution
    pred_prob = torch.softmax(z_out, dim=-1)    # (Beam_Batch, O)

    # beam_processing = list(beam_processing_batch_idx)
    beam_processing = current_beam_length.astype(bool)
    if True in beam_processing:
        beam_processing_idx = current_beam_length.nonzero()[0]
        if z_out.size(0) - len(beam_processing_idx) * (beam_width - 1) != len(current_beam_length):
            raise RuntimeError("Batchfied beam size is not correct")
    else:
        beam_processing_idx = []

    # Update scores & z_out & z_list using beam search
    n_vocab = pred_prob.size(-1)
    batch_size = pred_prob.size(0)
    numerator = n_vocab if n_vocab > batch_size else batch_size

    # list to tensor
    z_temp = torch.zeros(len(z_list), z_list[0].size(0), z_list[0].size(1))
    for i in range(len(z_list)):
        z_temp[i] = z_list[i].clone()
    z_list = z_temp

    # update score, z_list considering current prediction probability
    for f_l in skip:
        f, l = f_l
        batch_idx = torch.topk(pred_prob[torch.arange(f, l)].view(-1), beam_width, -1)[1] // numerator
        vocab_idx = torch.topk(pred_prob[torch.arange(f, l)].view(-1), beam_width, -1)[1] % numerator
        temp_score = score[torch.arange(f, l)].clone()
        temp_z_list = z_list[:, torch.arange(f, l)].clone()  # (dlayer, beam_size, dunits)
        for beam in range(beam_width):
            prev_score = score[torch.arange(f, l)][batch_idx[beam]]  # (Beam_size)
            z_out_insert = torch.zeros(n_vocab)

            temp_score[beam] = prev_score + pred_prob[torch.arange(f, l)][batch_idx[beam]][vocab_idx[beam]]
            temp_z_list[:, beam] = z_list[:, torch.arange(f, l)][:, batch_idx[beam]]
            z_out_insert[vocab_idx[beam]] = 1

            z_out[beam] = z_out_insert

            # TODO: stop scoring with <eos>?

        score[f:l] = temp_score
        z_list[:, f:l] = temp_z_list
    z_list = list(z_list)

    skip = []   # skip confidence checking for the batch where beam search is processing
    start_beam = []     # which batch idx needs to start beam search = threshold > confidence

    # update skip & shrink memories
    t = 0
    for i in beam_processing_idx:
        k = i + t
        if current_beam_length[i] == beam_length:
            z_out, score, one_best = pick_one_best(z_out, (k, k + beam_width), score)
            # one_bests = {"batch_range": [B, Beam], "best_batch_idx": [B, Beam]}
            one_bests = update_one_bests(one_best, (k, k + beam_width), i, one_bests)
            # shrink memories
            z_list = shrink(z_list, one_best, (k, k + beam_width)) # [dlayer, B, dunits]
            c_list = shrink(c_list, one_best, (k, k + beam_width))  # [dlayer, B, dunits]
            ys_in_pad = shrink(ys_in_pad, one_best, (k, k + beam_width))
            hs_pad[0] = shrink(hs_pad[0], one_best, (k, k + beam_width))  # [B, L, eunits]
            hlens[0] = shrink(hlens[0], one_best, (k, k + beam_width))  # [B, 1]
            att_w = shrink(att_w, one_best, (k, k + beam_width))  # [B, eunits]
            att_c = shrink(att_c, one_best, (k, k + beam_width))  # [B, L]
            current_beam_length[i] = 0
        else:
            skip.append((k, k + beam_width))
            current_beam_length[i] += 1
            t += (beam_width - 1)

    # Check confidence
    pred_prob = torch.softmax(z_out, dim=-1)
    batch_size = pred_prob.size(0)
    needs_filter = pred_prob.view(batch_size, -1)[torch.arange(batch_size),
                                                 ys_in_pad[:, current_time_step].view(batch_size)]
    needs_filter = needs_filter.view(batch_size) < th_beta  # (Beam_Batch)
    needs_filter = needs_filter.int()
    needs_filter = untile(needs_filter, skip, beam_width)  # (Batch)
    needs_filter_idx = needs_filter.nonzero(as_tuple=False).view(-1) if len(needs_filter.nonzero()) != 0 else None

    # return if no confidence is low
    if needs_filter_idx is None:
        z_out = np.argmax(z_out, axis=1)
        return z_out, score, z_list, c_list, ys_in_pad, hs_pad, hlens, att_w, att_c, skip, beam_processed_idx, current_beam_length, one_bests

    # Find expand range
    k = 0
    for l, i in enumerate(current_beam_length):
        if l in needs_filter_idx and i == 0:
            start_beam.append((k, k + beam_width))

            skip.append((k, k + beam_width))
            skip.sort()
            skip_inserted_idx = skip.index((k, k + beam_width))
            for skip_idx in range(skip_inserted_idx + 1, len(skip)):
                skip_f, skip_l = skip[skip_idx]
                skip[skip_idx] = (skip_f + beam_width - 1, skip_l + beam_width - 1)

            k += beam_width
            current_beam_length[l] += 1
        elif i > 0:
            k += beam_width
        else:
            k += 1

    # beam batchfy memories
    z_list = expand(z_list, start_beam, beam_width)  # [dlayer, B, dunits]
    c_list = expand(c_list, start_beam, beam_width)  # [dlayer, B, dunits]
    ys_in_pad = expand(ys_in_pad, start_beam, beam_width)  # [dlayer, B, dunits]
    hs_pad[0] = expand(hs_pad[0], start_beam, beam_width)  # [# enc, B, L, eunits]
    hlens[0] = expand(hlens[0], start_beam, beam_width)  # [# enc, B, 1]
    att_w = expand(att_w, start_beam, beam_width)  # [B, eunits]
    att_c = expand(att_c, start_beam, beam_width)  # [B, L]
    score = expand(score, start_beam, beam_width)  # [B, 1]
    z_out = tile(z_out, pred_prob, beam_width, start_beam)  # get topk results and beam-batchfy them

    # Save beam search processed time range for generating ys_pad, and one best paths for filtering z_list memories
    for idx in needs_filter_idx:
        if beam_processed_idx[idx][0] is None:
            beam_processed_idx[idx] = [
                (current_time_step, current_time_step + beam_length)] if current_time_step + beam_length < olength else [(current_time_step, olength)]
        else:
            if current_time_step + beam_length < olength:
                beam_processed_idx[idx].append((current_time_step, current_time_step + beam_length))
            else:
                beam_processed_idx[idx].append((current_time_step, olength))

    return z_out, score, z_list, c_list, ys_in_pad, hs_pad, hlens, att_w, att_c, skip, beam_processed_idx, current_beam_length, one_bests

def make_beamset(z_out, score, z_list, c_list, ys_in_pad, hs_pad, hlens, att_w, att_c, th_beta, skip, current_time_step,
                 beam_processed_idx, beam_width):
    z_out = z_out.detach().cpu()
    th_beta = th_beta.detach().cpu()

    pred_prob = torch.softmax(z_out, dim=-1)  # (Beam_Batch, O)

    # Update scores & z_out & z_list using beam search
    n_vocab = pred_prob.size(-1)
    batch_size = pred_prob.size(0)
    numerator = n_vocab if n_vocab > batch_size else batch_size

    # list to tensor
    z_temp = torch.zeros(len(z_list), z_list[0].size(0), z_list[0].size(1))
    for i in range(len(z_list)):
        z_temp[i] = z_list[i].clone()
    z_list = z_temp

    for f_l in skip:
        f, l = f_l
        batch_idx = torch.topk(pred_prob[torch.arange(f, l)].view(-1), beam_width, -1)[1] // numerator
        vocab_idx = torch.topk(pred_prob[torch.arange(f, l)].view(-1), beam_width, -1)[1] % numerator
        temp_score = score[torch.arange(f, l)].clone()
        temp_z_list = z_list[:, torch.arange(f, l)].clone()  # (dlayer, beam_size, dunits)
        for beam in range(beam_width):
            prev_score = score[torch.arange(f, l)][batch_idx[beam]]  # (Beam_size)
            z_out_insert = torch.zeros(n_vocab)

            temp_score[beam] = prev_score + pred_prob[torch.arange(f, l)][batch_idx[beam]][vocab_idx[beam]]
            temp_z_list[:, beam] = z_list[:, torch.arange(f, l)][:, batch_idx[beam]]
            z_out_insert[vocab_idx[beam]] = 1

            z_out[beam] = z_out_insert

            # TODO: stop scoring with <eos>?

        score[f:l] = temp_score
        z_list[:, f:l] = temp_z_list
    z_list = list(z_list)

    # Check confidence
    pred_prob = torch.softmax(z_out, dim=-1)
    batch_size = pred_prob.size(0)
    needs_filter = pred_prob.view(batch_size, -1)[torch.arange(batch_size),
                                                  ys_in_pad[:, current_time_step].view(batch_size)]
    needs_filter = needs_filter.view(batch_size) < th_beta  # (Beam_Batch)
    needs_filter = needs_filter.int()
    needs_filter = untile(needs_filter, skip, beam_width)  # (Batch)
    needs_filter_idx = needs_filter.nonzero(as_tuple=False).view(-1) if len(needs_filter.nonzero()) != 0 else None

    # return if no confidence is low
    if needs_filter_idx is None:
        z_out = np.argmax(z_out, axis=1)
        return z_out, score, z_list, c_list, ys_in_pad, hs_pad, hlens, att_w, att_c, skip, beam_processed_idx

    # Find expand range
    skip = []  # skip confidence checking for the batch where beam search is processing
    start_beam = []
    k = 0
    for l, i in enumerate(beam_processed_idx):
        t = l + k
        if i[0] is not None:
            skip.append((t, t + beam_width))
            k += (beam_width - 1)
        elif i[0] is None and l in needs_filter_idx:
            start_beam.append((t, t + beam_width))
            skip.append((t, t + beam_width))
            k += (beam_width - 1)

            # Save beam search processed time range for generating ys_pad, and one best paths for filtering z_list memories
            beam_processed_idx[l] = [current_time_step]

    # beam batchfy memories
    z_list = expand(z_list, start_beam, beam_width)  # [dlayer, B, dunits]
    c_list = expand(c_list, start_beam, beam_width)  # [dlayer, B, dunits]
    ys_in_pad = expand(ys_in_pad, start_beam, beam_width)  # [dlayer, B, dunits]
    hs_pad[0] = expand(hs_pad[0], start_beam, beam_width)  # [# enc, B, L, eunits]
    hlens[0] = expand(hlens[0], start_beam, beam_width)  # [# enc, B, 1]
    att_w = expand(att_w, start_beam, beam_width)  # [B, eunits]
    att_c = expand(att_c, start_beam, beam_width)  # [B, L]
    score = expand(score, start_beam, beam_width)  # [B, 1]
    z_out = tile(z_out, pred_prob, beam_width, start_beam)  # get topk results and beam-batchfy them

    return z_out, score, z_list, c_list, ys_in_pad, hs_pad, hlens, att_w, att_c, skip, beam_processed_idx
