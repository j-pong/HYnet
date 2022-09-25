import logging
from itertools import groupby
import numpy as np
import torch

from fairseq import utils

class Segmentor(object):
    def __init__(
        self, 
        models,
        tgt_dict,
        text_seg_type="word",
        use_cuda=True,
        debug_level=False,
    ):
        super().__init__()

        # Model related
        self.models = models
        self.tgt_dict = tgt_dict

        self.bos_index = self.tgt_dict.bos_index
        self.split_index = self.tgt_dict.unk_index + 1
        self.use_cuda = use_cuda

        # Data related
        self.text_seg_type = text_seg_type

        # Other
        self.debug_level = debug_level
    
    @torch.no_grad()
    def _calc_hyps_and_state(
        self,
        output,
    ):
        # Decoding (greedy)
        hyps = output.transpose(0, 1).argmax(-1) # (B, T)

        # Caculate clip point mask
        prev_state = hyps == self.split_index
        current_state = hyps != self.split_index

        prev_state = torch.cat([
            torch.ones_like(prev_state)[:, 0:1], 
            prev_state
            ], dim=-1
        )
        prev_state = prev_state[:,:-1]

        state = current_state * prev_state

        return hyps, state

    @torch.no_grad()
    def _calc_durations(
        self,
        sample,
        prop2raw
    ):
        # Caculate hypothesis of the inputs
        output = self.models(**sample["net_input"], mask=False)
        output = output["encoder_out"]

        # ratio 
        input_length = sample["net_input"]["source"].size(-1)
        output_length = output.size(0)
        if prop2raw:
            io_ratio = input_length / output_length
        else:
            io_ratio = 1

        # Parse the targets from hypothesis
        if self.text_seg_type == "char":
            raise NotImplementedError

        elif self.text_seg_type == "word":
            hyps, state = self._calc_hyps_and_state(output)

            # Duration prediction
            dur_batch = []
            gtui_batch = []
            for i, s in enumerate(state):
                # T only tensor 
                cps = s.nonzero(as_tuple=True)[0]
                gtui = np.array_split(
                    hyps[i].cpu().numpy(),
                    cps[1:].cpu().numpy().tolist() # the list is [True, False, False, True, False, False, ... ]  thus fisrt is not a split point
                )
                for enum, gt in enumerate(gtui):
                    gtui[enum] = torch.from_numpy(gt)
                gtui = tuple(gtui)
                
                dur = cps * io_ratio
                dur = torch.cat([dur, torch.ones_like(dur[0:1]) * input_length])
 
                dur_batch.append(dur)
                gtui_batch.append(gtui)

        return dur_batch, gtui_batch, io_ratio

    def segment(
        self,
        sample,
        prop2raw=False
    ):
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        # Greedy decoding for speech selection 
        dur_batch, gtui_batch, io_ratio = self._calc_durations(sample, prop2raw)
        assert len(dur_batch) == 1
        for batch_ind, dur in enumerate(dur_batch):
            # Calculate duration on sample                
            gtui = gtui_batch[batch_ind]
            gtui_len = len(gtui)
            prev_d = dur[0]

            sample_results = {"dur":[], 'ltr':[]}
            for i, (d, g) in enumerate(zip(dur[1:], gtui)):
                # segmentation and save the results
                g = g.cpu().numpy().tolist()
                gtui_ltr = [x[0] for x in groupby(g)]
                ltr = self.tgt_dict.string(gtui_ltr)

                # caculate margin
                if gtui_len > 1:
                    if i == 0 :
                        post_g = gtui[i+1]
                        m = torch.logical_or(post_g == self.bos_index, post_g == self.split_index).float()
                        post_m = m.cumprod(-1).sum() * io_ratio
                        prev_m = 0.0
                    elif i == len(gtui) - 1:
                        prev_g = gtui[i-1]
                        m = torch.logical_or(prev_g == self.bos_index, prev_g == self.split_index).float()
                        prev_m = m.flip(0).cumprod(-1).sum() * io_ratio
                        post_m = 0.0
                    else:
                        post_g = gtui[i+1]
                        m = torch.logical_or(post_g == self.bos_index, post_g == self.split_index).float()
                        post_m = m.cumprod(-1).sum() * io_ratio
                        prev_g = gtui[i-1]
                        m = torch.logical_or(prev_g == self.bos_index, prev_g == self.split_index).float()
                        prev_m = m.flip(0).cumprod(-1).sum() * io_ratio
                else:
                    prev_m = 0
                    post_m = 0
                    logging.warn(f"Segment model inferene that {batch_ind}-th utterance has 1 word.")

                d_start = max(int(prev_d - prev_m), 0)
                d_end = min(int(d + post_m), int(dur[-1]))

                sample_results["dur"].append([d_start, d_end])
                ltr = ltr.strip().split()
                if len(ltr) > 0:
                    if ltr[-1] == "|":
                        # Expected text is "* * ... * |"
                        ltr = list(filter(lambda l: l != "|", ltr))
                        if len(ltr) == 0:
                            ltr = "<blank>"
                    else:
                        # Expected text is "* * ... *"
                        ltr = "<blank>"
                else:
                    # Expected text is "" 
                    ltr = "<blank>"
                sample_results["ltr"].append(" ".join(ltr) + " |")
                
                # sabe the buffer
                prev_d = d
            sample_results["dur"] = np.array(sample_results["dur"])
            sample_results["ltr"] = " ".join(sample_results["ltr"])
            if len(sample_results["dur"]) != len(sample_results["ltr"].split("|")[:-1]):
                logging.info(
                    f"Length difference between duration and word are {len(sample_results['dur'])}, {len(sample_results['ltr'].split('|')[:-1])}"
                )
                logging.info(f"dur : {sample_results['dur']}")
                logging.info(f"ltr : {sample_results['ltr']}")
                raise ValueError
            
            return sample_results # batch-size 1 thus we can ignore loop
