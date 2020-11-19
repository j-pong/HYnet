import torch
from torch import nn

class ACT_basic(nn.Module):
    def __init__(self,hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, xs, fn, max_hop, masks=None, encoder_output=None):
        if isinstance(xs, tuple):
            inputs, pos_emb = xs[0], xs[1]
        else:
            inputs, pos_emb = xs, None

        # init_hdd
        ## [B, T]
        halting_probability = torch.zeros(xs[0].shape[0],inputs.shape[1]).to(xs[0].device)
        ## [B, T]
        remainders = torch.zeros(xs[0].shape[0],xs[0].shape[1]).to(xs[0].device)
        ## [B, T]
        n_updates = torch.zeros(xs[0].shape[0],xs[0].shape[1]).to(xs[0].device)
        ## [B, T, HDD]
        previous_state = torch.zeros_like(xs[0]).to(xs[0].device)
        step = 0
        # for l in range(self.num_layers):
        while( ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            # state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            # state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
            state = xs[0]

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            state, masks = fn(xs, masks)
            if isinstance(xs, tuple):
                state = state[0]
                xs = (state, pos_emb)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1
        return previous_state, masks
