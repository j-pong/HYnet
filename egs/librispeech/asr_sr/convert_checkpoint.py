from fairseq import checkpoint_utils, utils

state = checkpoint_utils.load_checkpoint_to_cpu("/home/Workspace/HYnet/egs/librispeech/asr_sr/models/SR_checkpoint.pt")
sm = state['cfg']['model']
st = state['cfg']['task']
sc = state['cfg']['criterion']

state_org = checkpoint_utils.load_checkpoint_to_cpu("/home/Workspace/HYnet/egs/librispeech/asr_sr/models/checkpoint_best.pt")
sm_org = state_org['cfg']['model']
st_org = state_org['cfg']['task']
sc_org = state_org['cfg']['criterion']

sm['_name'] = sm_org['_name']
for key in sm.keys():
    if key in sm_org.keys():
        sm_org[key] = sm[key]
    else:
        continue
state['cfg']['model'] = sm_org

st['_name'] = st_org['_name']
for key in st.keys():
    if key in st_org.keys():
        st_org[key] = st[key]
    else:
        continue
state['cfg']['task'] = st_org

sc['_name'] = sc_org['_name']
for key in sc.keys():
    if key in sc_org.keys():
        sc_org[key] = sc[key]
    else:
        continue
state['cfg']['criterion'] = sc_org


def save_checkpoint(state, filename):
    """Save all training state in a checkpoint file."""
    # call state_dict on all ranks in case it needs internal communication
    state_dict = utils.move_to_cpu(state)
    checkpoint_utils.torch_persistent_save(
        state_dict,
        filename,
        async_write=False,
    )

save_checkpoint(state, "/home/Workspace/HYnet/egs/librispeech/asr_sr/models/SR_checkpoint_save.pt")

# print(state['task'])
exit()
