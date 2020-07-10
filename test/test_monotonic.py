import numpy as np
import torch
import matplotlib.pyplot as plt

arr = np.load('116-288045-0000:seq_energy_dec.ep.7.npy')
arr = arr / np.max(arr)

arr = torch.from_numpy(arr)

arrs = []
cum_arr = arr  # torch.cumprod(arr, dim=-1)
max_len = cum_arr.size(0)
cum_idx = 0

while (True):
    max_idx = torch.argmax(cum_arr, dim=-1)
    cum_arr = cum_arr[max_idx + 10:]
    cum_idx += (max_idx + 1)
    if cum_idx > max_len - 30:
        break
    arrs.append(arr[:max_idx])
    arr = arr[max_idx + 10:]
    print(cum_idx)
arrs.append(arr)

for h, arr in enumerate(arrs, 1):
    arr = arr.numpy()

    plt.subplot(len(arrs), 1, h)
    plt.plot(arr)
    plt.xlabel("time")
    plt.grid()
    plt.autoscale(enable=True, axis='x', tight=True)

# arr = arr.numpy()
#
# plt.plot(arr)
# plt.xlabel("time")
# plt.grid()
# plt.autoscale(enable=True, axis='x', tight=True)

plt.savefig('test.png')
