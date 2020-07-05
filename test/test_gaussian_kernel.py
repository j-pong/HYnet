import numpy as np


def gaussian_func(x, m=0.0, sigma=1.0):
    norm = np.sqrt(2 * np.pi * sigma ** 2)
    dist = (x - m) ** 2 / (2 * sigma ** 2)
    return 1 / norm * np.exp(-dist)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dim = 83
    x = np.linspace(0, 511, 512)
    kernel = np.stack([gaussian_func(x, int(i * 512/83), 13.0) for i in range(dim)], axis=0)
    kernel = kernel / np.amax(kernel)

    u_k = np.expand_dims(kernel, 0)
    print(np.shape(u_k))
    print(np.amax(kernel), np.amin(kernel))

    plt.imshow(kernel, aspect='auto')
    plt.savefig("test.png")
