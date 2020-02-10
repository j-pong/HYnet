import numpy as np

from moneynet.utils.pikachu_dataset import Pikachu as Datasets


def scale(x, n):
    x_f = np.fft.fft(x, n=len(x))
    x_ = np.fft.ifft(x_f, n=n).real

    return x_


def shift_matching(x_query, x_key, b):
    len_q = len(x_query)
    len_k = len(x_key)

    delta = len_q - len_k - np.abs(b)
    if delta > 0:
        if b >= 0:
            x_key = np.pad(x_key, pad_width=(b, delta),
                           mode='constant',
                           constant_values=(0, 0))
        elif b < 0:
            x_key = np.pad(x_key, pad_width=(delta, -b),
                           mode='constant',
                           constant_values=(0, 0))

    elif delta < 0:
        if b > 0:
            x_key = np.pad(x_key, pad_width=(b, 0),
                           mode='constant',
                           constant_values=(0, 0))
            x_key = x_key[:delta]
        elif b <= 0:
            x_key = np.pad(x_key, pad_width=(0, -b),
                           mode='constant',
                           constant_values=(0, 0))
            x_key = x_key[-delta:]

    else:
        if b > 0:
            x_key = np.pad(x_key, pad_width=(b, 0),
                           mode='constant',
                           constant_values=(0, 0))
        elif b < 0:
            x_key = np.pad(x_key, pad_width=(0, -b),
                           mode='constant',
                           constant_values=(0, 0))

    return x_query * x_key


datadir = '../../data/pikachuSFX'
feat_type = 'mfcc'
sample_idx = 22

if __name__ == '__main__':
    datasets = Datasets(root=datadir)
    feats, raw = datasets.__getitem__(index=sample_idx)

    # feature select high freq element
    feats_ = feats
    feats = feats[10:, :]

    # init variables
    base_feat = None
    epsilon = 1e-8

    # display variables
    buffer = [[], [], [], []]
    results = []

    base_feat = feats[:, 30]
    for feat in feats.T:
        if base_feat is not None:
            global_similarity = 0.0
            x_key = base_feat
            x_query = feat
            theta1 = 0.0
            theta2 = 1.0
            n = len(feat)
            for m in np.arange(1, int(n * 5)):
                # previous feature transform with scale
                feat_hat = scale(feat, n=m)

                # get power
                norm_feat_hat = np.linalg.norm(feat_hat, ord=2)
                norm_base_feat = np.linalg.norm(base_feat, ord=2)

                # previous feature transform with shift
                # c.f. the measurment is cosine similiarity
                similarity = np.correlate(feat_hat, base_feat, "full")  # we need to weight for masking data
                numerator = norm_base_feat * norm_feat_hat
                similarity *= 1 / (numerator + epsilon)
                # power uping parameter
                ratio = norm_feat_hat / (norm_base_feat + epsilon)

                # pick just one shift parameter
                local_similarity = np.max(similarity)

                # save the parameter diff
                if local_similarity > global_similarity:
                    theta1 = np.argmax(similarity) - len(base_feat) + 1
                    theta2 = m / n
                    x_query = feat_hat

                    global_similarity = local_similarity

            sim = global_similarity

            results.append(shift_matching(x_key, x_query, b=theta1))

            buffer[0].append(sim)
            buffer[1].append(theta1)
            buffer[2].append(theta2 - 1)
            buffer[3].append(ratio)
        # base_feat = feat

    result = np.stack(results, axis=0).T

    import matplotlib.pyplot as plt
    plt.imshow(result)
    plt.show()
