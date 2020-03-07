import numpy as np
import matplotlib.pyplot as plt


def set_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.autoscale(enable=True, axis='x', tight=True)


t = np.arange(0.0, 1.0 + 0.01, 0.01)
s = np.cos(2 * 2 * np.pi * t)
t[41:60] = np.nan

fig = plt.figure()
ax = fig.add_subplot(4, 1, 1)
set_style(ax)
ax.plot(t, s, '-', lw=2)
ax.set_title('A sine wave with a gap of NaNs between 0.4 and 0.6')
ax.grid()

ax = fig.add_subplot(4, 1, 2)
set_style(ax)
t[0] = np.nan
t[-1] = np.nan
ax.plot(t, s, '-', lw=2)
ax.set_title('Also with NaN in first and last point')
ax.grid()

ax = fig.add_subplot(4, 1, 3)
set_style(ax)
t[0] = np.nan
t[-1] = np.nan
ax.plot(t, s, '-', lw=2)
ax.set_title('Also with NaN in first and last point')
ax.grid()

ax = fig.add_subplot(4, 1, 4)
set_style(ax)
t[0] = np.nan
t[-1] = np.nan
ax.plot(t, s, '-', lw=2)
ax.set_title('Also with NaN in first and last point')
ax.grid()

fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
plt.show()
