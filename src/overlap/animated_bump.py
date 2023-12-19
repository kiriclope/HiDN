import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def init(frames, ax):
    (line,) = ax.plot(frames[0])
    ax.set_xlabel("Neuron #")
    ax.set_ylabel("dF")
    ax.set_ylim([0, int(np.amax(frames)) + 0.1])

    return line


def animate(frame, frames, line):
    line.set_ydata(frames[frame])


def animated_bump(X, window=15, interval=1000):
    frames = []
    n_frames = X.shape[-1]

    for frame in range(n_frames):
        df_i = X[:, frame]
        frames.append(df_i)

    fig, ax = plt.subplots()
    line = init(frames, ax)

    anim = FuncAnimation(
        fig,
        lambda i: animate(i, frames, line),
        frames=n_frames,
        interval=interval,
        repeat=False,
        cache_frame_data=False,
    )

    plt.draw()
    # plt.show()

    writergif = PillowWriter(fps=n_frames)
    anim.save("bump.gif", writer=writergif, dpi=150)

    # plt.close("all")
