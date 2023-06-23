import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_records(record_dir: str, save: bool = False):
    loss_records_filename = os.path.join(record_dir, 'loss_records.npz')

    # check record_dir
    if not os.path.isdir(record_dir):
        print("No records directory found.")
        sys.exit()

    # plot costs curve
    loss_records = np.load(loss_records_filename)

    train_steps = []  # see line 343 of dqn.py, starting from 0 in each record of `loss_records`
    loss_values = []
    for k, v in sorted(loss_records.items(), key=lambda kwpair: kwpair[0]):
        print(k)
        if len(train_steps):  # not empty
            train_steps = np.append(
                train_steps,
                v[:, 0] - v[0, 0] + train_steps[-1] + 1
            )
        else:
            train_steps = v[:, 0]
        loss_values = np.append(loss_values, v[:, 1])

    print(train_steps)

    ma_len = min(50, len(train_steps))
    moving_avg = np.zeros(len(train_steps) - ma_len + 1)
    moving_std = np.zeros(len(train_steps) - ma_len + 1)
    for i in range(0, len(train_steps) - ma_len + 1):
        moving_avg[i] = np.mean(loss_values[i:i + ma_len])
        moving_std[i] = np.std(loss_values[i:i + ma_len], ddof=1)

    fig, axes = plt.subplots()
    axes.scatter(train_steps, loss_values, s=0.6)
    axes.plot(
        train_steps[ma_len - 1:],
        moving_avg,
        color="orange",
        label=f"simple moving average (ma_len={ma_len})"
    )
    axes.fill_between(
        train_steps[ma_len - 1:],
        moving_avg - moving_std,
        moving_avg + moving_std,
        facecolor="orange",
        alpha=0.3
    )
    axes.set_xlabel("Training step")
    axes.set_ylabel("Loss")
    axes.legend()

    if save:
        fig.savefig(os.path.join(record_dir, "loss_plot.pdf"))
