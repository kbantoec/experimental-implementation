import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_learning_curve(x, scores, out_filename: Optional[str] = None):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    if isinstance(out_filename, str):
        plt.savefig(out_filename)

