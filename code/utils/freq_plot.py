import numpy as np
import matplotlib.pyplot as plt


def frqplot(tags=['tag' for _ in range(50)], freq=range(50), title='tag freq. in train'):

    ind = np.arange(len(freq))  # the x locations for the groups
    width = 0.1       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, freq, width, color='r')

    # add some text for labels, title and axes ticks
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(tags, rotation=50)
    plt.show()
