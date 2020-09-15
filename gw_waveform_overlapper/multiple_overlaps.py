import os

from celluloid import Camera
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .overlap_computer import compute_overlap


def calculate_multiple_overlaps(w1s, w2s):
    return [compute_overlap(w1, w2) for w1, w2 in zip(w1s, w2s)]


def plot_overlap_line(overlap, x_data_dict, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_data_dict['data'], overlap, 'r')
    ax.set_xlabel(x_data_dict['label'])
    ax.set_ylabel('Overlap Fraction')
    ax.set_xlim(min(x_data_dict['data']), max(x_data_dict['data']))
    return ax


def plot_overlaps(w1s, w2s, overlap_x_data=None, filename='overlap.mp4'):
    overlaps = calculate_multiple_overlaps(w1s, w2s)
    fig, ax = plt.subplots(3, 1, figsize=(5, 10))
    camera = Camera(fig)
    time_ax, freq_ax, overlap_ax = ax[0], ax[1], ax[2]
    if overlap_x_data is None:
        overlap_x_data = dict(label="i", data=[i for i in range(len(w1s))])

    w1_kwargs = dict(color='orange', label=f"Waveform 1")
    w2_kwargs = dict(color='blue', label=f"Waveform 2")

    for w1, w2, o, ox in zip(w1s, w2s, overlaps, overlap_x_data['data']):
        time_ax = w1.plot_time_domain_data(time_ax, **w1_kwargs)
        freq_ax = w1.plot_frequency_domain_data(freq_ax, **w1_kwargs)
        time_ax = w2.plot_time_domain_data(time_ax, **w2_kwargs)
        freq_ax = w2.plot_frequency_domain_data(freq_ax, **w2_kwargs)
        legend_elements = [Line2D([0], [0], **w1_kwargs), Line2D([0], [0], **w2_kwargs)]
        time_ax.legend(loc='upper right', handles=legend_elements)
        overlap_ax = plot_overlap_line(overlaps, overlap_x_data, overlap_ax)
        overlap_ax.scatter([ox], [o], 'k')
        plt.tight_layout()
        camera.snap()

    animation = camera.animate()
    _, ext = os.path.splitext(filename)
    if ext == '.gif':
        animation.save(filename, writer='pillow')
    elif ext == '.mp4':
        animation.save(filename, writer='ffmpeg')
    else:
        raise ValueError(f'Invalid Extension {ext}')
    print(f'File saved at {filename}')
