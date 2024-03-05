# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from cycler import cycler
import pylab

# input latex symbols in matplotlib
# https://stackoverflow.com/questions/43741928/matplotlib-raw-latex-epsilon-only-yields-varepsilon
plt.rcParams["mathtext.fontset"] = "cm"


# Plot number in a row: "2", "3", "4"
# 2: Two plots in a row (the smallest fonts)
# 3: Three plots in a row
# 4: Four plots in a row (the biggest fonts)
def get_font_settings(size):
    if size == "2":
        font_size_dict = {"l": 21, "m": 18, "s": 16}
        fig_width = 8  # by default is 6.4 x 4.8
        fig_height = 4
    elif size == "3":
        font_size_dict = {"l": 25, "m": 21, "s": 19}
        fig_width = 8
        fig_height = 4
    else:
        font_size_dict = {"l": 25, "m": 25, "s": 23}
        fig_width = 6.4
        fig_height = 4.8

    xy_label_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["l"])
    title_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["m"])
    legend_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["s"])
    ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', size=font_size_dict["s"])
    cs_xy_label_font = {'fontproperties': xy_label_font}
    cs_title_font = {'fontproperties': title_font}
    cs_xy_ticks_font = {'fontproperties': ticks_font}
    font_factory = {
        'legend_font': legend_font,
        'cs_xy_label_font': cs_xy_label_font,
        'cs_title_font': cs_title_font,
        'cs_xy_ticks_font': cs_xy_ticks_font,
        'fig_width': fig_width,
        'fig_height': fig_height,
    }
    return font_factory


def get_cycle_settings():
    # color names: https://matplotlib.org/stable/gallery/color/named_colors.html
    # colors = plt.get_cmap('tab10').colors  # by default
    # colors = ("tab:blue",) + plt.get_cmap('Set2').colors
    # colors = [plt.cm.Spectral(i / float(6)) for i in range(6)]
    # colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    colors = plt.get_cmap('tab10').colors
    markers = ["D", "o", "^", "s", "*", "X", "d", "x", "1"]
    # my_cycler = cycler(color=colors, marker=markers)
    my_cycler = cycler(color=colors)
    return my_cycler


def plot_time_acc(title, data, legend_pos="", y_lim=None, save_path=None, plot_size="2"):
    font_settings = get_font_settings(plot_size)
    cycle_settings = get_cycle_settings()
    x = range(1, len(list(data.values())[0]) + 1)

    fig, axes = plt.subplots(layout='constrained')
    axes.set_prop_cycle(cycle_settings)
    markers = ['o', 'd', 'v', '>', '*', 'X']
    line_width = 1
    marker_size = 4

    count = 0
    for label, value in data.items():
        if label == "BASS":
            axes.plot(x, data[label], label=label, marker=markers[count], linewidth=5, markersize=marker_size,
                      markevery=5, zorder=11)
        else:
            axes.plot(x, data[label], label=label, marker=markers[count], linewidth=line_width, markersize=marker_size, markevery=5)
        count += 1

    axes.set_xlabel("Training Round", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    if y_lim:
        plt.ylim(y_lim)
    if legend_pos == "in":
        plt.legend(prop=font_settings.get("legend_font"), loc='lower right').set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path, format='pdf')
    else:
        plt.show()
    if legend_pos == "out":
        plot_legend_head(axes, 5, 13.5, 0.85, save_path, plot_size)


def plot_time_acc_ablation(title, bc_ns, bc_FedAvg, ns, in_legend=False, ex_legend=False, save_path=None, plot_size="2"):
    font_settings = get_font_settings(plot_size)
    cycle_settings = get_cycle_settings()
    x = range(1, len(bc_ns) + 1)

    fig, axes = plt.subplots(layout='constrained')
    axes.set_prop_cycle(cycle_settings)

    axes.plot(x, bc_ns, label="BC+NS (BEFS)", linewidth=4.5, zorder=10)
    axes.plot(x, bc_FedAvg, label="BC+FedAvg")
    axes.plot(x, ns, label="NS")

    axes.set_xlabel("Training Round", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    if in_legend:
        plt.legend(prop=font_settings.get("legend_font"), loc='lower right').set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path, format='pdf')
    else:
        plt.show()
    if ex_legend:
        plot_legend_head(axes, 3, 20.6, 0.7, save_path, plot_size)


def plot_time_acc_sensitivity(title, lr_1, lr_01, lr_001, save_path=None, plot_size="2"):
    font_settings = get_font_settings(plot_size)
    cycle_settings = get_cycle_settings()
    x = range(1, len(lr_1) + 1)

    fig, axes = plt.subplots(layout='constrained')
    axes.set_prop_cycle(cycle_settings)

    axes.plot(x, lr_01, label="lr=0.01", linewidth=4.5, zorder=10)
    axes.plot(x, lr_1, label="lr=0.1")
    axes.plot(x, lr_001, label="lr=0.001")

    axes.set_xlabel("Training Round", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.legend(prop=font_settings.get("legend_font"), loc='lower right').set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path, format='pdf')
    else:
        plt.show()


def plot_time_acc_attack(title, data, legend_pos="", save_path=None, plot_size="2"):
    font_settings = get_font_settings(plot_size)
    cycle_settings = get_cycle_settings()
    x = range(1, len(data["fed_sync_sgd"]) + 1)

    fig, axes = plt.subplots(layout='constrained')
    axes.set_prop_cycle(cycle_settings)

    axes.plot(x, data["fed_sync_sgd"], label="BASS", linewidth=4.5, zorder=10)
    axes.plot(x, data["fed_ecsign"], label="EC-SignSGD")
    axes.plot(x, data["fed_efsign"], label="EF-SignSGD")
    axes.plot(x, data["fed_mvsign"], label="MV-SignSGD")
    axes.plot(x, data["fed_rlrsign"], label="RLR-SignSGD")
    axes.plot(x, data["fed_err"], label="ERR-FedAvg")
    axes.plot(x, data["fed_lfr"], label="LFR-FedAvg")
    axes.plot(x, data["fed_trust"], label="FLTrust")
    axes.plot(x, data["fed_fleam"], label="FLEAM")
    axes.plot(x, data["fed_avg"], label="FedAvg")

    axes.set_xlabel("Training Round", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    # plt.tight_layout()
    # plt.xlim(0, xrange)
    if legend_pos == "in":
        plt.legend(prop=font_settings.get("legend_font"), loc='lower right').set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path, format='pdf')
    else:
        plt.show()
    if legend_pos == "out":
        plot_legend_head(axes, 5, 13.4, 1.1, save_path, plot_size)


def plot_time_acc_fall(title, fed_sync_sgd, fed_efsign, legend_pos="", y_lim=None, save_path=None, plot_size="2"):
    font_settings = get_font_settings(plot_size)
    cycle_settings = get_cycle_settings()
    x = range(1, len(fed_sync_sgd) + 1)

    fig, axes = plt.subplots(layout='constrained')
    axes.set_prop_cycle(cycle_settings)

    axes.plot(x, fed_sync_sgd, label="BASS", linewidth=4.5, zorder=10)
    axes.plot(x, fed_efsign, label="ARE")

    axes.set_xlabel("Training Round", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    if y_lim:
        plt.ylim(y_lim)
    if legend_pos == "in":
        plt.legend(prop=font_settings.get("legend_font")).set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path, format='pdf')
    else:
        plt.show()
    if legend_pos == "out":
        plot_legend_head(axes, 2, font_settings.get("fig_width"), 0.85, save_path, plot_size)


def plot_model_size_bar(title, sgd, sign_sgd, save_path=None, plot_size="2"):
    font_settings = get_font_settings(plot_size)
    x = ["CNN-CIFAR10", "CNN-FMNIST", "MLP-FMNIST"]

    fig, axes = plt.subplots(1, 3, layout='constrained')

    width = 0.15  # the width of the bars
    axes[0].bar(0 - width / 2, height=sgd[0], width=width, label="SGD", alpha=.99, hatch='x')
    axes[0].bar(0 + width / 2, height=sign_sgd[0], width=width, label="SignSGD", alpha=.99, hatch='*')

    axes[1].bar(0 - width / 2, height=sgd[1], width=width, label="SGD", alpha=.99, hatch='x')
    axes[1].bar(0 + width / 2, height=sign_sgd[1], width=width, label="SignSGD", alpha=.99, hatch='*')

    axes[2].bar(0 - width / 2, height=sgd[2], width=width, label="SGD", alpha=.99, hatch='x')
    axes[2].bar(0 + width / 2, height=sign_sgd[2], width=width, label="SignSGD", alpha=.99, hatch='*')

    for i in range(len(axes)):
        # Use the pyplot interface to change just one subplot
        plt.sca(axes[i])
        plt.grid()
        plt.xticks([0], [x[i]], **font_settings.get("cs_xy_ticks_font"))
        plt.yticks(**font_settings.get("cs_xy_ticks_font"))

    fig.supxlabel("The Type of Model and Datasets", **font_settings.get("cs_xy_label_font"))
    fig.supylabel("Gradient Size (MB)", **font_settings.get("cs_xy_label_font"))
    fig.suptitle(title, **font_settings.get("cs_title_font"))

    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path, format='pdf')
    else:
        plt.show()

    plot_legend_head(axes[0], 2, font_settings.get("fig_width"), 0.6, save_path, plot_size)


def plot_cache_level(title, data, save_path=None, plot_size="2"):
    font_settings = get_font_settings(plot_size)
    cycle_settings = get_cycle_settings()
    my_colors = plt.get_cmap('tab10').colors
    x = ["60%", "70%", "80%", "90%"]

    fig, ax1 = plt.subplots(layout='constrained')

    width = 0.25  # the width of the bars
    ax1.set_prop_cycle(cycle_settings)
    ax1.bar(range(len(x)), height=data["ACC"], width=width, label="ACC", alpha=.99, hatch='+', color=my_colors[1])
    ax1.set_ylim(86, 90)

    ax2 = ax1.twinx()
    ax2.plot(data["TIME"], label="TIME", marker='*', linewidth=4, markersize=15)
    ax2.set_ylim(5, 40)

    plt.xticks(range(len(x)), x)
    ax1.set_xlabel("Cache Level", **font_settings.get("cs_xy_label_font"))
    ax1.set_ylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))
    ax2.set_ylabel("Average Time (s)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    ax1.set_xticks(range(len(x)), x, **font_settings.get("cs_xy_ticks_font"))
    ax1.set_yticks(np.arange(86, 90.5, 0.5), np.arange(86, 90.5, 0.5), **font_settings.get("cs_xy_ticks_font"))
    ax2.set_yticks(range(5, 45, 5), range(5, 45, 5), **font_settings.get("cs_xy_ticks_font"))

    ax1.legend(prop=font_settings.get("legend_font"), bbox_to_anchor=(0.5, 1.2), loc='upper left').set_zorder(11)
    ax2.legend(prop=font_settings.get("legend_font"), bbox_to_anchor=(0.5, 1.2), loc='upper right').set_zorder(11)

    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path, format='pdf')
    else:
        plt.show()


def plot_time_cost_bar(title, bass, sign_sgd, dtwn, fed_avg, local, save_path=None, plot_size="L"):
    font_settings = get_font_settings(plot_size)
    x = ["Training", "Testing", "Communication", "Waiting", "Total"]

    fig, axes = plt.subplots(layout='constrained')

    width = 0.15  # the width of the bars

    axes.bar([(p - width * 2) for p in range(len(x))], height=bass, width=width, label="BASS", alpha=.99, hatch='x')
    axes.bar([p - width for p in range(len(x))], height=sign_sgd, width=width, label="SignSGD", alpha=.99, hatch='o')
    axes.bar(range(len(x)), height=dtwn, width=width, label="DTWN", alpha=.99, hatch='+')
    axes.bar([p + width for p in range(len(x))], height=fed_avg, width=width, label="FedAvg", alpha=.99, hatch='*')
    axes.bar([(p + width * 2) for p in range(len(x))], height=local, width=width, label="Local", alpha=.99, hatch='/')

    plt.xticks(range(len(x)), x)
    axes.set_xlabel("Stage in Training Rounds", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Average Time (s)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.legend(prop=font_settings.get("legend_font"), loc='upper left').set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path, format='pdf')
    else:
        plt.show()


def plot_sensitivity_bar(title, lr_0_01, lr_0_1, lr_0_001, save_path=None, plot_size="L"):
    font_settings = get_font_settings(plot_size)
    x = ["CNN-CIFAR10", "CNN-FMNIST", "MLP-FMNIST"]

    fig, axes = plt.subplots(1, 3, layout='constrained')

    width = 0.15  # the width of the bars

    axes[0].bar(0 - width, height=lr_0_01[0], width=width, label="lr = 0.01", alpha=.99, hatch='x')
    axes[0].bar(0, height=lr_0_1[0], width=width, label="lr = 0.1", alpha=.99, hatch='*')
    axes[0].bar(0 + width, height=lr_0_001[0], width=width, label="lr = 0.001", alpha=.99, hatch='o')

    axes[1].bar(0 - width, height=lr_0_01[1], width=width, label="lr = 0.01", alpha=.99, hatch='x')
    axes[1].bar(0, height=lr_0_1[1], width=width, label="lr = 0.1", alpha=.99, hatch='*')
    axes[1].bar(0 + width, height=lr_0_001[1], width=width, label="lr = 0.001", alpha=.99, hatch='o')

    axes[2].bar(0 - width, height=lr_0_01[2], width=width, label="lr = 0.01", alpha=.99, hatch='x')
    axes[2].bar(0, height=lr_0_1[2], width=width, label="lr = 0.1", alpha=.99, hatch='*')
    axes[2].bar(0 + width, height=lr_0_001[2], width=width, label="lr = 0.001", alpha=.99, hatch='o')

    for i in range(len(axes)):
        # Use the pyplot interface to change just one subplot
        plt.sca(axes[i])
        plt.grid()
        plt.xticks([0], [x[i]], **font_settings.get("cs_xy_ticks_font"))
        plt.yticks(**font_settings.get("cs_xy_ticks_font"))

    fig.supxlabel("The Type of Model and Datasets", **font_settings.get("cs_xy_label_font"))
    fig.supylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))
    fig.suptitle(title, **font_settings.get("cs_title_font"))

    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path, format='pdf')
    else:
        plt.show()

    plot_legend_head(axes[0], 3, font_settings.get("fig_width"), 0.6, save_path, plot_size)


def plot_legend_head(axes, legend_column, width, height, save_path=None, plot_size="3"):
    font_settings = get_font_settings(plot_size)
    figlegend = pylab.figure(layout='constrained')
    figlegend.legend(axes.get_legend_handles_labels()[0], axes.get_legend_handles_labels()[1],
                     prop=font_settings.get("legend_font"), ncol=legend_column, loc='upper center')
    figlegend.set_size_inches(width, height)
    if save_path:
        save_path = save_path[:-4] + "-legend.pdf"
        figlegend.savefig(save_path, format='pdf')
    else:
        figlegend.show()
