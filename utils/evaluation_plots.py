import matplotlib.pyplot as plt
import numpy as np
from utils import std_error
from scipy import stats

import my_matplotlib_style as ms


# Plotting functions
def plot_residuals(
    pred,
    data,
    range=None,
    variable_names=["m", "pT", "phi", "eta"],
    variable_list=[
        r"$(m_{out} - m_{in}) / m_{in}$",
        r"$(p_{T,out} - p_{T,in}) / p_{T,in}$",
        r"$(\phi_{out} - \phi_{in}) / \phi_{in}$",
        r"$(\eta_{out} - \eta_{in}) / \eta_{in}$",
    ],
    bins=1000,
    save=None,
    title=None,
    figsize=(12, 9),
):
    ms.set_style()
    data[data == 0] = 0.001
    residuals = (pred - data) / data
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Residuals for 4D data")
    for kk, ax in enumerate(axs.flatten()):
        _, _, _ = ax.hist(
            residuals[:, kk], label="Residuals", alpha=0.6, bins=bins, range=(-2, 2)
        )
        if title is None:
            ax.title.set_text("Residuals of %s" % variable_names[kk])
        else:
            ax.title.set_text(title)
        ax.set_xlabel(variable_list[kk])
        ax.set_ylabel("Number of jets")
        std = np.std(residuals[:, kk])
        std_err = std_error(residuals[:, kk])
        mean = np.nanmean(residuals[:, kk])
        sem = stats.sem(residuals[:, kk], nan_policy="omit")
        ms.sciy()

        ax.text(
            0.5,
            0.8,
            "Mean = %f$\pm$%f\n$\sigma$ = %f$\pm$%f" % (mean, sem, std, std_err),
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 10, "edgecolor": "black"},
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=10,
        )

        if save is not None:
            plt.savefig(save + "_%s" % variable_names[kk])
    plt.tight_layout()


def plot_histograms(
    pred,
    data,
    bins,
    same_bin_edges=True,
    variable_list=[r"$m$", r"$p_T$", r"$\phi$", r"$\eta$"],
    variable_names=["m", "pT", "phi", "eta"],
    unit_list=[r"$[\frac{GeV \cdot c^2}{m^2}]$", "[GeV]", "[rad]", "[rad]",],
    title=None,
    figsize=(12, 9),
    labels=["Input", "Output"],
):
    ms.set_style()
    n_bins = bins

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Histograms for 4D data")
    for kk, ax in enumerate(axs.flatten()):
        _, bin_edges, _ = ax.hist(data[:, kk], label=labels[0], alpha=1, bins=n_bins)
        if same_bin_edges:
            n_bins_2 = bin_edges
        else:
            n_bins_2 = bins
        _, _, _ = ax.hist(pred[:, kk], label=labels[1], alpha=0.6, bins=n_bins_2)
        if title is None:
            ax.title.set_text(variable_names[kk])
        else:
            ax.title.set_text(title)
        ax.set_xlabel(variable_list[kk] + " " + unit_list[kk])
        ax.set_ylabel("Number of events")
        ms.sciy()
        ax.legend()
    plt.tight_layout()


def plot_activations(
    learn, figsize=(4, 3), lines=["-", ":"], save=None, linewd=1, fontsz=7
):
    ms.set_style()
    plt.figure(figsize=figsize)
    for i in range(learn.activation_stats.stats.shape[1]):
        thiscol = ms.colorprog(i, learn.activation_stats.stats.shape[1])
        plt.plot(
            learn.activation_stats.stats[0][i],
            linewidth=linewd,
            color=thiscol,
            label=str(learn.activation_stats.modules[i]).split(",")[0],
            linestyle=lines[i % len(lines)],
        )
    plt.title("Weight means")
    plt.legend(fontsize=fontsz)
    plt.xlabel("Mini-batch")
    if save is not None:
        plt.savefig(save + "_means")

    plt.figure(figsize=figsize)
    for i in range(learn.activation_stats.stats.shape[1]):
        thiscol = ms.colorprog(i, learn.activation_stats.stats.shape[1])
        plt.plot(
            learn.activation_stats.stats[1][i],
            linewidth=linewd,
            color=thiscol,
            label=str(learn.activation_stats.modules[i]).split(",")[0],
            linestyle=lines[i % len(lines)],
        )
    plt.title("Weight standard deviations")
    plt.xlabel("Mini-batch")
    plt.legend(fontsize=fontsz)
    if save is not None:
        plt.savefig(save + "_stds")
    plt.tight_layout()
