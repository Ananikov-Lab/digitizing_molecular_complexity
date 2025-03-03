import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from utils import get_predictor, load_file
from mc.utils import valid_mol


def plot_mc_distributions(
    datasets,
    labels=None,
    output_path="plots/mc_comparison.pdf",
    cmap="Blues",
):
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.linewidth": 1,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(datasets))]
    if len(labels) != len(datasets):
        raise ValueError("`labels` length must match `datasets` length.")

    n = len(datasets)
    color_array = plt.get_cmap(cmap)(np.linspace(0.3, 0.9, n))

    plt.figure(figsize=(6, 3), dpi=600)
    ax = plt.gca()

    for mc_vals, label, color in zip(datasets, labels, color_array):
        mc_vals = np.array(mc_vals)

        kde = gaussian_kde(mc_vals)
        x_grid = np.linspace(0, 10, 300)
        y_grid = kde(x_grid)

        ax.plot(x_grid, y_grid, color=color, linewidth=1.5, label=label)

        ax.fill_between(x_grid, y_grid, color=color, alpha=0.3)

    ax.set_xlim(0, 10)

    ax.set_xlabel("Molecular Complexity")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=600)
    plt.close()
    return output_path


def filter_valid_mols(smiles):
    return [smi for smi in smiles if valid_mol(smi)]


def main():
    if not os.path.exists("plots"):
        os.makedirs("plots")
    predictor = get_predictor()

    qm9 = load_file("data/tdc/qm9.csv", "csv")
    tox21 = load_file("data/tdc/tox21.csv", "csv")
    hiv = load_file("data/tdc/hiv.csv", "csv")

    qm9_smiles = filter_valid_mols(qm9["smiles"].values.tolist()[:10_0])
    tox21_smiles = filter_valid_mols(tox21["Drug"].values.tolist()[:10_0])
    hiv_smiles = filter_valid_mols(hiv["Drug"].values.tolist()[:10_0])

    qm9_mc = predictor.predict(qm9_smiles)
    tox21_mc = predictor.predict(tox21_smiles)
    hiv_mc = predictor.predict(hiv_smiles)

    for output_path in [
        "plots/benchmarks_mc_distribution.pdf",
        "plots/benchmarks_mc_distribution.png",
    ]:
        plot_mc_distributions(
            [qm9_mc, tox21_mc, hiv_mc],
            labels=["QM9", "Tox21", "HIV"],
            output_path=output_path,
            cmap="Greens",
        )


if __name__ == "__main__":
    main()
