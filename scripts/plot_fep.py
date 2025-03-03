import json
import os
from io import BytesIO
from pprint import pprint
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import gaussian_kde, kendalltau
from utils import get_predictor

from mc.analyzers import Predictor
from mc.baselines.spatial_score import calculate_score_from_smiles as nsps


def filter_duplicates(smiles_list, properties_list=None):
    if properties_list is None:
        return list(set(smiles_list))

    smiles_set = set()
    smiles_list_upd = []
    properties_list_upd = []

    for smiles, entry in zip(smiles_list, properties_list):
        if smiles in smiles_set:
            continue
        smiles_set.add(smiles)
        smiles_list_upd.append(smiles)
        properties_list_upd.append(entry)
    return smiles_list_upd, properties_list_upd


def inspect_molecules(smiles_list: List[str], mc_aff_list: List[Tuple[float, float]]):
    smiles_upd, mc_aff_list_upd = filter_duplicates(smiles_list, mc_aff_list)

    mols = [Chem.RemoveHs(Chem.MolFromSmiles(smiles)) for smiles in smiles_upd]
    print(len(mols))
    if mc_aff_list_upd is not None:
        for i, mol in enumerate(mols):
            mol.SetProp(
                "legend",
                f"Affinity: {mc_aff_list_upd[i][0]:.2f}, MC: {mc_aff_list_upd[i][1]:.2f}",
            )
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        legends=[mol.GetProp("legend") for mol in mols],
    )
    img.show()


def plot(drugs_info: Dict, system: str, predictor: Predictor):
    smiles_list = [x[0] for x in drugs_info[system]]
    affinities = [x[1] for x in drugs_info[system]]
    mc_list = predictor.predict(smiles_list)
    plt.scatter(mc_list, affinities)
    plt.xlabel("Molecular Complexity")
    plt.ylabel("Affinity")
    plt.title(f"{system}")
    plt.show()


def match_mols_with_scaffold(smiles_list, scaffold):
    mols = [Chem.RemoveHs(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]
    scaffold_mol = Chem.RemoveHs(Chem.MolFromSmiles(scaffold))
    # matches = [mol.GetSubstructMatch(scaffold_mol) for mol in mols]
    smiles_matched = []
    for i, mol in enumerate(mols):
        if mol.HasSubstructMatch(scaffold_mol):
            smiles_matched.append(i)

    return set(smiles_matched)


def plot_molecule_data(smiles_list, mc_list, affinities):
    palette = plt.cm.Greens(np.linspace(0, 1, 20))
    mpl.rcParams.update({"axes.spines.right": True})
    # Validate input lengths
    if not (len(smiles_list) == len(mc_list) == len(affinities)):
        raise ValueError("All input lists must have the same length")

    d = {"smiles": smiles_list, "mc": mc_list, "aff": affinities}
    # pprint(smiles_list)
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        smiles_chemdraw = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        print(smiles_chemdraw)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 3))

    # Plotting mc_list
    # empty ticks
    ax1.set_xticks([])
    ax1.set_ylabel("Molecular Complexity")
    ax1.plot(
        range(len(d["mc"])),
        d["mc"],
        color=palette[8],
        label="MC List",
        marker="o",
        markersize=5,
    )
    ax1.tick_params(axis="y", labelcolor=palette[8])

    # Creating a second Y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Binding Affinity, kcal/mol")
    ax2.plot(
        range(len(d["aff"])),
        d["aff"],
        color=palette[15],
        label="Affinities",
        marker="o",
        markersize=5,
    )
    ax2.tick_params(axis="y", labelcolor=palette[15])
    # color the axis
    # fig.tight_layout()
    plt.savefig("plots/lead_opt/scaffold1_double_axis.pdf")


def plot_with_custom_markers(x, y, colors, sizes, labels, title=""):
    palette = plt.cm.Greens(np.linspace(0, 1, 20))
    gray_palette = plt.cm.Greys(np.linspace(0, 1, 20))
    fig, ax = plt.subplots(figsize=(4, 3))
    map_color = {
        "blue": palette[10],
        "red": gray_palette[3],
        "both": gray_palette[7],
    }
    legend_labels = {
        "blue": "Scaffold 1",
        "red": "Scaffold 2",
        "both": "Both Scaffolds",
        "best_binder": "Best binder",
        "hit": "Hit",
    }

    handles = {}

    for xi, yi, ci, si, li in zip(x, y, colors, sizes, labels):
        if ci == "both":
            sc = ax.scatter(
                [xi],
                [yi],
                c=[map_color[ci]],
                s=si,
                alpha=0.9,
                edgecolor="w",
                # edge color alpha
            )
        else:
            sc = ax.scatter(
                [xi],
                [yi],
                c=[map_color[ci]],
                s=si,
                alpha=0.9,
                edgecolor="w",
            )

        if legend_labels[ci] not in handles:
            handles[legend_labels[ci]] = sc

    # Manually add legend entries for best binder and hit if they exist
    if "Best binder" in labels:
        idx = labels.index("Best binder")
        print(x[idx], y[idx])
        plt.annotate(
            "Best binder",
            (x[idx], y[idx] - 0.6),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=6,
            bbox=dict(boxstyle="round", fc="w"),
        )

    if "Hit" in labels:
        idx = labels.index("Hit")
        plt.annotate(
            "Hit",
            (x[idx] + 0.05, y[idx] - 0.18),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=6,
            bbox=dict(boxstyle="round", fc="w"),
        )

    ax.set_xlabel("Molecular Complexity")
    # y axes limit is -11
    ax.set_ylim(-11.5, -6.5)

    ax.set_ylabel("Binding Affinity, kcal/mol")
    # ax.legend(handles=handles.values(), labels=handles.keys())
    ordered_labels = ["Scaffold 1", "Scaffold 2", "Both Scaffolds"]
    ordered_handles = [handles[label] for label in ordered_labels]

    ax.legend(handles=ordered_handles, labels=ordered_labels)

    ax.set_title(title)


def plot_kendall_tau_kde(drugs_info: Dict):
    palette = plt.cm.Greens(np.linspace(0, 1, 20))
    color_usps = palette[10]
    color_mc = palette[15]
    systems = set([k for k, v in drugs_info.items() if len(v) > 10])
    drugs_info = {k: v for k, v in drugs_info.items() if k in systems}
    predictor = get_predictor()
    corr_per_system = {"mc": [], "usps": []}
    for method in ["mc", "usps"]:
        for system, data in drugs_info.items():
            smiles_list = [x[0] for x in data]
            affinities = [x[1] for x in data]
            if method == "mc":
                mc_list = predictor.predict(smiles_list)
            else:
                mc_list = [nsps(smiles) for smiles in smiles_list]
            tau, _ = kendalltau(mc_list, affinities)
            corr_per_system[method].append(tau)

    # Calculate means
    mean_mc = np.mean(corr_per_system["mc"])
    mean_usps = np.mean(corr_per_system["usps"])

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(5, 5))

    # Plot KDE for both methods on the first subplot
    kde_mc = gaussian_kde(corr_per_system["mc"], bw_method=0.5)
    kde_usps = gaussian_kde(corr_per_system["usps"], bw_method=0.5)
    x_range = np.linspace(-1, 1, 100)

    # Compute KDE values
    kde_mc_values = kde_mc(x_range)
    kde_usps_values = kde_usps(x_range)

    # Plot KDE
    (line1,) = axes[0].plot(
        x_range, kde_mc_values, label="Molecular Complexity", color=color_mc
    )
    (line2,) = axes[0].plot(x_range, kde_usps_values, label="nSPS", color=color_usps)

    kde_mc_value = kde_mc(mean_mc)[0]
    kde_usps_value = kde_usps(mean_usps)[0]

    (line3,) = axes[0].plot(
        [mean_mc, mean_mc],
        [0, kde_mc_value],
        color=color_mc,
        linestyle="--",
        linewidth=1,
    )
    (line4,) = axes[0].plot(
        [mean_usps, mean_usps],
        [0, kde_usps_value],
        color=color_usps,
        linestyle="--",
        linewidth=1,
    )

    # axes[0].set_xlabel("Correlation (Kendall's Tau)")
    axes[0].set_ylabel("Density")
    axes[0].set_ylim(0, max(max(kde_mc_values), max(kde_usps_values)) * 1.1)

    # Plot CDF for both methods on the second subplot
    sorted_mc = np.sort(corr_per_system["mc"])
    sorted_usps = np.sort(corr_per_system["usps"])

    ecdf_mc = np.arange(1, len(sorted_mc) + 1) / len(sorted_mc)
    ecdf_usps = np.arange(1, len(sorted_usps) + 1) / len(sorted_usps)

    (line5,) = axes[1].step(sorted_mc, ecdf_mc, color=color_mc)
    (line6,) = axes[1].step(sorted_usps, ecdf_usps, color=color_usps)

    # Calculate CDF values at the means
    cdf_mc_value = np.searchsorted(sorted_mc, mean_mc, side="right") / len(sorted_mc)
    cdf_usps_value = np.searchsorted(sorted_usps, mean_usps, side="right") / len(
        sorted_usps
    )

    (line7,) = axes[1].plot(
        [mean_mc, mean_mc],
        [0, cdf_mc_value],
        color=color_mc,
        linestyle="--",
        linewidth=1,
    )
    (line8,) = axes[1].plot(
        [mean_usps, mean_usps],
        [0, cdf_usps_value],
        color=color_usps,
        linestyle="--",
        linewidth=1,
    )

    # axes[1].set_xlabel("Kendall Tau Coefficient")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_ylim(0, 1.01)

    # Create a single legend for the entire figure
    handles = [line1, line2, line3, line4, line5, line6, line7, line8]
    labels = [line.get_label() for line in handles]
    fig.legend(
        handles, labels, loc="upper center", ncol=1, bbox_to_anchor=(0.8, 1), fontsize=8
    )
    fig.supxlabel("Kendall Tau Coefficient", fontsize=8)

    plt.tight_layout()
    if not os.path.exists("plots/lead_opt"):
        os.makedirs("plots/lead_opt")
    plt.savefig("plots/lead_opt/kendall_tau_kde.pdf", dpi=600)


def analyze_pfkfb3(drugs_info: Dict, scaffolds_to_plot="all"):
    hit_smiles = "[H]c1nc2c([H])c(N([H])c3c([H])c([H])c([H])c([H])c3[H])c([H])c(-c3c([H])c([H])c4c([H])c([H])n(C([H])([H])[H])c4c3[H])c2nc1[H]"
    # best_binder = "[H]c1nc2c([H])c(N([H])c3c([H])nc([H])c([H])c3S(=O)(=O)C([H])([H])[H])c([H])c(-c3c([H])c([H])c4sc([H])c(C([H])([H])[H])c4c3[H])c2nc1[H]"
    smiles_list = [x[0] for x in drugs_info["pfkfb3"]]
    affinities = [x[1] for x in drugs_info["pfkfb3"]]
    scaffold1 = "CN1C=CC2=C1C=C(C3=C(N=CC=N4)C4=CC=C3)C=C2"
    scaffold2 = "O=S(C(C=CN=C1)=C1NC2=CC3=C(C=C2)N=CC=N3)(C)=O"

    smiles_list, affinities = filter_duplicates(smiles_list, affinities)
    best_binder = smiles_list[affinities.index(min(affinities))]
    smiles_chemdraw = Chem.MolToSmiles(
        Chem.MolFromSmiles(best_binder), isomericSmiles=True, kekuleSmiles=True
    )
    print(smiles_chemdraw)
    predictor = get_predictor()
    mc_list = predictor.predict(smiles_list)
    matched_smiles_idx1 = match_mols_with_scaffold(smiles_list, scaffold1)
    matched_smiles_idx2 = match_mols_with_scaffold(smiles_list, scaffold2)

    colors = ["gray"] * len(smiles_list)
    sizes = [40] * len(smiles_list)
    labels = [""] * len(smiles_list)

    for idx in matched_smiles_idx1:
        colors[idx] = "blue"
    for idx in matched_smiles_idx2:
        if colors[idx] == "blue":
            colors[idx] = "both"
        else:
            colors[idx] = "red"

    # Label hit and best binder according to the same rules but bigger size
    best_binder_idx = smiles_list.index(best_binder)
    hit_idx = smiles_list.index(hit_smiles)

    if colors[best_binder_idx] == "blue":
        best_binder_color = "blue"
    elif colors[best_binder_idx] == "red":
        best_binder_color = "red"
    else:
        best_binder_color = "both"

    if colors[hit_idx] == "blue":
        hit_color = "blue"
    elif colors[hit_idx] == "red":
        hit_color = "red"
    else:
        hit_color = "both"

    colors[best_binder_idx] = best_binder_color
    colors[hit_idx] = hit_color
    sizes[best_binder_idx] = 80
    sizes[hit_idx] = 80
    labels[best_binder_idx] = "Best binder"
    labels[hit_idx] = "Hit"

    if scaffolds_to_plot == "all":
        plot_with_custom_markers(mc_list, affinities, colors, sizes, labels)
    elif scaffolds_to_plot == "scaffold1":
        mc_list = [
            mc_list[i] for i in matched_smiles_idx1 if i not in matched_smiles_idx2
        ]
        affinities = [
            affinities[i] for i in matched_smiles_idx1 if i not in matched_smiles_idx2
        ]
        sizes = [sizes[i] for i in matched_smiles_idx1 if i not in matched_smiles_idx2]
        colors = [
            colors[i] for i in matched_smiles_idx1 if i not in matched_smiles_idx2
        ]
        labels = [
            labels[i] for i in matched_smiles_idx1 if i not in matched_smiles_idx2
        ]
        smiles_list = [
            smiles_list[i] for i in matched_smiles_idx1 if i not in matched_smiles_idx2
        ]
        zipped = sorted(
            list(zip(smiles_list, mc_list, affinities)),
            key=lambda x: x[-1],
            # reverse=True,
        )
        smiles_list, mc_list, affinities = zip(*zipped)

        d = {"smiles": smiles_list, "mc": mc_list, "aff": affinities}
        plot_molecule_data(d["smiles"], d["mc"], d["aff"])

    elif scaffolds_to_plot == "scaffold2":
        scaffold2_mc = [
            mc_list[i] for i in matched_smiles_idx2 if i not in matched_smiles_idx1
        ]
        scaffold2_aff = [
            affinities[i] for i in matched_smiles_idx2 if i not in matched_smiles_idx1
        ]
        mc_list = scaffold2_mc
        affinities = scaffold2_aff
        sizes = [sizes[i] for i in matched_smiles_idx2 if i not in matched_smiles_idx1]
        colors = [
            colors[i] for i in matched_smiles_idx2 if i not in matched_smiles_idx1
        ]
        labels = [
            labels[i] for i in matched_smiles_idx2 if i not in matched_smiles_idx1
        ]
    else:
        both_scaffold_mc = [
            mc_list[i] for i in matched_smiles_idx1 if i in matched_smiles_idx2
        ]
        both_scaffold_aff = [
            affinities[i] for i in matched_smiles_idx1 if i in matched_smiles_idx2
        ]
        mc_list = both_scaffold_mc
        affinities = both_scaffold_aff
        sizes = [sizes[i] for i in matched_smiles_idx1 if i in matched_smiles_idx2]
        colors = [colors[i] for i in matched_smiles_idx1 if i in matched_smiles_idx2]
        labels = [labels[i] for i in matched_smiles_idx1 if i in matched_smiles_idx2]

    # plot_with_custom_markers(mc_list, affinities, colors, sizes, labels)
    if not os.path.exists("plots/lead_opt"):
        os.makedirs("plots/lead_opt")
    plt.savefig(f"plots/lead_opt/pfkfb3_{scaffolds_to_plot}.pdf", dpi=600)


def main():
    with open("data/fep/fep_smiles.json", "r") as f:
        fep_drugs = json.load(f)
    analyze_pfkfb3(fep_drugs, "all")
    plot_kendall_tau_kde(fep_drugs)


if __name__ == "__main__":
    main()
