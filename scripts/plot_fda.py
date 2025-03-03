import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
from rdkit import Chem
from scipy import stats
from sklearn.mixture import GaussianMixture
from utils import get_predictor

from mc.utils import hide_frame, valid_mol


def process_json(path="data/fda/fda_drugs.json") -> List[Tuple[str]]:
    with open(path, "r") as f:
        fda_drugs = json.load(f)
    data = []
    for drug_name, info in fda_drugs.items():
        smiles, year = info["smiles"], info["year_approved"]
        if valid_mol(smiles):
            data.append((drug_name, smiles, year))
    with open("data/fda/fda_drugs_info.json", "w") as f:
        json.dump(data, f)
    return data


def open_processed_json(path="data/fda/fda_drugs_info.json") -> List[Tuple[str]]:
    with open(path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} drugs")
    return data


def plot_hist_and_statistical_test(info_per_year: Dict[int, List[float]]) -> None:
    plt.figure(figsize=(8, 6))
    colors = plt.cm.Blues(np.linspace(0, 1, 10))

    means = {year: np.mean(mc) for year, mc in info_per_year.items()}
    stds = {year: np.std(mc) for year, mc in info_per_year.items()}

    plt.bar(info_per_year.keys(), list(means.values()), color=colors[5], alpha=0.6)

    ci95 = [
        1.96 * std / np.sqrt(len(mc))
        for std, mc in zip(stds.values(), info_per_year.values())
    ]

    plt.errorbar(
        list(info_per_year.keys()),
        list(means.values()),
        yerr=ci95,
        fmt="o",
        color=colors[7],
        # label='Error'
    )

    plt.xlabel("Year", fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel("MC", fontsize=14)
    plt.yticks(fontsize=12)

    test_result = mk.original_test(list(means.values()))
    test_name = "Mann-Kendall"
    trend = test_result.trend
    p_value = test_result.p

    p_value_sci = format(p_value, ".2e")
    parts = p_value_sci.split("e")
    base = parts[0]
    exponent = int(parts[1].lstrip("+"))
    p_value_latex = f"{base} \\times 10^{{{exponent}}}"

    plt.annotate(
        f"{test_name} test:\nTrend: {trend},\np-value: ${p_value_latex}$",
        xy=(0.05, 0.9),
        xycoords="axes fraction",
        fontsize=14,
        bbox=dict(boxstyle="round", fc="w"),
    )

    years = np.array(list(info_per_year.keys()))
    means_values = np.array(list(means.values()))
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, means_values)
    plt.plot(
        years,
        intercept + slope * years,
        "r",
        color=colors[-1],
    )

    plt.legend()
    hide_frame(plt)

    plt.tight_layout()
    plt.savefig("plots/mc_per_year_with_stat_test.png", dpi=600)


def plot_2d_histogram_by_year(info_per_year, smiles_per_year, mc_bin_count=24):
    mpl.rcParams.update({"xtick.direction": "out", "ytick.direction": "out"})
    fig, ax = plt.subplots(figsize=(6, 3))

    chunked_info = defaultdict(list)
    smiles_info = defaultdict(list)
    for (year, values), (_, smiles) in zip(
        info_per_year.items(), smiles_per_year.items()
    ):
        chunk = (year // 5) * 5
        for value, s in zip(values, smiles):
            chunked_info[chunk].append(value)
            smiles_info[chunk].append(s)

    sorted_chunked_info = dict(sorted(chunked_info.items()))

    all_values = []
    for values in sorted_chunked_info.values():
        all_values.extend(values)

    min_value, max_value = min(all_values), max(all_values)

    total_hist = None
    xedges_list, yedges_list = None, None

    for chunk, values in sorted_chunked_info.items():
        hist, xedges, yedges = np.histogram2d(
            [chunk] * len(values),
            values,
            bins=[
                range(
                    min(sorted_chunked_info.keys()),
                    max(sorted_chunked_info.keys()) + 6,
                    5,
                ),
                np.linspace(min_value, max_value, mc_bin_count),
            ],
        )

        hist = hist / hist.sum()

        if total_hist is None:
            total_hist = hist
            xedges_list = xedges
            yedges_list = yedges
        else:
            total_hist += hist

    x_pos, y_pos = np.meshgrid(
        xedges_list[:-1] + (xedges_list[1] - xedges_list[0]) / 2, yedges_list[:-1]
    )

    color_mesh = ax.pcolormesh(
        x_pos, y_pos, total_hist.T, cmap="Oranges", shading="auto"
    )

    gmm_data = fit_gmm_to_chunks(info_per_year)
    higher_means = []
    lower_means = []
    chunk_centers = []

    for chunk, vals in gmm_data.items():
        x_center = chunk + 2.5
        means_sorted = sorted(vals["means"])
        chunk_centers.append(x_center)
        lower_means.append(means_sorted[0])
        higher_means.append(means_sorted[1])
        ax.scatter(
            [x_center] * len(vals["means"]),
            vals["means"],
            color=plt.cm.Greens(0.5),
            label=f"{chunk}-{chunk + 5}",
            s=80 * np.array(vals["weights"]),
            edgecolor="k",
            alpha=0.9,
        )

    ax.plot(
        chunk_centers,
        higher_means,
        linestyle="--",
        color=plt.cm.Greens(0.5),
        linewidth=0.5,
    )
    ax.plot(
        chunk_centers,
        lower_means,
        linestyle="--",
        color=plt.cm.Greens(0.8),
        linewidth=0.5,
    )

    median_values = []
    years = []
    for chunk, values in sorted_chunked_info.items():
        median_index = np.argsort(values)[len(values) // 2]
        median_value = values[median_index]
        median_smiles = smiles_info[chunk][median_index]
        x_center = chunk + 2.5
        years.append(x_center)
        median_values.append(median_value)
        mol = Chem.MolFromSmiles(median_smiles)
        smiles_chemdraw = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)

        print(f"{chunk}-{chunk + 5}: {median_value} {smiles_chemdraw}")

        ax.scatter(
            x_center,
            median_value,
            color=plt.cm.Oranges(0.1),
            edgecolor="black",
            s=20,
            zorder=5,
            alpha=0.9,
            label="Median" if chunk == min(sorted_chunked_info.keys()) else None,
        )

    ax.plot(
        years,
        median_values,
        linestyle="--",
        color=plt.cm.Oranges(0.1),
        linewidth=0.5,
    )

    ax.set_xlabel("Year Intervals")
    ax.set_ylabel("Molecular Complexity")
    plt.colorbar(color_mesh, ax=ax, label="Percentage")

    plt.xticks(
        list(
            range(
                min(sorted_chunked_info.keys()), max(sorted_chunked_info.keys()) + 1, 5
            )
        ),
    )

    plt.savefig("plots/mc_per_year_2d.pdf", dpi=600)
    plt.close(fig)


def fit_gmm_to_chunks(info_per_year, n_components=2):
    chunked_info = defaultdict(list)
    for year, values in info_per_year.items():
        chunk = (year // 5) * 5
        chunked_info[chunk].extend(values)

    sorted_chunked_info = dict(sorted(chunked_info.items()))

    gmm_parameters_per_chunk = {}

    for chunk, values in sorted_chunked_info.items():
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        values_array = np.array(values).reshape(-1, 1)
        gmm.fit(values_array)

        gmm_parameters_per_chunk[chunk] = {
            "means": gmm.means_.flatten().tolist(),
            "weights": gmm.weights_.flatten().tolist(),
        }

    return gmm_parameters_per_chunk


def statistical_test(info_per_year: Dict[int, List[float]]) -> None:
    means = {year: np.mean(mc) for year, mc in info_per_year.items()}
    test = mk.original_test(list(means.values()))
    print(test)


def plot_number_approved_drugs():
    df = pd.read_csv("data/fda/fda_drugs.csv")
    years = df["Approval Year"].value_counts().sort_index()

    colors = plt.cm.Blues(np.linspace(0, 1, 10))
    plt.bar(years.index, years.values, color=colors[0])
    plt.xlabel("Year")
    plt.ylabel("Number of drugs approved")
    plt.title("Number of drugs approved per year")
    plt.tight_layout()

    plt.savefig("plots/number_approved_drugs.png", dpi=600)
    plt.close()


def analyze_mc_per_year(mc_per_year: Dict[int, List[float]]) -> None:
    plt.figure(figsize=(8, 6))
    data = fit_gmm_to_chunks(mc_per_year)
    for year, vals in data.items():
        plt.scatter(
            vals["means"],
            vals["weights"],
            color=plt.cm.Blues((year - 1985) / 35),
            label=f"{year}-{year + 5}",
            s=100,
        )

    plt.xlabel("Mean", fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel("Weight", fontsize=14)
    plt.yticks(fontsize=12)
    # plt.title("Parameters of Gaussian Mixture Model with 2 components", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join("plots", "gmm_parameters_per_year.png"),
        dpi=600,
    )


def main():
    if not os.path.exists("plots"):
        os.makedirs("plots")
    predictor = get_predictor()
    data = open_processed_json()
    mc_list = predictor.predict([x[1] for x in data])

    info_per_year = {}
    for i, (drug, smiles, year) in enumerate(data):
        info_per_year[year] = info_per_year.get(year, [])
        info_per_year[year].append((drug, smiles, mc_list[i]))

    mc_per_year = {}
    smiles_per_year = {}

    for year, info in info_per_year.items():
        mc_values = [x[2] for x in info]
        mc_per_year[year] = mc_values
        smiles_per_year[year] = [x[1] for x in info]

    smiles_w_median_mc = {}
    for year, info in info_per_year.items():
        mean = np.mean(mc_per_year[year])
        info_sorted = sorted(info, key=lambda x: abs(x[2] - mean))
        smiles_w_median_mc[year] = info_sorted[0]
    with open("data/fda/mean_fda.json", "w") as f:
        json.dump(smiles_w_median_mc, f)
    with open("data/fda/mc_per_year.json", "w") as f:
        json.dump(mc_per_year, f)

    analyze_mc_per_year(mc_per_year)
    plot_hist_and_statistical_test(mc_per_year)
    plot_2d_histogram_by_year(mc_per_year, smiles_per_year)

    statistical_test(mc_per_year)
    plot_number_approved_drugs()


if __name__ == "__main__":
    main()
