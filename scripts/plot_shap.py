import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRanker, Pool

from mc.data import build_pools
from utils import get_predictor

name_mapping = {
    "weight": "molecular weight",
    "arom_cycles": "# arom. cycles",
    "tpsa": "TPSA",
    "arom_heterocycles": "# arom. heterocycles",
    "num_heteroatoms": "# heteroatoms",
    "num_of_atoms": "# atoms",
    "aliph_heterocycles": "# aliph. heterocycles",
}


def truncate_colormap(cmapIn="jet", minval=0.0, maxval=1.0, n=100):
    """truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)"""
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)),
    )

    return new_cmap


def plot_shap(ranker: CatBoostRanker, pool: Pool):
    mpl.rcParams.update(
        {
            "font.size": 10,  # Default font size for text
            "axes.labelsize": 10,  # Font size for axis labels
            "axes.linewidth": 1,  # Width of the axes lines
            "legend.fontsize": 10,  # Font size for legend
            "xtick.labelsize": 10,  # Font size for X-axis tick labels
            "ytick.labelsize": 8,  # Font size for Y-axis tick labels
        }
    )
    shap_values = ranker.get_feature_importance(data=pool, type="ShapValues")
    shap.summary_plot(
        shap_values[:, :-1],
        features=pool.get_features(),
        feature_names=[name_mapping.get(f, f) for f in ranker.feature_names_],
        color=truncate_colormap("Greens", 0.2, 1, 100),
        show=False,
        max_display=7,
        plot_size=(4, 3.5),
    )
    newcmp = truncate_colormap("Greens", 0.2, 1, 100)
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                fcc.set_cmap(newcmp)
    plt.xlabel("SHAP value", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join("plots", "shap_plot_new.pdf"),
        dpi=600,
        bbox_inches="tight",
    )


def compute_complexities(predictor):
    ineleganolide = "O=C1C[C@H](C(C)=C)C[C@]2([H])[C@]1([H])[C@@H](O[C@]3(C)C[C@@H]4O5)C([C@@]3([H])[C@@]4([H])[C@]2([H])C5=O)=O"
    iso1 = "O=C6C[C@H](C(C)=C)CC7=C6C(C([C@@]([C@@H](C)CC8)([H])[C@@]8([H])[C@]7([H])C(O)=O)=O)=O"
    iso2 = "OC9=C(O)C(C(C)=C)=CC([C@](C(O)=O)([H])C%10=C([C@@H](C)CC%10)C(C)=O)=C9"
    mcs = predictor.predict([ineleganolide, iso1, iso2])
    print(mcs)


def main():
    if not os.path.exists("plots"):
        os.makedirs("plots")
    predictor = get_predictor()
    df = pd.read_csv("data/data_processed.csv")
    df = df.drop(columns=["num_of_heavy_atoms"])
    _, test_pool = build_pools(df, (0.7, 0.1, 0.2))
    plot_shap(predictor.ranker, test_pool)
    compute_complexities(predictor)


if __name__ == "__main__":
    main()
