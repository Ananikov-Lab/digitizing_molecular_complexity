import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw

from mc.analyzers import Predictor
from mc.reaction_processing import ReactionsManager
from mc.utils import hide_frame
from utils import get_predictor


def analyze_synthesis(rm, predictor: Predictor, mode="max", with_smiles=False):
    complexities = []
    if mode == "max":
        for i in range(len(rm)):
            reagents = rm.get_reagents(i)
            tmp = []
            if any("[x]" in _ for _ in reagents):
                complexities.append(None)
                continue

            for reagent in reagents:
                try:
                    pred = predictor._get_prediction_scaled(reagent)
                except:
                    print(reagent)
                    pred = 0
                tmp.append(pred)
                print(f"Predicted score for {reagent}: {pred}")
            if with_smiles:
                complexities.append((max(tmp), reagents[tmp.index(max(tmp))]))
            else:
                complexities.append(max(tmp))

        final_product = rm.get_products(len(rm) - 1)[0]
        print(final_product)
        pred = predictor._get_prediction_scaled(final_product)
        if with_smiles:
            complexities.append((pred, final_product))
        else:
            complexities.append(pred)

    return complexities


def analyze_all_synthesis(predictor, path, sep=">>", with_smiles=False):
    all_complexities = []
    for i, txt in enumerate(os.listdir(path=path)):
        # print(txt)
        # continue
        if not txt.endswith(".txt"):
            continue
        print("Processing", txt)
        rm = ReactionsManager(os.path.join(path, txt), sep=sep)
        all_complexities.append(
            (analyze_synthesis(rm, predictor, with_smiles=with_smiles), rm)
        )
    return all_complexities


def mc_in_total_synthesis(mc_list, rm):
    plt.figure(figsize=(len(rm), 20))
    palette = sns.color_palette("mako", n_colors=6)
    x = np.arange(0, len(rm) + 1)
    y = [round(c, 1) for c in mc_list]

    plt.scatter(x, y, s=100, color=palette[3])
    plt.plot(x, y, linestyle="dashed", color=palette[-1])

    plt.xticks(x, fontsize=14)
    plt.yticks(np.arange(0, 11), fontsize=14)

    for i, txt in enumerate(y):
        plt.text(x[i] - 0.6, y[i] - 0.2, txt, color=palette[1], fontsize=12)

    plt.grid()
    plt.savefig(f"{rm.get_title()}.pdf")


def transform_title(title):
    """
    Woodward_1954
    """
    idx = len(title)
    for i, letter in enumerate(title):
        if letter.isdigit():
            idx = i
            break
    if title[idx - 1] == "_":
        name = title[: idx - 1]
    else:
        name = title[:idx]
    name = name.capitalize()
    year = title[idx:]
    return f"{name} {year}"


def plot_synthesis(
    all_complexities,
    path_to_save,
    add_captions=False,
    height=4,
    target_molecule="Strychnine",
    y_min=2,
    y_max=9,
    fontsize=8,
    use_width=True,
    width=2,
    legend_loc="lower right",
):
    x_max = max(len(pair[-1]) for pair in all_complexities)
    # if use_width:
    #     plt.figure(figsize=(x_max // 4.5, height))
    # else:
    plt.figure(figsize=(5, 3))
    colors = plt.cm.Set2(np.linspace(0, 1, x_max))
    step = x_max // len(all_complexities)
    for i, (mc_list, rm) in enumerate(all_complexities):
        color = colors[i * step]
        x = np.arange(x_max - len(rm), x_max + 1)
        y = [mc if mc is not None else 9.9 for mc in mc_list]
        bad_ids = []
        for i, mc in enumerate(mc_list):
            if mc is None:
                bad_ids.append(i)

        # start
        plt.scatter(x[0], y[0], s=60, color=color, marker="x", linewidths=3)
        # intermediates
        plt.scatter(x[1:-1], y[1:-1], s=35, color=color, alpha=0.7, edgecolors="none")
        plt.scatter(np.array(x)[bad_ids], np.array(y)[bad_ids], s=35, color="red")

        plt.plot(
            x,
            y,
            linestyle="dashed",
            color=color,
            label=transform_title(rm.get_title()),
            linewidth=1,
        )

        if add_captions:
            for j, txt in enumerate(y):
                plt.text(x[j] - 0.6, y[j] - 0.2, txt, color=color)

    tm_color = colors[-1]
    plt.axhline(y=mc_list[-1], linestyle="dotted", color=tm_color, linewidth=2)
    adjust_text(
        [
            plt.text(
                0.1,
                mc_list[-1] - 0.2,
                f"MC of {target_molecule}",
                # fontsize=fontsize,
                color=tm_color,
            )
        ]
    )
    # plt.title(f"Synthesis of {target_molecule}", fontsize=8)
    # TM
    plt.scatter(x[-1], y[-1], s=200, color=tm_color, marker="*", linewidth=1)

    plt.xticks(np.arange(0, x_max + 1), labels=[])
    plt.yticks(list(range(0, 11)))

    plt.ylim(y_min, y_max)
    # plt.legend(fontsize=8, loc=legend_loc)
    # plt.ylabel("Molecular Complexity", fontsize=12)
    plt.legend(loc=legend_loc)
    plt.ylabel("Molecular Complexity")
    plt.tight_layout()
    hide_frame(plt)
    plt.savefig(
        os.path.join(path_to_save, "syntheses", f"{target_molecule}.pdf"), dpi=600
    )


def prepare_si(complexities: List[Tuple[float, str]], name: str) -> None:
    smiles = [c[1] for c in complexities]
    mcs = [c[0] for c in complexities]
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    for mol, mc in zip(mols, mcs):
        mol.SetProp("_mc", str(mc))

    # Parameters for layout
    molsPerRow = 5
    subImgSize = (200, 120)  # Adjust if necessary
    arrowSize = (50, subImgSize[1])
    padding = 10  # Space between rows
    textHeight = 20  # Space for text below each molecule

    # Calculate full image size
    numRows = (len(mols) + molsPerRow - 1) // molsPerRow
    totalWidth = molsPerRow * subImgSize[0] + (molsPerRow - 1) * arrowSize[0]
    totalHeight = numRows * (subImgSize[1] + textHeight) + (numRows - 1) * padding

    # Create a blank canvas
    fullImage = Image.new("RGB", (totalWidth, totalHeight), (255, 255, 255))
    draw = ImageDraw.Draw(fullImage)

    # Attempt to use a default font
    try:
        font = ImageFont.truetype("Arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for i, (mol, mc) in enumerate(zip(mols, mcs)):
        row, col = divmod(i, molsPerRow)

        # Calculate offsets
        x_offset = col * (subImgSize[0] + arrowSize[0])
        y_offset = row * (subImgSize[1] + padding + textHeight)

        # Render molecule to an image
        molImg = Draw.MolToImage(mol, size=subImgSize)

        # Paste molecule image

        fullImage.paste(molImg, (x_offset, y_offset))

        # Draw complexity below the molecule image
        complexity_text = f"{mc}"
        text_x = x_offset
        text_y = y_offset + subImgSize[1]
        draw.text((text_x, text_y), complexity_text, fill=(0, 0, 0), font=font)

        # Draw arrow if not the last molecule in a row and not the last one overall
        if col < molsPerRow - 1 and i < len(mols) - 1:
            arrow_start = x_offset + subImgSize[0]
            arrow_end = arrow_start + arrowSize[0]
            mid_y = y_offset + subImgSize[1] // 2
            draw.line(
                [(arrow_start, mid_y), (arrow_end, mid_y)], fill=(0, 0, 0), width=2
            )
            draw.polygon(
                [
                    (arrow_end - 10, mid_y - 5),
                    (arrow_end, mid_y),
                    (arrow_end - 10, mid_y + 5),
                ],
                fill=(0, 0, 0),
            )

    fullImage.save(os.path.join("plots", "SI", f"{name}.png"))


def main():
    if not os.path.exists("plots"):
        os.makedirs("plots")
    if not os.path.exists("plots/SI"):
        os.makedirs("plots/SI")
    if not os.path.exists("plots/syntheses"):
        os.makedirs("plots/syntheses")

    artemisinin = "C[C@@]3(OO4)CC[C@]5([H])[C@@]4([C@@]([C@@H](C)C(O6)=O)([H])CC[C@H]5C)[C@@]6([H])O3"
    strictosidine = "C=C[C@@H]1[C@H](C[C@]2([H])NCCC3=C2NC4=C3C=CC=C4)C(C(OC)=O)=CO[C@H]1O[C@]([H])([C@](O)5[H])O[C@]([H])(CO)[C@@]([H])(O)[C@@]5(O)[H]"
    quinine = "COC1=CC=C(N=CC=C2[C@H](O)[C@@H]3C[C@H]4CC[N@@]3C[C@H]4C=C)C2=C1"
    camptothecin = "O=C1N2CC3=CC4=CC=CC=C4N=C3C2=CC5=C1COC([C@@]5(CC)O)=O"
    catharanthine = "CCC1=CC2CC(C13)(C(OC)=O)C4=C(CCN3C2)C5=CC=CC=C5N4"
    strychnine = "[H][C@@]12C[C@@]3([H])[C@]45[C@@]([C@]1([H])[C@@]6([H])OCC=C2CN3CC5)([H])N(C(C6)=O)C7=C4C=CC=C7"
    predictor = get_predictor()

    print(predictor._get_prediction_scaled(artemisinin), "!!!")

    smiles_list = [strictosidine, quinine, camptothecin, catharanthine, strychnine]
    names = ["Strictosidine", "Quinine", "Camptothecin", "Catharanthine", "Strychnine"]
    mc_res = {}
    for i, smiles in enumerate(smiles_list):
        pred = predictor._get_prediction_scaled(smiles)
        mc_res[names[i]] = pred
    print(mc_res)

    total_synthesis_dir = os.path.join("data", "total_synthesis")

    fontsize = 10

    strychnine = analyze_all_synthesis(
        predictor,
        path=os.path.join(total_synthesis_dir, "strychnine"),
        sep=">",
        with_smiles=True,
    )
    woodward = strychnine[0][0]
    prepare_si(woodward, "Strychnine Woodward 1954")
    overman = strychnine[1][0]
    prepare_si(overman, "Strychnine Overman 1993")
    vanderwal = strychnine[2][0]
    prepare_si(vanderwal, "Strychnine Vanderwal 2011")
    biosynthesis = strychnine[3][0]
    prepare_si(biosynthesis, "Strychnine Biosynthesis")
    mori = strychnine[4][0]
    prepare_si(mori, "Strychnine Mori 2002")

    all_complexities_strychnine = analyze_all_synthesis(
        predictor,
        path=os.path.join(total_synthesis_dir, "strychnine"),
        sep=">",
        with_smiles=False,
    )

    plot_synthesis(
        all_complexities_strychnine,
        "plots",
        add_captions=False,
        fontsize=fontsize,
        y_min=-0.5,
        y_max=8,
        height=4,
        width=6,
    )
    print("Strychnine done")

    artemisinin = analyze_all_synthesis(
        predictor,
        path=os.path.join(total_synthesis_dir, "artemisinin"),
        with_smiles=True,
    )
    avery = artemisinin[0][0]
    prepare_si(avery, "Artemisinin Avery 1992")
    krieger = artemisinin[1][0]
    prepare_si(krieger, "Artemisinin Krieger 2018")
    biosynthesis = artemisinin[2][0]
    prepare_si(biosynthesis, "Artemisinin Biosynthesis")
    cook = artemisinin[3][0]
    prepare_si(cook, "Artemisinin Cook 2018")

    artemisinin_complexities = analyze_all_synthesis(
        predictor,
        path=os.path.join(total_synthesis_dir, "artemisinin"),
        with_smiles=False,
    )
    plot_synthesis(
        artemisinin_complexities,
        "plots",
        add_captions=False,
        target_molecule="Artemisinin",
        y_max=6,
        y_min=-0.5,
        height=4,
        width=6,
        use_width=False,
        fontsize=fontsize,
    )
    print("Artemisinin done")


if __name__ == "__main__":
    main()
