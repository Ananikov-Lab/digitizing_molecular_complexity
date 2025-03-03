import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import (
    StereoEnumerationOptions,
    GetStereoisomerCount,
)

from mc.scscorer.scscore.standalone_model_numpy import SCScorer


def _get_scorer():
    model = SCScorer()
    model.restore(
        os.path.join(
            "mc",
            "scscorer",
            "models",
            "full_reaxys_model_2048bool",
            "model.ckpt-10654.as_numpy.json.gz",
        ),
        FP_len=2048,
    )
    return model

def featurize(data: pd.DataFrame) -> pd.DataFrame:
    """
    :param data: filtered dataset
    :return: dataset with molecular and structural features
    """
    data["weight"] = data["mol"].map(Descriptors.ExactMolWt)
    data["num_of_atoms"] = data["mol"].apply(lambda x: x.GetNumAtoms())
    data["tpsa"] = data["mol"].apply(lambda x: Descriptors.TPSA(x))
    data["num_heteroatoms"] = (
        data["mol"]
        .apply(lambda x: Descriptors.NumHeteroatoms(x))
        .apply(lambda x: float(x))
    )
    data["spiro"] = data["mol"].map(rdMolDescriptors.CalcNumSpiroAtoms)
    data["rotb"] = data["mol"].map(rdMolDescriptors.CalcNumRotatableBonds)
    data["aliph_cycles"] = data["mol"].map(rdMolDescriptors.CalcNumAliphaticCarbocycles)
    data["arom_cycles"] = data["mol"].map(rdMolDescriptors.CalcNumAromaticCarbocycles)
    data["aliph_heterocycles"] = data["mol"].map(
        rdMolDescriptors.CalcNumAliphaticHeterocycles
    )
    data["arom_heterocycles"] = data["mol"].map(
        rdMolDescriptors.CalcNumAromaticHeterocycles
    )
    data["bridge_atoms"] = data["mol"].map(rdMolDescriptors.CalcNumBridgeheadAtoms)
    data["atom_stereo_centers"] = data["smiles"].apply(
        lambda x: rdMolDescriptors.CalcNumAtomStereoCenters(Chem.MolFromSmiles(x))
    )
    opts = StereoEnumerationOptions(onlyUnassigned=False, unique=True)
    data["num_of_stereoisomers"] = (
        data["mol"]
        .apply(lambda x: GetStereoisomerCount(x, options=opts))
        .apply(lambda x: float(x))
    )
    scorer = _get_scorer()
    data["scscore"] = data["smiles"].apply(lambda x: scorer.get_score_from_smi(x)[-1])

    return data
