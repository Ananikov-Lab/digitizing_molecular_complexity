import os
from pprint import pprint
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import datamol as dm
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .data import build_inference_df
from .reaction_processing import ReactionsManager
from .utils import FUNCTIONAL_GROUPS, replace_hydrogen_with_group as modify, valid_mol


class PredictorAbstract(ABC):
    def __init__(self, ranker: CatBoostRanker, scaler: MinMaxScaler = None):
        self.ranker = ranker
        self.scaler = scaler

    def _get_prediction(self, smi: str) -> float:
        df = pd.DataFrame({"smiles": [smi]})
        smi_df, _ = build_inference_df(df)
        pred = self.ranker.predict(smi_df).reshape(-1, 1)
        return pred[0][0]

    def _get_prediction_scaled(self, smi: str) -> float:
        if self.scaler is None:
            raise ValueError("Scaler is not defined")
        df = pd.DataFrame({"smiles": [smi]})
        smi_df, _ = build_inference_df(df)
        pred_ = self.ranker.predict(smi_df).reshape(-1, 1)
        pred = round(self.scaler.transform(pred_)[0][0] * 10, 2)
        return pred

    def predict(self, smiles_list: List[str]) -> List[float]:
        _df = pd.DataFrame({"smiles": smiles_list})
        smi_df, _ = build_inference_df(_df)
        pred = self.ranker.predict(smi_df).reshape(-1, 1)
        if self.scaler is None:
            return [p[0] for p in pred]
        scaled = self.scaler.transform(pred)
        scaled = [round(s[0] * 10, 2) for s in scaled]
        return scaled

    def return_mc_w_feature(
        self, smiles_list: List[str], feature_name: str
    ) -> List[Tuple[float, float]]:
        _df = pd.DataFrame({"smiles": smiles_list})
        smi_df, _ = build_inference_df(_df)
        pred = self.ranker.predict(smi_df).reshape(-1, 1)
        feature_vals = smi_df[feature_name].values.tolist()
        if self.scaler is None:
            return list(zip([(p[0], 0) for p in pred], feature_vals))
        scaled = self.scaler.transform(pred)
        scaled = [round(s[0] * 10, 2) for s in scaled]
        return list(zip(scaled, feature_vals))

    @abstractmethod
    def benchmark(self) -> Dict[str, float]:
        pass


class Predictor(PredictorAbstract):
    def __init__(self, ranker: CatBoostRanker, scaler: MinMaxScaler = None):
        super().__init__(ranker, scaler)

    def benchmark(self):
        return 1


class StrychnineAnalyzer(PredictorAbstract):
    def __init__(
        self,
        ranker: CatBoostRanker,
        # path: str,
        scaler=None,
    ):
        super().__init__(ranker, scaler)
        self.path = os.path.join("data", "total_synthesis", "strychnine")
        self.complexities = {
            rm.get_title(): mc for mc, rm in self._get_complexities(self.path)
        }
        self.biosynthesis_complexities = self.complexities["biosynthesis"]

    def _get_complexities(self, path, sep=">>"):
        all_complexities = []
        for txt in os.listdir(path=path):
            if not txt.endswith(".txt"):
                continue
            print("Processing", txt)
            rm = ReactionsManager(os.path.join(path, txt), sep=sep)
            all_complexities.append((self._analyze_synthesis(rm), rm))
        return all_complexities

    def _analyze_synthesis(self, rm: ReactionsManager, mode="max"):
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
                        pred = self._get_prediction_scaled(reagent)
                    except:
                        print(reagent)
                        pred = 0
                    tmp.append(pred)
                complexities.append(max(tmp))

            final_product = rm.final_product
            pred = self._get_prediction_scaled(final_product)
            complexities.append(pred)
        return complexities

    def strictosidine_benchmark_all(self):
        """
        Checks if the complexity of strictosidine is above all the other molecules
        in all the syntheses
        """
        strictosidine_index = 10
        strictosidine_complexity = self.complexities["biosynthesis"][
            strictosidine_index
        ]
        for _, complexities in self.complexities.items():
            if strictosidine_complexity < max(complexities):
                return False
        return True

    def strictosidine_benchmark_bio(self):
        """
        Checks if the complexity of strictosidine is above all the other molecules
        in the biosynthesis
        """
        strictosidine_index = 10
        strictosidine_complexity = self.complexities["biosynthesis"][
            strictosidine_index
        ]
        if strictosidine_complexity < max(self.biosynthesis_complexities):
            return False
        return True

    def bigger_mc_than_product(self):
        number_of_compounds = 0
        strychnine_complexity = self.biosynthesis_complexities[-1]
        for mc in self.biosynthesis_complexities[:-1]:
            if mc > strychnine_complexity:
                number_of_compounds += 1
        return number_of_compounds

    def most_complex_intermediate(self):
        """
        len(MC) - argmax(MC)
        number of steps between the most complex intermediate and the final product
        """
        return len(self.biosynthesis_complexities) - (
            np.argmax(self.biosynthesis_complexities) + 1
        )

    def most_complex_relative_difference(self):
        """
        (max(MC) - MC_p) / MC_p
        difference between the most complex structure during the synthesis and final product
        """
        strychnine_complexity = self.biosynthesis_complexities[-1]
        return (
            max(self.biosynthesis_complexities) - strychnine_complexity
        ) / strychnine_complexity

    def benchmark(self):
        report = {
            "strictosidine_is_most_comlex_all": self.strictosidine_benchmark_all(),
            "strictosidine_is_most_comlex_bio": (
                self.strictosidine_benchmark_bio(),
                self.biosynthesis_complexities,
            ),
            "num_of_compounds_w_bigger_mc_than_prod": self.bigger_mc_than_product(),
            "num_steps_between_most_complex_and_prod": self.most_complex_intermediate(),
            "rel_diff_between_most_complex_and_prod": self.most_complex_relative_difference(),
        }
        return report


class BasicAnalyzer(PredictorAbstract):
    def __init__(self, ranker: CatBoostRanker, scaler: MinMaxScaler = None):
        super().__init__(ranker, scaler)
        # Ph-R, where R = H, F, Cl, Br, NO2, Ph
        self.molecules = list(
            map(
                self._canonicalize,
                [
                    "c1ccccc1",
                    "C1=CC=C(C=C1)F",
                    "C1=CC=C(C=C1)Cl",
                    "c1ccc(cc1)Cl",
                    "O=N(=O)C1=CC=CC=C1",
                    "C1=CC=C(C=C1)C2=CC=CC=C2",
                ],
            )
        )
        self.alkanes = self._get_alkanes()

    def _get_alkanes(self, n=10):
        return ["C" * i for i in range(1, n + 1)]

    def _is_monotonous(self, lst):
        return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

    def benchmark(self):
        alkanes_mc = [
            self._get_prediction_scaled(alkane) for alkane in self._get_alkanes()
        ]
        func_groups_mc = [
            self._get_prediction_scaled(molecule) for molecule in self.molecules
        ]
        report = {
            "alkanes_monotonously_incr": (self._is_monotonous(alkanes_mc), alkanes_mc),
            "functional_groups_monotonously_incr": (
                self._is_monotonous(func_groups_mc),
                func_groups_mc,
            ),
        }
        return report

    def _canonicalize(self, smiles: str, include_stereocenters=True):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
        else:
            return None


class FunctionalGroupsAnalyzer(PredictorAbstract):
    def __init__(
        self,
        ranker: CatBoostRanker,
        smiles_with_modifications: Dict[str, Dict[str, str]],
        scaler: MinMaxScaler = None,
    ):
        """
        smiles_with_modifications: dict of the form {smiles: {functional_group: modified_smiles}}
        where smiles is a default molecule and smiles_with_modifications is a dict of the form
        {functional_group: modified_smiles}
        """
        super().__init__(ranker, scaler)
        self.smiles_with_modifications = smiles_with_modifications

    def benchmark(self):
        predictions_per_fg = self._predict_per_fg()
        report = {}
        binary_report = {}
        for group, predictions in predictions_per_fg.items():
            if group == "default":
                continue
            binary_report[group] = np.array(predictions) > np.array(
                predictions_per_fg["default"]
            )
            report[group] = binary_report[group].mean()
        return report

    def _predict_per_fg(self):
        predictions = {}
        for group, modifications in self.smiles_with_modifications.items():
            predictions[group] = self.predict(modifications)
        return predictions


def get_smiles_with_modifications(
    smiles_lst=dm.data.chembl_drugs()["smiles"].tolist(),
) -> Dict[str, List[str]]:
    smiles_modifications = {"default": []}
    smiles_lst = list(filter(valid_mol, smiles_lst))
    for smiles in tqdm(smiles_lst):
        d = {}
        valid = True
        for group in FUNCTIONAL_GROUPS:
            if group not in smiles_modifications:
                smiles_modifications[group] = []
            try:
                modification = modify(smiles, group)
            except:
                valid = False
                print(f"Invalid modification: {group}")
                break
            if valid_mol(modification):
                d[group] = modification
            else:
                valid = False
                print(f"Invalid modification: {modification}")
                break
        if not valid:
            continue

        smiles_modifications["default"].append(smiles)
        for group, modification in d.items():
            smiles_modifications[group].append(modification)

    return smiles_modifications


def get_stats(
    model: CatBoostRanker,
    smiles_with_modifications: Dict[str, Dict[str, str]],
    scaler: MinMaxScaler = None,
) -> Dict[str, float]:
    analyzers = [
        # BasicAnalyzer(model, scaler),
        # StrychnineAnalyzer(model, scaler),
        FunctionalGroupsAnalyzer(model, smiles_with_modifications, scaler),
    ]
    report = {}
    for analyzer in analyzers:
        report |= analyzer.benchmark()
    return report
