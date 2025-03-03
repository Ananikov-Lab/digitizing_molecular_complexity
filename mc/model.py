import pandas as pd
from .data import build_inference_df

def get_prediction(smi, ranker):
    """
    :param smi: smiles string
    :param ranker: CatBoostRanker object
    :param scaler: MinMaxScaler object
    :return: molecular complexity of molecule with smiles `smi`
    """
    df = pd.DataFrame({"smiles": [smi]})
    smi_df, _ = build_inference_df(df)
    pred = ranker.predict(smi_df).reshape(-1, 1)
    return pred[0][0]