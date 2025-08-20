"""
This script is implemented by Retrosim https://github.com/connorcoley/retrosim
We use it for similarity search.
"""
from typing import List, Tuple
import time
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction

from src.rag.generate_retro_templates import process_an_example
from src.utils.chemistry_utils import MORGAN_FP_GENERATOR, SIMILARITY_METRIC


def get_data_df(fpath: str="./data/data_processed.csv"):
    return pd.read_csv(fpath)

def split_data_df(
    data: pd.DataFrame,
    val_frac: float = 0.0,
    test_frac: float = 0.0,
    shuffle: bool = False,
    seed: int = None
) -> None:
    """Split the data into train, val, and test sets."""
    # Define shuffling
    if shuffle:
        np.random.seed(int(time.time())) if seed is None else np.random.seed(seed)
        def shuffle_func(x):
            np.random.shuffle(x)
    else:
        def shuffle_func(x):
            pass

    # Go through each class
    logging.info("Reference routes' statistics:")
    classes = sorted(np.unique(data["class"]))
    for class_ in classes:
        indices = data.loc[data["class"] == class_].index
        N = len(indices)
        logging.info("{} rows with class value {}".format(N, class_))

        shuffle_func(indices)
        train_end = int((1.0 - val_frac - test_frac) * N)
        val_end = int((1.0 - test_frac) * N)

        for i in indices[:train_end]:
            data.at[i, "dataset"] = "train"
        for i in indices[train_end:val_end]:
            data.at[i, "dataset"] = "val"
        for i in indices[val_end:]:
            data.at[i, "dataset"] = "test"
    logging.info(data["dataset"].value_counts())

def do_one(
    product_smiles: str,
    datasub: pd.DataFrame,
    jx_cache: dict,
    max_prec: int = 100
) -> Tuple[List[Tuple[str, float]], dict]:
    """Find similar reaction templates for a product molecule using fingerprint similarity."""
    rct = rdchiralReactants(product_smiles)
    fp = MORGAN_FP_GENERATOR(Chem.MolFromSmiles(product_smiles))
    
    sims = SIMILARITY_METRIC(fp, [fp_ for fp_ in datasub["prod_fp"]])
    js = np.argsort(sims)[::-1]

    # Get probability of precursors
    probs = {}
    
    for ji, j in enumerate(js[:max_prec]):
        jx = datasub.index[j]

        if jx in jx_cache:
            (rxn, template, rcts_ref_fp) = jx_cache[jx]
        else:
            template = "(" + process_an_example(datasub["rxn_smiles"][jx], super_general=True).replace(">>", ")>>")
            rcts_ref_fp = MORGAN_FP_GENERATOR(Chem.MolFromSmiles(datasub["rxn_smiles"][jx].split(">")[0]))
            rxn = rdchiralReaction(template)
            jx_cache[jx] = (rxn, template, rcts_ref_fp)
            
        try:
            outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
        except Exception:
            outcomes = []
            
        for precursors in outcomes:
            precursors_fp = MORGAN_FP_GENERATOR(Chem.MolFromSmiles(precursors))
            precursors_sim = SIMILARITY_METRIC(precursors_fp, [rcts_ref_fp])[0]

            if template in probs:
                probs[template] = max(probs[template], precursors_sim * sims[j])
            else:
                probs[template] = precursors_sim * sims[j]
    
    testlimit = 100
    template_list = []

    for r, (template, prob) in enumerate(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:testlimit]):
        template_list.append((template, prob))
    return template_list, jx_cache
