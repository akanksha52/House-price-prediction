import pandas as pd
import numpy as np


def binary_encoder(series):
    categories={cat: idx+1 for idx, cat in enumerate(series.unique())}
    int_encoded=series.map(categories)
    max_value=int_encoded.max()
    n_bits=len(bin(max_value))-2
    binaries=int_encoded.apply(lambda x: list(map(int, bin(x)[2:].zfill(n_bits))))
    binary_df = pd.DataFrame(binaries.tolist(), index=series.index, columns=[f"{series.name}_bin{i}" for i in range(n_bits)])
    return categories, binary_df
