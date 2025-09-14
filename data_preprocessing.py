import pandas as pd
import binary_encoding
import boolean_encoding


def normalize(series):
    return (series-series.min())/(series.max()-series.min())
 
def preprocess_data(csv):
    df=pd.read_csv(csv)
    df_copy=df.copy()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df_copy[column]): 
            continue
        
        n=len(df_copy[column].dropna().unique())
        
        if n==2:
            mapping, encoded=boolean_encoding.boolean_encoder(df_copy[column])
            df_copy[column]=encoded
        else:
            mapping, encoded=binary_encoding.binary_encoder(df_copy[column])
            df_copy=df_copy.drop(column, axis=1).join(encoded)
            
    for col in df_copy.columns:
            df_copy[col]=normalize(df_copy[col])

    return df_copy
