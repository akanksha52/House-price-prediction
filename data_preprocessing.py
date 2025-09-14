import pandas as pd
import binary_encoding
import boolean_encoding

def preprocess_data(csv):
    df=pd.read_csv(csv)

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]): 
            continue
        
        n=len(df[column].dropna().unique())
        
        if n==2:
            mapping, encoded=boolean_encoding.boolean_encoder(df[column])
            df[column]=encoded
        else:
            mapping, encoded=binary_encoding.binary_encoder(df[column])
            df=df.drop(column, axis=1).join(encoded)

    return df
