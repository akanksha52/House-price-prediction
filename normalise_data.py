import pandas as pd
import feature_engineering

def normalise_data(csv):
    df=feature_engineering.feature_engineer(csv)
    
    for cols in df.columns:
        if pd.api.types.is_bool_dtype(df[cols]):
            continue
        
        df[cols]=(df[cols]-df[cols].mean())/(df[cols].std())

    return df