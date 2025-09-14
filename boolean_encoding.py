import pandas as pd


def boolean_encoder(series, true_values=None, false_values=None):
    true_values=true_values or ['yes', 1, 'y', '1', 'true', 't', True, 'on']
    false_values=false_values or ['no', 0, 'n', '0', 'false', 'f', False, 'off']
    true_values=[str(v).lower() for v in true_values]
    false_values=[str(v).lower() for v in false_values]
    mapping={}
    for val in  series.unique():
        str_val = str(val).lower().strip()
        if str_val in true_values:
            mapping[val] = 1
        elif str_val in false_values:
            mapping[val] = 0
        else:
            mapping[val] = None
            

    encoded_series=series.apply(lambda x:
                                1 if str(x).lower().strip() in true_values else
                                0 if str(x).lower().strip() in false_values else
                                None)
    return mapping, encoded_series
