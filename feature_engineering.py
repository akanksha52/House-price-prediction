import pandas as pd
import data_preprocessing

def feature_engineer(csv):
    df=data_preprocessing.preprocess_data(csv)
    
    df['price_per_sqft']=df['price']/df['area']
    df['total_rooms']=df['bedrooms']+df['bathrooms']+df['guestroom']
    df['rooms_per_sqft']=df['total_rooms']/df['area']
    df['stories_per_area']=df['stories']/df['area']
    df['bed_bath']=df['bedrooms'] * df['bathrooms']
    df['area_ac']=df['area']*df['airconditioning']
    df['parking_area']=df['parking']*df['area']
    df['large_area']=df['area']>df['area'].median()
    df['guestroom_multi_story']=df['guestroom'] & (df['stories']>1)
    df['premium_location']=df['mainroad'] & df['prefarea']
    df['bedrooms_per_area']=df['bedrooms']/df['area']
    df['stories_per_area']=df['stories']/df['area']
    df['parking_per_area']=df['parking']/df['area']

    return df