import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'Time' in df.columns:
        df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600) % 24)

    if 'Hour' in df.columns and 'Amount' in df.columns:
        df['Transaction_Count_Per_Hour'] = df.groupby('Hour')['Amount'].transform('count')
        df['Transaction_Mean_Per_Hour'] = df.groupby('Hour')['Amount'].transform('mean')
        df['Amount_Ratio'] = df['Amount'] / df['Transaction_Mean_Per_Hour']

    if 'Time' in df.columns:
        df['Time_Delta'] = df['Time'].diff().fillna(0)

    if 'Time' in df.columns:
        df['Event_Block'] = (df['Time'] // 3600).astype(int)

    df = df.fillna(0) 

    return df