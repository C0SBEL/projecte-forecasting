import pandas as pd
import numpy as np

TECNOLOGIES_MODELITZABLES = [
    'Elèctric',
    'Gasolina Mild Hybrid',
    'Gasolina/Elèctric',
    'Gasolina/Elèctric Endollable',
    'Diesel Mild Hybrid',
    'Diesel/Elèctric Endollable',
]

TECNOLOGIES_ESPARSES = [
    'Gas Mild Hybrid',
    'Diesel/Elèctric',
]

COVID_START = '2020-03-01'
COVID_END = '2020-09-01'
MOVES3_START = '2021-06-01'

TRAIN_END = '2024-12-01'
TEST_START = '2025-01-01'


def create_lag_features(df: pd.DataFrame, target_col: str = 'Matriculacions',
                        lags: list = [1, 2, 3, 6, 12]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = 'Matriculacions',
                            windows: list = [3, 6, 12]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f'rolling_mean_{w}'] = df[target_col].shift(1).rolling(w).mean()
        df[f'rolling_std_{w}'] = df[target_col].shift(1).rolling(w).std()
    return df


def create_date_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    df['month'] = dates.dt.month
    df['quarter'] = dates.dt.quarter
    df['trend'] = np.arange(len(df))
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


def create_intervention_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    df['covid_dummy'] = ((dates >= COVID_START) & (dates <= COVID_END)).astype(int)
    df['moves3_dummy'] = (dates >= MOVES3_START).astype(int)
    return df


def build_ml_features(series_df: pd.DataFrame, target_col: str = 'Matriculacions',
                      date_col: str = 'date') -> pd.DataFrame:
    df = series_df.copy().sort_values(date_col).reset_index(drop=True)
    df = create_date_features(df, date_col)
    df = create_intervention_features(df, date_col)
    df = create_lag_features(df, target_col)
    df = create_rolling_features(df, target_col)
    df['yoy_ratio'] = df['lag_1'] / (df['lag_12'] + 1e-8)
    return df


def get_feature_columns() -> list:
    lag_cols = [f'lag_{l}' for l in [1, 2, 3, 6, 12]]
    roll_cols = [f'rolling_{s}_{w}' for s in ['mean', 'std'] for w in [3, 6, 12]]
    date_cols = ['month_sin', 'month_cos', 'quarter', 'trend']
    dummy_cols = ['covid_dummy', 'moves3_dummy']
    return lag_cols + roll_cols + date_cols + dummy_cols + ['yoy_ratio']


def get_pivot(df: pd.DataFrame, tecnologies: list = None) -> pd.DataFrame:
    """Pivot dataset_treball.csv to wide format: date × tecnologia."""
    if tecnologies:
        df = df[df['Tecnologia'].isin(tecnologies)]
    pivot = df.pivot(index='date', columns='Tecnologia', values='Matriculacions')
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.sort_index()
