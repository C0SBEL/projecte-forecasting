import numpy as np
import pandas as pd
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray,
         seasonality: int = 12) -> float:
    """MAE scaled by seasonal naive in-sample MAE."""
    y_train = np.asarray(y_train)
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)


def compute_metrics(y_true, y_pred, y_train=None, seasonality: int = 12) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics = {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
    }
    if y_train is not None:
        metrics['MASE'] = mase(y_true, y_pred, np.asarray(y_train), seasonality)
    return metrics


def walk_forward_cv(series: pd.Series, model_fn, initial: int = 48,
                    step: int = 6, horizon: int = 12) -> pd.DataFrame:
    """
    Expanding window walk-forward cross-validation.
    model_fn: callable(train_series: pd.Series) -> np.ndarray of length horizon
    """
    n = len(series)
    results = []
    for start in range(initial, n - horizon + 1, step):
        train = series.iloc[:start]
        test = series.iloc[start:start + horizon]
        try:
            forecast = np.asarray(model_fn(train))
            fold_metrics = compute_metrics(test.values, forecast, train.values)
            fold_metrics['fold_start'] = series.index[start]
            fold_metrics['n_train'] = start
        except Exception as e:
            fold_metrics = {
                'fold_start': series.index[start],
                'n_train': start,
                'error': str(e),
            }
        results.append(fold_metrics)
    return pd.DataFrame(results)


def walk_forward_cv_full(series: pd.Series, model_fn, initial: int = 60,
                         step: int = 3, horizon: int = 12) -> tuple:
    """Expanding-window WFV that returns both per-fold metrics AND per-(fold, horizon) forecasts.

    Returns (metrics_df, forecasts_df):
      metrics_df:   one row per fold with RMSE, MAE, MAPE, MASE, fold_start, n_train
      forecasts_df: one row per (fold, h) with fold_start, h, date, predicted, actual
    """
    n = len(series)
    metric_rows, forecast_rows = [], []
    for start in range(initial, n - horizon + 1, step):
        train = series.iloc[:start]
        test  = series.iloc[start:start + horizon]
        try:
            y_pred = np.asarray(model_fn(train), dtype=float)
            y_true = test.values
            m = compute_metrics(y_true, y_pred, train.values)
            m['fold_start'] = series.index[start]
            m['n_train']    = start
            metric_rows.append(m)
            for h in range(horizon):
                forecast_rows.append({
                    'fold_start': series.index[start],
                    'h':          h + 1,
                    'date':       test.index[h],
                    'predicted':  float(y_pred[h]),
                    'actual':     float(y_true[h]),
                })
        except Exception as e:
            metric_rows.append({
                'fold_start': series.index[start],
                'n_train':    start,
                'error':      str(e),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(forecast_rows)


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray,
                         h: int = 1, power: int = 2) -> tuple:
    """
    DM test for equal predictive accuracy.
    H0: both models have equal MSE/MAE.
    Returns (dm_stat, p_value) two-tailed.
    """
    e1, e2 = np.asarray(e1), np.asarray(e2)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    autocovariances = [gamma0]
    for i in range(1, h):
        if i < n:
            autocovariances.append(np.cov(d[i:], d[:-i])[0, 1])
    var_d = (autocovariances[0] + 2 * sum(autocovariances[1:])) / n
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * stats.norm.cdf(-np.abs(dm_stat))
    return float(dm_stat), float(p_value)


def summarize_cv_results(cv_results: dict) -> pd.DataFrame:
    """cv_results: {model_name: DataFrame from walk_forward_cv}"""
    rows = []
    for name, df in cv_results.items():
        numeric = df[['RMSE', 'MAE', 'MAPE', 'MASE']].dropna()
        if len(numeric) == 0:
            continue
        rows.append({
            'Model': name,
            'RMSE_mean': numeric['RMSE'].mean(),
            'RMSE_std': numeric['RMSE'].std(),
            'MAE_mean': numeric['MAE'].mean(),
            'MAPE_mean': numeric['MAPE'].mean(),
            'MASE_mean': numeric['MASE'].mean() if 'MASE' in numeric else np.nan,
            'n_folds': len(numeric),
        })
    return pd.DataFrame(rows).set_index('Model').round(2)
