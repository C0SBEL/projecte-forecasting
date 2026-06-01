import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseForecaster(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, series: pd.Series, **kwargs):
        pass

    @abstractmethod
    def predict(self, h: int, **kwargs) -> np.ndarray:
        pass

    def fit_predict(self, train: pd.Series, h: int, **kwargs) -> np.ndarray:
        self.fit(train, **kwargs)
        return self.predict(h, **kwargs)


class SeasonalNaive(BaseForecaster):
    def __init__(self, seasonality: int = 12):
        super().__init__('SeasonalNaive')
        self.seasonality = seasonality

    def fit(self, series: pd.Series, **kwargs):
        self._last_season = series.values[-self.seasonality:]
        self.is_fitted = True
        return self

    def predict(self, h: int, **kwargs) -> np.ndarray:
        reps = (h // self.seasonality) + 1
        return np.tile(self._last_season, reps)[:h]


class SeasonalMean(BaseForecaster):
    """Forecast = historical mean for the same month."""
    def __init__(self):
        super().__init__('SeasonalMean')

    def fit(self, series: pd.Series, **kwargs):
        idx = pd.to_datetime(series.index) if not isinstance(series.index, pd.DatetimeIndex) else series.index
        s = pd.Series(series.values, index=idx)
        self._monthly_means = s.groupby(s.index.month).mean()
        self._last_month = idx[-1].month
        self.is_fitted = True
        return self

    def predict(self, h: int, **kwargs) -> np.ndarray:
        months = [(self._last_month + i) % 12 + 1 for i in range(h)]
        return np.array([self._monthly_means.get(m, self._monthly_means.mean()) for m in months])


class ETSForecaster(BaseForecaster):
    def __init__(self, error: str = 'add', trend: str = 'add',
                 seasonal: str = 'add', seasonal_periods: int = 12, name: str = None):
        super().__init__(name or f'ETS({error},{trend},{seasonal})')
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

    def fit(self, series: pd.Series, **kwargs):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        s = series.copy()

        # Assignar freqüència si falta (evita ValueWarning de statsmodels)
        if isinstance(s.index, pd.DatetimeIndex) and s.index.freq is None:
            s = s.asfreq('MS')

        # ETS multiplicatiu requereix valors estrictament positius.
        # Si hi ha zeros, desplacem tota la sèrie +1 i ho compensem a predict().
        self._offset = 0
        if (self.seasonal == 'mul' or self.trend == 'mul') and (s <= 0).any():
            self._offset = 1
            s = s + self._offset

        self._fitted = ExponentialSmoothing(
            s,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        ).fit(optimized=True, use_brute=False)
        self.is_fitted = True
        return self

    def predict(self, h: int, **kwargs) -> np.ndarray:
        return np.maximum(self._fitted.forecast(h).values - self._offset, 0)


class SARIMAForecaster(BaseForecaster):
    """Auto-ARIMA wrapper using pmdarima."""
    def __init__(self, d: int = 1, D: int = 1, m: int = 12,
                 max_p: int = 3, max_q: int = 3, max_P: int = 2, max_Q: int = 2,
                 name: str = 'AutoSARIMA'):
        super().__init__(name)
        self.d, self.D, self.m = d, D, m
        self.max_p, self.max_q = max_p, max_q
        self.max_P, self.max_Q = max_P, max_Q

    def fit(self, series: pd.Series, exog=None, **kwargs):
        import pmdarima as pm
        self._model = pm.auto_arima(
            series,
            seasonal=True, m=self.m,
            d=self.d, D=self.D,
            max_p=self.max_p, max_q=self.max_q,
            max_P=self.max_P, max_Q=self.max_Q,
            exogenous=exog,
            stepwise=True,
            information_criterion='aic',
            suppress_warnings=True,
            error_action='ignore',
        )
        self._exog = exog
        self.is_fitted = True
        return self

    def predict(self, h: int, exog_future=None, **kwargs) -> np.ndarray:
        return np.asarray(self._model.predict(n_periods=h, exogenous=exog_future))

    def summary(self):
        return self._model.summary()


class SARIMAXForecaster(SARIMAForecaster):
    """SARIMA with COVID and MOVES3 dummy regressors."""
    def __init__(self, **kwargs):
        super().__init__(name='SARIMAX', **kwargs)


class ProphetForecaster(BaseForecaster):
    def __init__(self, changepoints: list = None, changepoint_prior_scale: float = 0.1,
                 extra_regressors: list = None, name: str = 'Prophet'):
        super().__init__(name)
        self.changepoints = changepoints or ['2020-03-01', '2021-06-01']
        self.changepoint_prior_scale = changepoint_prior_scale
        self.extra_regressors = extra_regressors or []

    def fit(self, series: pd.Series, extra_regressors_df: pd.DataFrame = None, **kwargs):
        from prophet import Prophet
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

        df = pd.DataFrame({'ds': pd.to_datetime(series.index), 'y': series.values})
        self._model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoints=self.changepoints,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        for reg in self.extra_regressors:
            self._model.add_regressor(reg)
        if extra_regressors_df is not None:
            df = df.merge(extra_regressors_df, on='ds', how='left')
        self._model.fit(df)
        self._last_date = pd.to_datetime(series.index[-1])
        self.is_fitted = True
        return self

    def predict(self, h: int, extra_regressors_df: pd.DataFrame = None, **kwargs) -> np.ndarray:
        future = self._model.make_future_dataframe(periods=h, freq='MS')
        if extra_regressors_df is not None:
            future = future.merge(extra_regressors_df, on='ds', how='left')
        forecast = self._model.predict(future)
        return np.maximum(forecast.tail(h)['yhat'].values, 0)

    def get_forecast_df(self, h: int, extra_regressors_df: pd.DataFrame = None) -> pd.DataFrame:
        future = self._model.make_future_dataframe(periods=h, freq='MS')
        if extra_regressors_df is not None:
            future = future.merge(extra_regressors_df, on='ds', how='left')
        return self._model.predict(future).tail(h)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


class LGBMForecaster(BaseForecaster):
    """Direct multi-step LightGBM: one model per forecast horizon step."""
    def __init__(self, horizon: int = 12, n_estimators: int = 500,
                 learning_rate: float = 0.05, num_leaves: int = 31, name: str = 'LightGBM'):
        super().__init__(name)
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self._models = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        import lightgbm as lgb
        for h in range(1, self.horizon + 1):
            y_h = y_train.shift(-h)
            mask = y_h.notna()
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                verbose=-1,
            )
            model.fit(X_train[mask], y_h[mask])
            self._models[h] = model
        self.is_fitted = True
        return self

    def predict(self, X_last: pd.DataFrame, **kwargs) -> np.ndarray:
        return np.array([
            max(self._models[h].predict(X_last)[0], 0)
            for h in range(1, self.horizon + 1)
        ])

    def feature_importance(self) -> pd.DataFrame:
        rows = []
        for h, model in self._models.items():
            imp = pd.Series(model.feature_importances_, index=model.feature_name_)
            rows.append(imp.rename(f'h={h}'))
        return pd.DataFrame(rows).T.sort_values('h=1', ascending=False)
