# Projecte Forecasting: matriculacions de vehicles electrificats i híbrids a Catalunya

Benchmark de models univariants de sèries temporals per predir matriculacions mensuals de vehicles electrificats i híbrids a Catalunya (DGT/IDESCAT), amb previsió per a l'any 2026.

Treball final de l'assignatura *Forecasting* del Màster en Data Science (La Salle BCN, curs 2025-26).

## Context

- **Dataset**: matriculacions mensuals per tecnologia de propulsió a Catalunya. Font: DGT/IDESCAT. Període: gener 2017 a desembre 2025 (108 mesos).
- **Sèries modelitzades** (7): Total agregat i 6 sub-tecnologies: Elèctric (BEV), Gasolina Mild Hybrid, Gasolina/Elèctric (HEV), Gasolina/Elèctric Endollable (PHEV), Diesel Mild Hybrid, Diesel/Elèctric Endollable.
- **Objectiu**: comparar 10 models univariants en dues avaluacions complementàries i generar una previsió a 12 mesos (2026).
- **Particions**:
  - *Backtest 2025*: train 2017-2024 (96 mesos), test 2025 (12 mesos, holdout únic).
  - *Walk-forward validation* (WFV): 13 origins amb `initial=60`, `step=3`, `horizon=12`.

El Total es modelitza com a sèrie pròpia (no com a suma de sub-tecnologies). Les mètriques es reporten **per separat** per al Total i per a les 6 sub-tecnologies; no es fa cap mitjana entre nivells jeràrquics.

## Mètriques

Es calculen i es reporten quatre mètriques a totes les taules i gràfics: **RMSE** (matrícules/mes), **MAE** (matrícules/mes), **MAPE** (%) i **MASE**. A la narrativa es discuteix principalment RMSE i MAPE perquè són les més directament interpretables (RMSE en unitats reals, MAPE en percentatge sobre el volum mensual). MAE complementa puntualment quan reforça una comparació, i MASE es manté com a referència contra el SeasonalNaive (lectura binària: MASE < 1 vol dir millor que repetir l'any anterior). Com que l'anàlisi és per cada sèrie de manera independent, la propietat scale-free del MASE no és imprescindible aquí.

## Models implementats

10 models al benchmark principal, en **3 famílies** (alineades amb la guia del professor: deterministes, estocàstics, avançats):

| Família | Models | Notebook |
|---|---|---|
| Deterministes (4) | SeasonalNaive, MA_12, OLS_Estacional, OLS_Complet (afegeix dummies COVID i MOVES III) | `models/deterministic/deterministic.ipynb` |
| Estocàstics (4) | ETS Additiu, ETS Multiplicatiu, AutoSARIMA, TBATS | `models/stochastic/ets.ipynb` (ETS), `sarima.ipynb`, `tbats.ipynb` |
| Avançats (2) | LightGBM, LSTM | `models/advanced/lgbm.ipynb`, `lstm.ipynb` |

Models exploratoris addicionals a `ensemble_global_lgbm.ipynb` (no entren al rànquing principal): Ensemble (mitjana ETS+SARIMA+TBATS) i Global LightGBM (entrenat sobre les 7 sèries amb `tech_id`).

Transformació `log(y+1)` aplicada NOMÉS al Total per a AutoSARIMA i ETS Additiu (BoxCox `lambda ≈ 0.49` a l'EDA).

## Resultats principals

**Backtest 2025, rànquing al Total** (ordenat per RMSE en matrícules/mes; MAPE en %):

| Model | RMSE | MAPE |
|---|---:|---:|
| AutoSARIMA (log) | 464 | 4,6% |
| ETS Additiu (log) | 877 | 8,9% |
| LSTM | 1.143 | 13,3% |
| TBATS | 1.159 | 11,9% |
| OLS_Estacional | 1.715 | 18,1% |
| OLS_Complet | 1.778 | 19,0% |
| SeasonalNaive | 2.140 | 25,4% |
| LightGBM | 2.350 | 24,1% |
| MA_12 | 2.360 | 23,6% |
| ETS Multiplicatiu | 2.449 | 29,6% |

**WFV, rànquing al Total** (mitjana sobre 13 folds, ordenat per RMSE):

| Model | RMSE | MAPE |
|---|---:|---:|
| OLS_Estacional | 857 | 12,0% |
| OLS_Complet | 860 | 11,9% |
| ETS Additiu | 979 | 15,7% |
| TBATS | 1.046 | 15,2% |
| LSTM | 1.122 | 15,9% |
| SeasonalNaive | 1.204 | 17,7% |
| AutoSARIMA | 1.336 | 21,2% |
| MA_12 | 1.336 | 18,5% |
| ETS Multiplicatiu | 1.441 | 22,1% |
| LightGBM | 1.494 | 20,9% |

**Guanyadors al Total**:
- *Backtest 2025*: **AutoSARIMA amb log** (RMSE 464 matrícules/mes, MAPE 4,6%). Marge ampli respecte a ETS Additiu amb log (RMSE 877, MAPE 8,9%).
- *WFV* (mitjana 13 folds): **OLS_Estacional** i OLS_Complet empatats al capdamunt (RMSE 857-860, MAPE ~12%); ETS Additiu queda tercer (RMSE 979, MAPE 15,7%); AutoSARIMA cau al setè (RMSE 1.336).

**Lectura clau**: backtest i WFV són complementaris, no contradictoris. AutoSARIMA(log) destaca al cicle 2025 (any excepcional, +34% YoY) però és fràgil en altres folds. OLS_Estacional guanya en mitjana WFV (resultat contraintuïtiu coherent amb la lliçó M-competitions: clàssics > ML quan N és petit). La recomanació de producció és OLS_Estacional per robustesa i simplicitat; per a una previsió puntual a 12 mesos com la del 2026, el guanyador del backtest per sèrie és l'opció natural.

**Previsió 2026** (model guanyador del backtest per sèrie):

| Tecnologia | Model | 2025 (real) | 2026 (prev.) | YoY |
|---|---|---:|---:|---:|
| Total | AutoSARIMA(log) | 96.175 | 114.411 | +19,0% |
| Elèctric | ETS Multiplicatiu | 20.495 | 22.253 | +8,6% |
| Gasolina MH | AutoSARIMA | 25.789 | 27.814 | +7,9% |
| Gasolina/Elèctric | ETS Multiplicatiu | 30.300 | 27.386 | -9,6% |
| PHEV gasolina | OLS_Estacional | 15.606 | 13.584 | -13,0% |
| Diesel MH | AutoSARIMA | 3.407 | 3.596 | +5,6% |
| Diesel PHEV | AutoSARIMA | 526 | 649 | +23,3% |

## Estructura del repositori

```
projecte-forecasting/
├── data/
│   └── processed/                # CSVs treballats (train/test, pivots, descriptiva)
│       └── results/              # Forecasts i mètriques de tots els models
├── docs/                         # EDA report i enunciat
├── figures/                      # Figures (PNG) generades pels notebooks
├── notebooks/
│   ├── data_preparation.ipynb    # Càrrega, neteja, particions
│   ├── eda.ipynb                 # Anàlisi exploratori, ADF, BoxCox, ACF/PACF
│   ├── models/                       # Notebooks agrupats per família metodològica
│   │   ├── deterministic/
│   │   │   └── deterministic.ipynb   # SeasonalNaive + MA_12 + OLS_Estacional + OLS_Complet
│   │   ├── stochastic/
│   │   │   ├── ets.ipynb             # ETS Additiu + ETS Multiplicatiu
│   │   │   ├── sarima.ipynb          # AutoSARIMA (amb log al Total)
│   │   │   └── tbats.ipynb           # TBATS
│   │   └── advanced/
│   │       ├── lgbm.ipynb            # LightGBM per sèrie
│   │       └── lstm.ipynb            # LSTM per sèrie
│   ├── ensemble_global_lgbm.ipynb    # Models exploratoris (Ensemble, Global LightGBM)
│   ├── evaluation.ipynb              # Comparativa global, rànquings, WFV vs backtest
│   └── forecast_2026.ipynb           # Previsió 2026 amb el guanyador del backtest
├── src/
│   ├── evaluation.py             # compute_metrics, walk_forward_cv_full
│   ├── features.py               # build_ml_features, get_feature_columns
│   ├── models.py                 # Wrappers dels models
│   └── plotting.py               # Helpers de plot reutilitzables
└── requirements.txt
```

## Reproductibilitat

Requisits: Python 3.10+, `pip`. Recomanat: entorn virtual.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter notebook
```

Ordre d'execució dels notebooks:

1. `data_preparation.ipynb` (genera els CSVs a `data/processed/`)
2. `eda.ipynb`
3. `models/deterministic/deterministic.ipynb` (SeasonalNaive + MA_12 + OLS_Estacional + OLS_Complet)
4. `models/stochastic/ets.ipynb`, `sarima.ipynb`, `tbats.ipynb`
5. `models/advanced/lgbm.ipynb`, `lstm.ipynb`
6. `ensemble_global_lgbm.ipynb`
7. `evaluation.ipynb`
8. `forecast_2026.ipynb`

Cada notebook persisteix forecasts i mètriques a `data/processed/results/` i figures a `figures/`. L'`evaluation.ipynb` agrega tots els resultats; `forecast_2026.ipynb` consumeix `model_ranking_total.csv` i les mètriques per sèrie per triar el model guanyador del backtest.

## Referències

- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3a ed.). OTexts. Capítol 5.10 (walk-forward validation).
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54-74.
</content>
</invoke>