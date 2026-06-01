"""Regenerate WFV per-tech figures using the saved *_wfv_*.csv files.

Useful after canviar el helper de plotting sense voler re-executar tot
el WFV (que pot trigar hores). Llegeix els CSVs ja generats i refà només
les figures fig_{model}_wfv_per_tech.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from features import TECNOLOGIES_MODELITZABLES
from plotting import plot_wfv_per_tech, format_ts_ax

DATA_DIR    = ROOT / 'data' / 'processed'
RESULTS_DIR = DATA_DIR / 'results'
FIGURES_DIR = ROOT / 'figures'

TECNOLOGIES = TECNOLOGIES_MODELITZABLES + ['Total']
METRICS_4 = ['RMSE', 'MAE', 'MAPE', 'MASE']


def load_wfv(prefix):
    metrics   = pd.read_csv(RESULTS_DIR / f'{prefix}_wfv_metrics.csv',
                            parse_dates=['fold_start'])
    forecasts = pd.read_csv(RESULTS_DIR / f'{prefix}_wfv_forecasts.csv',
                            parse_dates=['fold_start', 'date'])
    return metrics.dropna(subset=['MASE']), forecasts


def main():
    pivot_train = pd.read_csv(DATA_DIR / 'pivot_train.csv',
                              index_col='date', parse_dates=True)
    pivot_test  = pd.read_csv(DATA_DIR / 'pivot_test.csv',
                              index_col='date', parse_dates=True)
    pivot_full  = pd.concat([pivot_train, pivot_test]).sort_index()

    # Single-model notebooks: use the standard helper
    single_models = {
        'lgbm':   ('LightGBM',   '#1a9641'),
        'sarima': ('AutoSARIMA', '#f46d43'),
        'tbats':  ('TBATS',      '#7b2d8b'),
        'lstm':   ('LSTM',       '#9e44b0'),
    }
    for prefix, (display_name, color) in single_models.items():
        metrics, forecasts = load_wfv(prefix)
        agg = (metrics.groupby('Tecnologia')[METRICS_4]
               .mean().round(2).reset_index())
        save_path = FIGURES_DIR / f'fig_{prefix}_wfv_per_tech.png'
        plot_wfv_per_tech(pivot_full, forecasts, agg,
                          tecnologies=TECNOLOGIES,
                          model_name=display_name,
                          save_path=save_path,
                          color_pred=color)
        plt.close('all')
        print(f'  fig_{prefix}_wfv_per_tech.png')

    # Baseline: multi-model, custom layout
    from matplotlib.lines import Line2D
    metrics, forecasts = load_wfv('baseline')
    agg = (metrics.groupby(['Tecnologia', 'Model'])[METRICS_4]
           .mean().round(2).reset_index())
    colors_wfv = {'SeasonalNaive': '#aaaaaa', 'ETS_Additiu': '#2166ac',
                  'ETS_Multiplicatiu': '#7b2d8b'}

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes_flat = axes.flatten()
    for i, tech in enumerate(TECNOLOGIES):
        ax = axes_flat[i]
        s = pivot_full[tech]
        ax.plot(s.index, s.values, color='#2c7bb6', lw=2, zorder=10)
        for model_name, c in colors_wfv.items():
            sub = forecasts[(forecasts['Tecnologia'] == tech)
                            & (forecasts['Model'] == model_name)
                            & (forecasts['h'] == 1)].sort_values('date')
            ax.plot(pd.to_datetime(sub['date']), sub['predicted'],
                    color=c, lw=1.5, ls='--', marker='o', ms=4, alpha=0.85)
        rows = agg[agg['Tecnologia'] == tech].sort_values('MASE')
        if len(rows):
            best = rows.iloc[0]
            ax.set_title(
                f'{tech}\nmillor WFV: {best["Model"]}  RMSE={best["RMSE"]:.0f}  '
                f'MAE={best["MAE"]:.0f}  MAPE={best["MAPE"]:.1f}%  MASE={best["MASE"]:.2f}',
                fontsize=9)
        else:
            ax.set_title(tech, fontsize=9)
        format_ts_ax(ax)

    axes_flat[-1].axis('off')
    legend = [Line2D([0], [0], color='#2c7bb6', lw=2, label='Observat')]
    for model_name, c in colors_wfv.items():
        legend.append(Line2D([0], [0], color=c, lw=1.5, ls='--', marker='o',
                             ms=4, label=model_name))
    axes_flat[-1].legend(handles=legend, loc='center', fontsize=11,
                         title='predicció h=1 per fold')
    plt.suptitle('Baselines: Walk-forward per tecnologia '
                 '(h=1 per fold, 13 origins)', fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_baseline_wfv_per_tech.png', bbox_inches='tight')
    plt.close('all')
    print('  fig_baseline_wfv_per_tech.png')


if __name__ == '__main__':
    main()
