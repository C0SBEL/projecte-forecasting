import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def format_ts_ax(ax, minor_months=(4, 7, 10)):
    """Consistent time-series x-axis: bold year labels + quarterly minor ticks."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=list(minor_months)))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    ax.tick_params(axis='x', which='major', labelsize=11, pad=14,
                   length=6, width=1.2, labelcolor='#222222')
    ax.tick_params(axis='x', which='minor', labelsize=8, pad=3,
                   length=3, color='#999999', labelcolor='#666666')
    ax.xaxis.grid(True, which='major', color='#aaaaaa', alpha=0.4, lw=0.8, zorder=0)
    ax.yaxis.grid(True, color='#cccccc', alpha=0.5, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _title_4metrics(tech, m):
    return (f'{tech}\nRMSE={m["RMSE"]:.0f}  MAE={m["MAE"]:.0f}  '
            f'MAPE={m["MAPE"]:.1f}%  MASE={m["MASE"]:.2f}')


def plot_backtest_per_tech(pivot_train, pivot_test, df_forecasts, df_metrics,
                           tecnologies, model_name, save_path=None,
                           color_pred='#d7191c', tail_train=24):
    """4x2 grid: per-tech backtest visualization with 4 metrics in title."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes_flat = axes.flatten()

    for i, tech in enumerate(tecnologies):
        ax = axes_flat[i]
        y_train_plot = pivot_train[tech].iloc[-tail_train:]
        y_test_plot  = pivot_test[tech]
        observed_dates = list(y_train_plot.index) + list(y_test_plot.index)
        observed_vals  = list(y_train_plot.values) + list(y_test_plot.values)
        ax.plot(observed_dates, observed_vals, color='#2c7bb6', lw=2,
                label='Observat', zorder=10)
        ax.axvspan(pivot_test.index[0], pivot_test.index[-1],
                   alpha=0.08, color='gray')
        ax.axvline(pivot_test.index[0], color='gray', ls=':', lw=1)

        sub = df_forecasts[df_forecasts['Tecnologia'] == tech].sort_values('date')
        ax.plot(pd.to_datetime(sub['date']), sub['predicted'],
                color=color_pred, lw=1.8, ls='--', marker='o', ms=4,
                label=model_name)

        row = df_metrics[df_metrics['Tecnologia'] == tech]
        if len(row):
            ax.set_title(_title_4metrics(tech, row.iloc[0]), fontsize=9)
        else:
            ax.set_title(tech, fontsize=9)
        format_ts_ax(ax)

    # Last unused subplot → legend
    for j in range(len(tecnologies), len(axes_flat)):
        axes_flat[j].axis('off')
        if j == len(axes_flat) - 1:
            handles, labels = axes_flat[0].get_legend_handles_labels()
            axes_flat[j].legend(handles, labels, loc='center', fontsize=11)

    plt.suptitle(f'{model_name} — Backtesting per tecnologia (test 2025)',
                 fontsize=13, y=1.0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_wfv_per_tech(pivot_full, df_wfv_forecasts, df_wfv_metrics_agg,
                      tecnologies, model_name, save_path=None,
                      color_pred='#7b2d8b'):
    """4x2 grid: per-tech WFV visualization.

    For each tech: real series (full history) + WFV predictions superposed
    (one thin dashed line per fold). Title shows mean 4 metrics over folds.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes_flat = axes.flatten()

    for i, tech in enumerate(tecnologies):
        ax = axes_flat[i]
        s = pivot_full[tech]
        ax.plot(s.index, s.values, color='#2c7bb6', lw=2,
                label='Observat', zorder=10)

        sub = df_wfv_forecasts[df_wfv_forecasts['Tecnologia'] == tech]
        for fold_start, group in sub.groupby('fold_start'):
            g = group.sort_values('h')
            ax.plot(pd.to_datetime(g['date']), g['predicted'],
                    color=color_pred, lw=1.0, ls='--', alpha=0.45)

        row = df_wfv_metrics_agg[df_wfv_metrics_agg['Tecnologia'] == tech]
        if len(row):
            ax.set_title(_title_4metrics(tech, row.iloc[0]), fontsize=9)
        else:
            ax.set_title(tech, fontsize=9)
        format_ts_ax(ax)

    for j in range(len(tecnologies), len(axes_flat)):
        axes_flat[j].axis('off')
        if j == len(axes_flat) - 1:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color='#2c7bb6', lw=2, label='Observat'),
                Line2D([0], [0], color=color_pred, lw=1.2, ls='--',
                       label=f'{model_name} — un fold per línia'),
            ]
            axes_flat[j].legend(handles=handles, loc='center', fontsize=11)

    plt.suptitle(f'{model_name} — Walk-forward per tecnologia '
                 f'(13 folds × 12 horitzons)', fontsize=13, y=1.0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    return fig
