import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
