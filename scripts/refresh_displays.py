"""Re-executa un notebook saltant les cel·les de WFV (que carreguen CSV en lloc).

Manté el codi font original de les cel·les WFV i els seus outputs originals;
només refresca les cel·les que vénen abans (backtest fit) i després (display
+ figures que ja teníem com a obsoletes). Estalvia hores de re-fit del WFV.

Ús:
    python scripts/refresh_displays.py notebooks/models/sarima.ipynb
"""
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

WFV_MARKER = "walk_forward_cv_full("

SUBSTITUTE_TEMPLATES = {
    'baseline': """# WFV skipped — load from CSV per refrescar només els displays
pivot_full = pd.concat([pivot_train, pivot_test]).sort_index()
df_wfv_metrics = pd.read_csv(RESULTS_DIR / 'baseline_wfv_metrics.csv', parse_dates=['fold_start'])
df_wfv_forecasts = pd.read_csv(RESULTS_DIR / 'baseline_wfv_forecasts.csv', parse_dates=['fold_start', 'date'])
wfv_agg = (df_wfv_metrics.dropna(subset=['MASE'])
           .groupby(['Tecnologia', 'Model'])[['RMSE', 'MAE', 'MAPE', 'MASE']]
           .mean().round(2).reset_index())
print('WFV carregat des de CSV')""",
}

def single_model_template(name):
    return f"""# WFV skipped — load from CSV per refrescar només els displays
pivot_full = pd.concat([pivot_train, pivot_test]).sort_index()
df_wfv_metrics = pd.read_csv(RESULTS_DIR / '{name}_wfv_metrics.csv', parse_dates=['fold_start'])
df_wfv_forecasts = pd.read_csv(RESULTS_DIR / '{name}_wfv_forecasts.csv', parse_dates=['fold_start', 'date'])
wfv_agg = (df_wfv_metrics.dropna(subset=['MASE'])
           .groupby('Tecnologia')[['RMSE', 'MAE', 'MAPE', 'MASE']]
           .mean().round(2).reset_index())
print('WFV carregat des de CSV')"""


def main(nb_path):
    nb_path = Path(nb_path)
    name = nb_path.stem
    if name == 'baseline':
        substitute = SUBSTITUTE_TEMPLATES['baseline']
    else:
        substitute = single_model_template(name)

    nb = nbformat.read(nb_path, as_version=4)

    wfv_cells = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and WFV_MARKER in cell.source:
            wfv_cells.append((
                i,
                cell.source,
                list(cell.get('outputs', [])),
                cell.get('execution_count'),
            ))
            cell.source = substitute
            cell['outputs'] = []
            cell['execution_count'] = None

    print(f'{nb_path.name}: substituïdes {len(wfv_cells)} cel·les WFV')

    client = NotebookClient(nb, kernel_name='python3', timeout=600)
    client.execute(cwd=str(nb_path.parent))

    # Restore originals
    for idx, orig_source, orig_outputs, orig_exec in wfv_cells:
        nb.cells[idx].source = orig_source
        nb.cells[idx]['outputs'] = orig_outputs
        nb.cells[idx]['execution_count'] = orig_exec

    nbformat.write(nb, nb_path)
    print(f'{nb_path.name}: done')


if __name__ == '__main__':
    main(sys.argv[1])
