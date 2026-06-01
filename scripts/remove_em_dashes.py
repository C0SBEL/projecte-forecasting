"""Remove em-dashes (—) from notebook code cells, replacing with appropriate punctuation."""
import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / 'notebooks'


def fix_string(s):
    """Replace em-dashes with sensible alternatives based on context."""
    # Common patterns
    s = s.replace(' — ', ': ')
    s = s.replace(' —', ':')
    s = s.replace('— ', ': ')
    s = s.replace('—', ':')
    return s


def main():
    nbs = list(NB_DIR.rglob('*.ipynb'))
    for nb_path in nbs:
        nb = json.loads(nb_path.read_text(encoding='utf-8'))
        changed = False
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            new_source = []
            for line in cell['source']:
                if '—' in line:
                    line2 = fix_string(line)
                    if line2 != line:
                        changed = True
                    new_source.append(line2)
                else:
                    new_source.append(line)
            cell['source'] = new_source
        if changed:
            nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n',
                               encoding='utf-8')
            print(f'  fixed: {nb_path.relative_to(NB_DIR.parent)}')


if __name__ == '__main__':
    main()
