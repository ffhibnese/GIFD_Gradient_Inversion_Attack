"""Statistical comparison of reconstruction metrics across attack runs.

`rec_mult.py` appends one row per reconstructed image to
`<output_dir>/<exp_name>/table_Metrics.csv`. Run every attack you want to compare
(several times if you want to report a standard deviation) and then point this
script at the resulting tables:

    python tools/stat_test.py \
        --runs GIFD=results/ex1_gifd/table_Metrics.csv \
               GIAS=results/ex1_gias/table_Metrics.csv \
               GGL=results/ex1_ggl/table_Metrics.csv \
        --metric psnr --reference GIFD

The script reports mean +- std for every method and runs a paired t-test of each
baseline against the reference method. Runs are paired on `target_id` when the
column is present, and on row order otherwise.

The metric column is picked automatically: columns whose name starts with
`Best_` are preferred (they hold the output selected by the least gradient
matching loss), otherwise the single column matching the metric is used. Pass
`--column` when several candidates exist.
"""

import argparse
import sys

import pandas as pd
from scipy import stats

# Column suffixes written by rec_mult.py.
METRIC_SUFFIX = {
    'psnr': '_psnr',
    'lpips': '_lpips(vgg)',
    'lpips_alex': '_lpips(alex)',
    'ssim': '_ssim',
    'mse': '_mse_i',
}


def parse_runs(pairs):
    runs = {}
    for item in pairs:
        if '=' not in item:
            raise SystemExit(f'--runs entries must look like NAME=PATH, got {item!r}')
        name, path = item.split('=', 1)
        runs[name] = path
    return runs


def pick_column(df, metric, column, name):
    if column:
        if column not in df.columns:
            raise SystemExit(f'[{name}] no column named {column!r}')
        return column

    suffix = METRIC_SUFFIX[metric]
    cands = [c for c in df.columns if c.lower().endswith(suffix.lower())]
    if not cands:
        raise SystemExit(f'[{name}] no column ending with {suffix!r}')

    best = [c for c in cands if c.startswith('Best_')]
    if len(best) == 1:
        return best[0]
    if len(cands) == 1:
        return cands[0]
    raise SystemExit(f'[{name}] ambiguous columns {cands}; pick one with --column')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True, metavar='NAME=PATH',
                        help='one or more NAME=PATH pairs pointing at table_Metrics.csv files')
    parser.add_argument('--metric', default='psnr', choices=sorted(METRIC_SUFFIX))
    parser.add_argument('--column', default=None, help='explicit metric column to use')
    parser.add_argument('--reference', default=None, help='method the baselines are tested against')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--out', default=None, help='optional path to also write the table as CSV')
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    reference = args.reference or next(iter(runs))

    series = {}
    for name, path in runs.items():
        df = pd.read_csv(path)
        col = pick_column(df, args.metric, args.column, name)
        values = pd.to_numeric(df[col], errors='coerce')
        # -1 is used by rec_mult.py for layers that were skipped
        values = values[values > -1]
        if 'target_id' in df.columns:
            values.index = pd.Index(df.loc[values.index, 'target_id'], name='target_id')
        else:
            values.index = pd.Index(range(len(values)), name='target_id')
        series[name] = values.dropna()
        print(f'[{name}] column={col!r}  n={len(series[name])}')

    if 'target_id' not in runs and len({len(v) for v in series.values()}) > 1:
        print('warning: runs have different lengths and no target_id column, '
              'rows are paired by position', file=sys.stderr)

    table = pd.concat(series, axis=1)
    if reference in table.columns:
        table = table.dropna(subset=[reference])

    print()
    rows = []
    for name in runs:
        v = table[name].dropna()
        mean, std, n = v.mean(), v.std(ddof=1), len(v)
        row = {'method': name, 'n': n, 'mean': mean, 'std': std}

        if name != reference and reference in table.columns:
            paired = pd.concat([table[reference], v], axis=1).dropna()
            if len(paired) > 1:
                t, p = stats.ttest_rel(paired.iloc[:, 0], paired.iloc[:, 1])
                row['t_stat'] = t
                row['p_value'] = p
                row[f'significant@{args.alpha}'] = bool(p < args.alpha)
        rows.append(row)

    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    if args.out:
        result.to_csv(args.out, index=False)
        print(f'\nwritten to {args.out}')


if __name__ == '__main__':
    main()
