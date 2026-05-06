from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_figures(release_root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = release_root / 'data' / 'aggregate_metrics.parquet'
    dir_path = release_root / 'data' / 'aggregate_metrics'
    if flat.is_file():
        metrics = pd.read_parquet(flat)
    elif dir_path.exists():
        metrics = pd.read_parquet(dir_path)
    else:
        metrics = pd.DataFrame()

    figure2 = []
    if not metrics.empty:
        for mode in ['direct', 'transitive']:
            sub = metrics[metrics['dependency_mode'] == mode]
            cols = [c for c in ['benchmark_id', 'model_family', 'f1', 'coverage'] if c in sub.columns]
            for _, row in sub[cols].iterrows():
                figure2.append({
                    'dependency_mode': mode,
                    'benchmark_id': row.get('benchmark_id'),
                    'model_family': row.get('model_family'),
                    'coverage': round(float(row.get('coverage', 0)), 3),
                    'f1': round(float(row.get('f1', 0)), 3),
                })

    payload = {
        'figure2_coverage_vs_f1': figure2,
    }

    (out_dir / 'figure2_coverage_vs_f1.json').write_text(
        json.dumps(figure2, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    (out_dir / 'figure_data.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return {'out_dir': str(out_dir), 'figure2_points': len(figure2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    res = build_figures(Path(args.release_root), Path(args.out))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
