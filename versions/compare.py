"""Comparison report generator — reads summary_metrics.json from all versions."""

import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSION_NAMES = [
    'v0_baseline', 'v1_physics_constrained', 'v2_physics_encoder',
    'v3_physics_corrector', 'v4_neural_ode', 'v5_coordinate_pinn',
    'v6_fno_decoder', 'v7_deeponet',
    'v8_curriculum_pinn', 'v9_pressure_only_pinn', 'v10_enhanced_pinn',
    'v11_well_masked_physics', 'v12_dual_latent', 'v13_physics_finetune',
    'v14_latent_physics', 'v15_pino', 'v16_obs_weighted_physics',
    'v17_finetune_ode', 'v18_finetune_fno', 'v19_perwell_mlp',
    'v20_multistep_rollout', 'v21_well_features', 'v22_full_combo',
    'v23_baseline_bs4', 'v24_baseline_bs8', 'v25_baseline_bs32',
]

SHORT_NAMES = [
    'V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
    'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
    'V17', 'V18', 'V19', 'V20', 'V21', 'V22',
    'V23', 'V24', 'V25',
]

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
    '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173', '#5254a3',
    '#e7969c', '#de9ed6', '#9c9ede',
]


def load_metrics(base_dir='.'):
    """Load summary_metrics.json from each version's output directory."""
    metrics = {}
    for vname in VERSION_NAMES:
        json_path = os.path.join(base_dir, f'outputs_{vname}', 'logs', 'summary_metrics.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metrics[vname] = json.load(f)
            print(f"  Loaded: {vname}")
        else:
            print(f"  Missing: {vname} ({json_path})")
    return metrics


def print_summary_table(metrics):
    """Print comparison table to terminal."""
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)

    header = (f"{'Version':<25} {'Sat RMSE':>10} {'Sat MAE':>10} {'Pres RMSE':>12} "
              f"{'Pres MAE':>12} {'Avg R²':>8} {'R²(≥0)':>8} {'#Wells':>6} "
              f"{'Time(s)':>10} {'Final Loss':>12}")
    print(header)
    print("-" * 115)

    for vname in VERSION_NAMES:
        if vname not in metrics:
            continue
        m = metrics[vname]
        t = m.get('train_time_seconds')
        t_str = f"{t:.0f}" if t is not None else "N/A"
        fl = m.get('final_train_loss')
        fl_str = f"{fl:.4f}" if fl is not None else "N/A"

        # Filtered R²: exclude negative wells
        r2_all = list(m.get('well_r2', {}).values())
        r2_good = [v for v in r2_all if v >= 0]
        avg_r2_filt = np.mean(r2_good) if r2_good else 0.0
        n_good = len(r2_good)

        print(f"{vname:<25} {m['sat_rmse']:>10.4f} {m['sat_mae']:>10.4f} "
              f"{m['pres_rmse_psia']:>12.2f} {m['pres_mae_psia']:>12.2f} "
              f"{m['avg_well_r2']:>8.3f} {avg_r2_filt:>8.3f} {n_good:>6} "
              f"{t_str:>10} {fl_str:>12}")

    print("=" * 115)


def save_summary_csv(metrics, out_path):
    """Save comparison table as CSV."""
    rows = []
    for vname in VERSION_NAMES:
        if vname not in metrics:
            continue
        m = metrics[vname]
        r2_all = list(m.get('well_r2', {}).values())
        r2_good = [v for v in r2_all if v >= 0]
        rows.append({
            'version': vname,
            'sat_rmse': m['sat_rmse'],
            'sat_mae': m['sat_mae'],
            'pres_rmse_psia': m['pres_rmse_psia'],
            'pres_mae_psia': m['pres_mae_psia'],
            'avg_well_r2': m['avg_well_r2'],
            'avg_well_r2_filtered': np.mean(r2_good) if r2_good else 0.0,
            'n_good_wells': len(r2_good),
            'train_time_seconds': m.get('train_time_seconds'),
            'final_train_loss': m.get('final_train_loss'),
            'epochs': m.get('epochs'),
            'batch_size': m.get('batch_size'),
        })
        for wname, r2val in m.get('well_r2', {}).items():
            rows[-1][f'r2_{wname}'] = r2val

    if not rows:
        return

    keys = list(rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {out_path}")


def plot_training_curves_overlay(metrics, base_dir='.', out_dir='outputs_comparison'):
    """Overlay training curves from all versions."""
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, vname in enumerate(VERSION_NAMES):
        csv_path = os.path.join(base_dir, f'outputs_{vname}', 'logs', 'loss_history.csv')
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, 'r') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        epochs = [int(r['epoch']) for r in rows]
        losses = [float(r['train_loss']) for r in rows]
        short = SHORT_NAMES[i] if i < len(SHORT_NAMES) else vname
        ax.plot(epochs, losses, color=COLORS[i % len(COLORS)],
                linewidth=2, label=short, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Training Loss', fontsize=14)
    ax.set_title('Training Loss Comparison (All Versions)', fontsize=16)
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves_overlay.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot: training_curves_overlay.png")


def plot_per_timestep_comparison(metrics, out_dir='outputs_comparison'):
    """Per-timestep RMSE grouped bar chart."""
    versions_with_data = [(vn, metrics[vn]) for vn in VERSION_NAMES
                          if vn in metrics and 'sat_rmse_per_timestep' in metrics[vn]]
    if not versions_with_data:
        return

    n_versions = len(versions_with_data)
    n_timesteps = len(versions_with_data[0][1]['sat_rmse_per_timestep'])
    x = np.arange(n_timesteps)
    width = 0.8 / n_versions

    # Saturation
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for i, (vname, m) in enumerate(versions_with_data):
        idx = VERSION_NAMES.index(vname)
        short = SHORT_NAMES[idx] if idx < len(SHORT_NAMES) else vname
        axes[0].bar(x + i * width, m['sat_rmse_per_timestep'], width,
                    label=short, color=COLORS[idx % len(COLORS)], alpha=0.8)
        axes[1].bar(x + i * width, m['pres_rmse_per_timestep'], width,
                    label=short, color=COLORS[idx % len(COLORS)], alpha=0.8)

    axes[0].set_xlabel('Timestep'); axes[0].set_ylabel('RMSE')
    axes[0].set_title('Saturation RMSE per Timestep')
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel('Timestep'); axes[1].set_ylabel('RMSE (psia)')
    axes[1].set_title('Pressure RMSE per Timestep')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_timestep_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot: per_timestep_comparison.png")


def plot_r2_comparison(metrics, out_dir='outputs_comparison'):
    """Well R² grouped bar chart."""
    versions_with_data = [(vn, metrics[vn]) for vn in VERSION_NAMES
                          if vn in metrics and 'well_r2' in metrics[vn]]
    if not versions_with_data:
        return

    well_names = list(versions_with_data[0][1]['well_r2'].keys())
    n_wells = len(well_names)
    n_versions = len(versions_with_data)
    x = np.arange(n_wells)
    width = 0.8 / n_versions

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (vname, m) in enumerate(versions_with_data):
        idx = VERSION_NAMES.index(vname)
        short = SHORT_NAMES[idx] if idx < len(SHORT_NAMES) else vname
        r2_vals = [m['well_r2'].get(w, 0) for w in well_names]
        ax.bar(x + i * width, r2_vals, width,
               label=short, color=COLORS[idx % len(COLORS)], alpha=0.8)

    ax.set_xlabel('Well', fontsize=14)
    ax.set_ylabel('R²', fontsize=14)
    ax.set_title('Well R² Comparison', fontsize=16)
    ax.set_xticks(x + width * n_versions / 2)
    ax.set_xticklabels(well_names, rotation=45)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'r2_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot: r2_comparison.png")


def plot_r2_comparison_filtered(metrics, out_dir='outputs_comparison'):
    """Well R² comparison — only wells with R² >= 0 (removes failed wells)."""
    versions_with_data = [(vn, metrics[vn]) for vn in VERSION_NAMES
                          if vn in metrics and 'well_r2' in metrics[vn]]
    if not versions_with_data:
        return

    well_names = list(versions_with_data[0][1]['well_r2'].keys())
    n_versions = len(versions_with_data)

    # For each version, filter to wells with R² >= 0
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: grouped bar chart with only non-negative R² wells
    # Find wells where ALL versions have R² >= 0
    common_good_wells = []
    for w in well_names:
        all_good = all(m.get('well_r2', {}).get(w, -1) >= 0
                       for _, m in versions_with_data)
        if all_good:
            common_good_wells.append(w)

    if common_good_wells:
        n_wells = len(common_good_wells)
        x = np.arange(n_wells)
        width = 0.8 / n_versions

        for i, (vname, m) in enumerate(versions_with_data):
            idx = VERSION_NAMES.index(vname)
            short = SHORT_NAMES[idx] if idx < len(SHORT_NAMES) else vname
            r2_vals = [m['well_r2'].get(w, 0) for w in common_good_wells]
            axes[0].bar(x + i * width, r2_vals, width,
                       label=short, color=COLORS[idx % len(COLORS)], alpha=0.8)

        axes[0].set_xlabel('Well', fontsize=14)
        axes[0].set_ylabel('R²', fontsize=14)
        axes[0].set_title('Well R² (only wells with R² ≥ 0 across all versions)', fontsize=13)
        axes[0].set_xticks(x + width * n_versions / 2)
        axes[0].set_xticklabels(common_good_wells, rotation=45)
        axes[0].legend(fontsize=9)
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(True, alpha=0.3, axis='y')

    # Right: average R² per version (excluding negative wells per version)
    labels = []
    avg_r2_filtered = []
    n_good_wells = []
    for vname, m in versions_with_data:
        idx = VERSION_NAMES.index(vname)
        labels.append(SHORT_NAMES[idx] if idx < len(SHORT_NAMES) else vname)
        r2_all = list(m.get('well_r2', {}).values())
        r2_good = [v for v in r2_all if v >= 0]
        avg_r2_filtered.append(np.mean(r2_good) if r2_good else 0)
        n_good_wells.append(len(r2_good))

    x = np.arange(len(labels))
    colors_bar = [COLORS[VERSION_NAMES.index(vn) % len(COLORS)] for vn, _ in versions_with_data]
    bars = axes[1].bar(x, avg_r2_filtered, color=colors_bar, alpha=0.8)
    for bar, n in zip(bars, n_good_wells):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{n} wells', ha='center', va='bottom', fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title('Average R² (excluding negative R² wells)', fontsize=13)
    axes[1].set_ylabel('Average R²')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'r2_comparison_filtered.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot: r2_comparison_filtered.png")


def plot_radar_chart(metrics, out_dir='outputs_comparison'):
    """Multi-metric normalized radar chart."""
    versions_with_data = [(vn, metrics[vn]) for vn in VERSION_NAMES if vn in metrics]
    if len(versions_with_data) < 2:
        return

    # Metrics to compare (lower is better for errors, higher for R2)
    metric_names = ['sat_rmse', 'pres_rmse_psia', 'avg_well_r2']
    display_names = ['Sat RMSE\n(lower=better)', 'Pres RMSE\n(lower=better)',
                     'Avg Well R²\n(higher=better)']
    invert = [True, True, False]  # True = lower is better

    # Collect raw values
    raw = np.zeros((len(versions_with_data), len(metric_names)))
    for i, (vn, m) in enumerate(versions_with_data):
        for j, mn in enumerate(metric_names):
            raw[i, j] = m.get(mn, 0)

    # Normalize to [0, 1] where 1 is best
    normalized = np.zeros_like(raw)
    for j in range(len(metric_names)):
        col = raw[:, j]
        if col.max() == col.min():
            normalized[:, j] = 1.0
        elif invert[j]:
            normalized[:, j] = 1 - (col - col.min()) / (col.max() - col.min())
        else:
            normalized[:, j] = (col - col.min()) / (col.max() - col.min())

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for i, (vn, m) in enumerate(versions_with_data):
        idx = VERSION_NAMES.index(vn)
        short = SHORT_NAMES[idx] if idx < len(SHORT_NAMES) else vn
        vals = normalized[i].tolist() + [normalized[i, 0]]
        ax.plot(angles, vals, 'o-', linewidth=2, label=short,
                color=COLORS[idx % len(COLORS)])
        ax.fill(angles, vals, alpha=0.1, color=COLORS[idx % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('Multi-Metric Comparison (normalized)', fontsize=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'radar_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot: radar_comparison.png")


def plot_metric_bars(metrics, out_dir='outputs_comparison'):
    """Simple bar charts of key metrics across versions."""
    versions_with_data = [(vn, metrics[vn]) for vn in VERSION_NAMES if vn in metrics]
    if not versions_with_data:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    labels = []
    sat_rmse, pres_rmse, avg_r2, train_time = [], [], [], []
    for vn, m in versions_with_data:
        idx = VERSION_NAMES.index(vn)
        labels.append(SHORT_NAMES[idx] if idx < len(SHORT_NAMES) else vn)
        sat_rmse.append(m['sat_rmse'])
        pres_rmse.append(m['pres_rmse_psia'])
        avg_r2.append(m['avg_well_r2'])
        t = m.get('train_time_seconds')
        train_time.append(t / 60 if t is not None else 0)

    x = np.arange(len(labels))
    colors = [COLORS[VERSION_NAMES.index(vn) % len(COLORS)] for vn, _ in versions_with_data]

    axes[0, 0].bar(x, sat_rmse, color=colors)
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_title('Saturation RMSE'); axes[0, 0].set_ylabel('RMSE')

    axes[0, 1].bar(x, pres_rmse, color=colors)
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_title('Pressure RMSE (psia)'); axes[0, 1].set_ylabel('RMSE (psia)')

    axes[1, 0].bar(x, avg_r2, color=colors)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_title('Average Well R²'); axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_ylim(0, 1.05)

    axes[1, 1].bar(x, train_time, color=colors)
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_title('Training Time'); axes[1, 1].set_ylabel('Minutes')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'metric_bars.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot: metric_bars.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare all E2CO versions")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory containing outputs_v*/ directories")
    parser.add_argument("--out_dir", type=str, default="outputs_comparison",
                        help="Output directory for comparison results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading metrics from all versions...")
    metrics = load_metrics(args.base_dir)

    if not metrics:
        print("No metrics found. Run training first.")
        return

    # Summary table
    print_summary_table(metrics)

    # CSV
    save_summary_csv(metrics, os.path.join(args.out_dir, 'comparison_summary.csv'))

    # Plots
    print("\nGenerating comparison plots...")
    plot_training_curves_overlay(metrics, args.base_dir, args.out_dir)
    plot_per_timestep_comparison(metrics, args.out_dir)
    plot_r2_comparison(metrics, args.out_dir)
    plot_r2_comparison_filtered(metrics, args.out_dir)
    plot_radar_chart(metrics, args.out_dir)
    plot_metric_bars(metrics, args.out_dir)

    print(f"\nAll comparison outputs saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
