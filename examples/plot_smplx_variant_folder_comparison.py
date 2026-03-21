#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from smplx.joint_names import JOINT_NAMES


BASE_VARIANT_FILE = 'motion_shape.npz'
OUTPUT_SUBDIR = 'comparison'
PLOT_DPI = 160
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
VARIANT_CMAP = 'tab10'
BASE_COLOR = 'black'
BASE_LINEWIDTH = 2.6
VARIANT_LINEWIDTH = 1.3
VARIANT_ALPHA = 0.8

BODY_POSE_JOINT_NAMES = JOINT_NAMES[1:22]
BODY_POSE_JOINT_INDEX = {name: idx for idx, name in enumerate(BODY_POSE_JOINT_NAMES)}
JOINT_INDEX = {name: idx for idx, name in enumerate(JOINT_NAMES)}

SELECTED_BODY_SPECS = [
    ('pelvis_body', ('joint', 'pelvis')),
    ('torso_body', ('midpoint', 'spine3', 'neck')),
    ('head_body', ('midpoint', 'neck', 'head')),
    ('left_forearm_body', ('midpoint', 'left_elbow', 'left_wrist')),
    ('right_forearm_body', ('midpoint', 'right_elbow', 'right_wrist')),
    ('left_shank_body', ('midpoint', 'left_knee', 'left_ankle')),
    ('right_shank_body', ('midpoint', 'right_knee', 'right_ankle')),
]

SELECTED_DOF_SPECS = [
    ('root_rx', ('root_orient', 0)),
    ('root_ry', ('root_orient', 1)),
    ('root_rz', ('root_orient', 2)),
    ('left_hip_rx', ('pose_body', 3 * BODY_POSE_JOINT_INDEX['left_hip'] + 0)),
    ('right_hip_rx', ('pose_body', 3 * BODY_POSE_JOINT_INDEX['right_hip'] + 0)),
    ('left_knee_rx', ('pose_body', 3 * BODY_POSE_JOINT_INDEX['left_knee'] + 0)),
    ('right_knee_rx', ('pose_body', 3 * BODY_POSE_JOINT_INDEX['right_knee'] + 0)),
    ('left_shoulder_rz', ('pose_body', 3 * BODY_POSE_JOINT_INDEX['left_shoulder'] + 2)),
    ('right_shoulder_rz', ('pose_body', 3 * BODY_POSE_JOINT_INDEX['right_shoulder'] + 2)),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare SMPL-X motion-shape variants inside one output folder.'
    )
    parser.add_argument('folder', type=str,
                        help='Folder that contains motion_shape.npz and random_shape_*.npz files.')
    return parser.parse_args()


def to_python(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_npz_dict(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def resolve_variant_paths(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f'Folder does not exist: {folder}')

    base_path = folder / BASE_VARIANT_FILE
    if not base_path.exists():
        raise FileNotFoundError(f'Expected base file at {base_path}')

    manifest_path = folder / 'manifest.json'
    variant_paths = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name in manifest.get('variants', []):
            path = folder / f'{name}.npz'
            if path.exists() and path.name != BASE_VARIANT_FILE:
                variant_paths.append(path)

    if not variant_paths:
        variant_paths = sorted(
            path for path in folder.glob('*.npz')
            if path.name != BASE_VARIANT_FILE
        )

    if not variant_paths:
        raise ValueError(f'No variant npz files found in {folder}')

    return folder, base_path, variant_paths


def validate_variant(base_data, variant_data, variant_path):
    for key in ['joints', 'root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']:
        if key not in variant_data:
            raise KeyError(f'Missing {key!r} in {variant_path}')
        if variant_data[key].shape != base_data[key].shape:
            raise ValueError(
                f'Shape mismatch for {key!r} in {variant_path}: '
                f'{variant_data[key].shape} vs base {base_data[key].shape}'
            )


def extract_body_series(joints):
    series = {}
    for name, spec in SELECTED_BODY_SPECS:
        kind = spec[0]
        if kind == 'joint':
            joint_name = spec[1]
            series[name] = joints[:, JOINT_INDEX[joint_name], :].astype(np.float32)
        elif kind == 'midpoint':
            joint_a = joints[:, JOINT_INDEX[spec[1]], :]
            joint_b = joints[:, JOINT_INDEX[spec[2]], :]
            series[name] = (0.5 * (joint_a + joint_b)).astype(np.float32)
        else:
            raise ValueError(f'Unknown body spec kind: {kind}')
    return series


def flatten_all_dofs(data):
    parts = [
        data['root_orient'].reshape(data['root_orient'].shape[0], -1),
        data['pose_body'].reshape(data['pose_body'].shape[0], -1),
        data['pose_hand'].reshape(data['pose_hand'].shape[0], -1),
        data['pose_jaw'].reshape(data['pose_jaw'].shape[0], -1),
        data['pose_eye'].reshape(data['pose_eye'].shape[0], -1),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def extract_selected_dof_series(data):
    series = {}
    for name, (field_name, index) in SELECTED_DOF_SPECS:
        field = data[field_name].reshape(data[field_name].shape[0], -1)
        series[name] = field[:, index].astype(np.float32)
    return series


def stack_series(series_dict):
    names = list(series_dict.keys())
    values = np.stack([series_dict[name] for name in names], axis=1)
    return names, values


def compute_variant_metrics(base_data, variant_data, base_body_series, base_all_dofs, base_selected_dofs):
    variant_body_series = extract_body_series(variant_data['joints'])
    variant_all_dofs = flatten_all_dofs(variant_data)
    variant_selected_dofs = extract_selected_dof_series(variant_data)

    _, base_body_stack = stack_series(base_body_series)
    _, variant_body_stack = stack_series(variant_body_series)
    body_error = np.linalg.norm(variant_body_stack - base_body_stack, axis=-1)

    dof_error = np.linalg.norm(variant_all_dofs - base_all_dofs, axis=-1)

    metrics = {
        'variant_name': str(to_python(variant_data.get('variant_name', 'unknown'))),
        'beta_l2_distance': float(np.linalg.norm(variant_data['betas'] - base_data['betas'])),
        'mean_body_position_error': float(body_error.mean()),
        'min_body_position_error': float(body_error.min()),
        'max_body_position_error': float(body_error.max()),
        'mean_joint_dof_error': float(dof_error.mean()),
        'min_joint_dof_error': float(dof_error.min()),
        'max_joint_dof_error': float(dof_error.max()),
        'selected_body_position_mean_errors': {},
        'selected_joint_dof_max_abs_errors': {},
    }

    for name, base_pos in base_body_series.items():
        err = np.linalg.norm(variant_body_series[name] - base_pos, axis=-1)
        metrics['selected_body_position_mean_errors'][name] = float(err.mean())

    for name, base_values in base_selected_dofs.items():
        err = np.abs(variant_selected_dofs[name] - base_values)
        metrics['selected_joint_dof_max_abs_errors'][name] = float(err.max())

    return metrics, variant_body_series, variant_selected_dofs


def variant_color(index, total):
    cmap = plt.get_cmap(VARIANT_CMAP)
    denom = max(total - 1, 1)
    return cmap(index / denom)


def apply_plot_style():
    plt.style.use(PLOT_STYLE)
    plt.rcParams.update({
        'figure.dpi': PLOT_DPI,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })


def make_grid(n_items, max_cols=3):
    ncols = min(max_cols, n_items)
    nrows = math.ceil(n_items / ncols)
    return nrows, ncols


def finalize_grid(fig, axes, used_count):
    flat_axes = np.asarray(axes).reshape(-1)
    for ax in flat_axes[used_count:]:
        ax.axis('off')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_body_trajectories(output_path, title, base_series, variant_series_list):
    names = list(base_series.keys())
    nrows, ncols = make_grid(len(names), max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    fig.suptitle(title)

    for ax, name in zip(axes.reshape(-1), names):
        base = base_series[name]
        ax.plot(base[:, 0], base[:, 2], color=BASE_COLOR, linewidth=BASE_LINEWIDTH, label='base_shape')
        for idx, (variant_name, variant_series) in enumerate(variant_series_list):
            arr = variant_series[name]
            ax.plot(
                arr[:, 0],
                arr[:, 2],
                color=variant_color(idx, len(variant_series_list)),
                linewidth=VARIANT_LINEWIDTH,
                alpha=VARIANT_ALPHA,
                label=variant_name,
            )
        ax.set_title(name)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.set_aspect('equal', adjustable='box')

    finalize_grid(fig, axes, len(names))
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_body_error_over_time(output_path, title, time_axis, base_series, variant_series_list):
    names = list(base_series.keys())
    nrows, ncols = make_grid(len(names), max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.0 * nrows), squeeze=False)
    fig.suptitle(title)

    for ax, name in zip(axes.reshape(-1), names):
        for idx, (variant_name, variant_series) in enumerate(variant_series_list):
            error = np.linalg.norm(variant_series[name] - base_series[name], axis=-1)
            ax.plot(
                time_axis,
                error,
                color=variant_color(idx, len(variant_series_list)),
                linewidth=VARIANT_LINEWIDTH,
                alpha=VARIANT_ALPHA,
                label=variant_name,
            )
        ax.set_title(name)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('body_position_error [m]')

    finalize_grid(fig, axes, len(names))
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_dof_traces(output_path, title, time_axis, base_series, variant_series_list):
    names = list(base_series.keys())
    nrows, ncols = make_grid(len(names), max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.0 * nrows), squeeze=False)
    fig.suptitle(title)

    for ax, name in zip(axes.reshape(-1), names):
        base = base_series[name]
        ax.plot(time_axis, base, color=BASE_COLOR, linewidth=BASE_LINEWIDTH, label='base_shape')
        for idx, (variant_name, variant_series) in enumerate(variant_series_list):
            ax.plot(
                time_axis,
                variant_series[name],
                color=variant_color(idx, len(variant_series_list)),
                linewidth=VARIANT_LINEWIDTH,
                alpha=VARIANT_ALPHA,
                label=variant_name,
            )
        ax.set_title(name)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('axis-angle component [rad]')

    finalize_grid(fig, axes, len(names))
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_dof_error_over_time(output_path, title, time_axis, base_series, variant_series_list):
    names = list(base_series.keys())
    nrows, ncols = make_grid(len(names), max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.0 * nrows), squeeze=False)
    fig.suptitle(title)

    for ax, name in zip(axes.reshape(-1), names):
        for idx, (variant_name, variant_series) in enumerate(variant_series_list):
            error = np.abs(variant_series[name] - base_series[name])
            ax.plot(
                time_axis,
                error,
                color=variant_color(idx, len(variant_series_list)),
                linewidth=VARIANT_LINEWIDTH,
                alpha=VARIANT_ALPHA,
                label=variant_name,
            )
        ax.set_title(name)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('joint_dof_error [rad]')

    finalize_grid(fig, axes, len(names))
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_metric_bars(output_path, metrics_list):
    metric_specs = [
        ('mean_body_position_error', 'mean_body_position_error [m]'),
        ('min_body_position_error', 'min_body_position_error [m]'),
        ('max_body_position_error', 'max_body_position_error [m]'),
        ('mean_joint_dof_error', 'mean_joint_dof_error [rad]'),
        ('min_joint_dof_error', 'min_joint_dof_error [rad]'),
        ('max_joint_dof_error', 'max_joint_dof_error [rad]'),
    ]
    variant_names = [item['variant_name'] for item in metrics_list]
    x = np.arange(len(variant_names))
    colors = [variant_color(idx, len(metrics_list)) for idx in range(len(metrics_list))]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)
    fig.suptitle('Variant body_position_error and joint_dof_error metrics')

    for ax, (key, title) in zip(axes.reshape(-1), metric_specs):
        values = [item[key] for item in metrics_list]
        ax.bar(x, values, color=colors)
        ax.set_title(title)
        ax.set_xticks(x, variant_names, rotation=35, ha='right')

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def write_csv(output_path, metrics_list):
    fieldnames = [
        'variant_name',
        'beta_l2_distance',
        'mean_body_position_error',
        'min_body_position_error',
        'max_body_position_error',
        'mean_joint_dof_error',
        'min_joint_dof_error',
        'max_joint_dof_error',
    ]
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in metrics_list:
            writer.writerow({key: metrics[key] for key in fieldnames})


def build_summary(folder, base_path, variant_paths, base_data, metrics_list):
    aggregate = {}
    keys = [
        'mean_body_position_error',
        'min_body_position_error',
        'max_body_position_error',
        'mean_joint_dof_error',
        'min_joint_dof_error',
        'max_joint_dof_error',
        'beta_l2_distance',
    ]
    for key in keys:
        values = np.array([item[key] for item in metrics_list], dtype=np.float64)
        aggregate[key] = {
            'mean': float(values.mean()),
            'min': float(values.min()),
            'max': float(values.max()),
        }

    return {
        'folder': str(folder),
        'base_variant_file': str(base_path.name),
        'num_variants': len(variant_paths),
        'num_frames': int(base_data['joints'].shape[0]),
        'num_smplx_joints': int(base_data['joints'].shape[1]),
        'selected_bodies': [name for name, _ in SELECTED_BODY_SPECS],
        'selected_joint_dofs': [name for name, _ in SELECTED_DOF_SPECS],
        'mocap_frame_rate': float(to_python(base_data['mocap_frame_rate'])),
        'aggregate_metrics': aggregate,
        'per_variant': metrics_list,
    }


def print_summary(metrics_list):
    print('Variant comparison summary:')
    print('variant                beta_l2   mean_body_position_error   max_body_position_error   mean_joint_dof_error   max_joint_dof_error')
    for item in metrics_list:
        print(
            f"{item['variant_name']:<20} "
            f"{item['beta_l2_distance']:>8.4f} "
            f"{item['mean_body_position_error']:>10.4f} "
            f"{item['max_body_position_error']:>9.4f} "
            f"{item['mean_joint_dof_error']:>9.6f} "
            f"{item['max_joint_dof_error']:>8.6f}"
        )


def main():
    args = parse_args()
    apply_plot_style()

    folder, base_path, variant_paths = resolve_variant_paths(args.folder)
    output_dir = folder / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    base_data = load_npz_dict(base_path)
    base_body_series = extract_body_series(base_data['joints'])
    base_all_dofs = flatten_all_dofs(base_data)
    base_selected_dofs = extract_selected_dof_series(base_data)
    fps = float(to_python(base_data['mocap_frame_rate']))
    time_axis = np.arange(base_data['joints'].shape[0], dtype=np.float32) / fps

    metrics_list = []
    body_variant_series = []
    dof_variant_series = []

    for variant_path in variant_paths:
        variant_data = load_npz_dict(variant_path)
        validate_variant(base_data, variant_data, variant_path)
        metrics, variant_body_series, variant_selected_dofs = compute_variant_metrics(
            base_data,
            variant_data,
            base_body_series,
            base_all_dofs,
            base_selected_dofs,
        )
        metrics['npz_file'] = variant_path.name
        metrics_list.append(metrics)
        body_variant_series.append((metrics['variant_name'], variant_body_series))
        dof_variant_series.append((metrics['variant_name'], variant_selected_dofs))

    plot_body_trajectories(
        output_dir / 'selected_body_trajectories_xz.png',
        'Selected Body Trajectories (X-Z)',
        base_body_series,
        body_variant_series,
    )
    plot_body_error_over_time(
        output_dir / 'selected_body_position_error_over_time.png',
        'Selected body_position_error vs Base Shape',
        time_axis,
        base_body_series,
        body_variant_series,
    )
    plot_dof_traces(
        output_dir / 'selected_joint_dof_traces.png',
        'Selected Joint DOF Traces',
        time_axis,
        base_selected_dofs,
        dof_variant_series,
    )
    plot_dof_error_over_time(
        output_dir / 'selected_joint_dof_error_over_time.png',
        'Selected joint_dof_error vs Base Shape',
        time_axis,
        base_selected_dofs,
        dof_variant_series,
    )
    plot_metric_bars(output_dir / 'variant_error_metrics.png', metrics_list)

    write_csv(output_dir / 'variant_metrics.csv', metrics_list)
    summary = build_summary(folder, base_path, variant_paths, base_data, metrics_list)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    print_summary(metrics_list)
    print(f'Wrote comparison outputs to {output_dir}')


if __name__ == '__main__':
    main()
