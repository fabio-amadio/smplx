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
BODY_QUAT_ORDER = 'wxyz'

BODY_NAMES_FALLBACK = tuple(JOINT_NAMES[:22])
BODY_PARENT_INDICES = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
)

SELECTED_BODY_SPECS = [
    ('pelvis_body', ('joint', 'pelvis')),
    ('torso_body', ('midpoint', 'spine3', 'neck')),
    ('head_body', ('midpoint', 'neck', 'head')),
    ('left_forearm_body', ('midpoint', 'left_elbow', 'left_wrist')),
    ('right_forearm_body', ('midpoint', 'right_elbow', 'right_wrist')),
    ('left_shank_body', ('midpoint', 'left_knee', 'left_ankle')),
    ('right_shank_body', ('midpoint', 'right_knee', 'right_ankle')),
]

SELECTED_ORIENTATION_BODY_NAMES = [
    'pelvis',
    'spine3',
    'head',
    'left_foot',
    'right_foot',
    'left_wrist',
    'right_wrist',
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


def to_name_tuple(values):
    if isinstance(values, np.ndarray):
        flat = values.reshape(-1).tolist()
    else:
        flat = list(values)
    return tuple(str(to_python(value)) for value in flat)


def normalize_last_dim(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.clip(norms, eps, None)


def axis_angle_to_quat(axis_angle):
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = 0.5 * angles
    scales = np.where(
        angles > 1e-8,
        np.sin(half_angles) / np.clip(angles, 1e-8, None),
        0.5 - (angles * angles) / 48.0,
    )
    quat = np.concatenate([np.cos(half_angles), axis_angle * scales], axis=-1)
    return normalize_last_dim(quat)


def quat_multiply(q1, q2):
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    product = np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)
    return normalize_last_dim(product)


def quat_geodesic_error(q1, q2):
    q1 = normalize_last_dim(q1)
    q2 = normalize_last_dim(q2)
    dot = np.sum(q1 * q2, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return (2.0 * np.arccos(dot)).astype(np.float32)


def build_body_quat_from_dof_pos(dof_pos):
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    expected_dims = 3 * len(BODY_NAMES_FALLBACK)
    if dof_pos.shape[1] != expected_dims:
        raise ValueError(
            f'Expected dof_pos with {expected_dims} channels, got {dof_pos.shape[1]}'
        )

    num_frames = dof_pos.shape[0]
    local_axis_angle = dof_pos.reshape(num_frames, len(BODY_NAMES_FALLBACK), 3)
    local_quat = axis_angle_to_quat(local_axis_angle)
    body_quat = np.empty_like(local_quat)
    for joint_idx, parent_idx in enumerate(BODY_PARENT_INDICES):
        if parent_idx < 0:
            body_quat[:, joint_idx, :] = local_quat[:, joint_idx, :]
        else:
            body_quat[:, joint_idx, :] = quat_multiply(
                body_quat[:, parent_idx, :],
                local_quat[:, joint_idx, :],
            )
    return normalize_last_dim(body_quat).astype(np.float32)


def standardize_variant_data(raw_data, variant_path):
    compact_key_aliases = {
        'body_names': ('body_link_names', 'body_names'),
        'body_pos': ('body_pos_w', 'body_pos'),
        'body_quat': ('body_quat_w', 'body_quat'),
        'fps': ('fps', 'hz'),
    }
    compact_values = {}
    for canonical_key, aliases in compact_key_aliases.items():
        for alias in aliases:
            if alias in raw_data:
                compact_values[canonical_key] = raw_data[alias]
                break
    if 'betas' in raw_data and len(compact_values) == len(compact_key_aliases):
        return {
            'variant_name': Path(variant_path).stem,
            'body_names': to_name_tuple(compact_values['body_names']),
            'body_pos': np.asarray(compact_values['body_pos'], dtype=np.float32),
            'body_quat': np.asarray(compact_values['body_quat'], dtype=np.float32),
            'betas': np.asarray(raw_data['betas'], dtype=np.float32),
            'hz': float(to_python(compact_values['fps'])),
            'body_quat_order': BODY_QUAT_ORDER,
        }

    body_names_key = 'body_link_names' if 'body_link_names' in raw_data else 'body_names' if 'body_names' in raw_data else None
    fps_key = 'fps' if 'fps' in raw_data else 'hz' if 'hz' in raw_data else None
    old_compact_keys = {'dof_names', 'body_pos', 'dof_pos', 'betas'}
    if body_names_key is not None and fps_key is not None and old_compact_keys.issubset(raw_data.keys()):
        dof_pos = np.asarray(raw_data['dof_pos'], dtype=np.float32)
        return {
            'variant_name': Path(variant_path).stem,
            'body_names': to_name_tuple(raw_data[body_names_key]),
            'body_pos': np.asarray(raw_data['body_pos'], dtype=np.float32),
            'body_quat': build_body_quat_from_dof_pos(dof_pos),
            'betas': np.asarray(raw_data['betas'], dtype=np.float32),
            'hz': float(to_python(raw_data[fps_key])),
            'body_quat_order': BODY_QUAT_ORDER,
        }

    legacy_keys = {'joints', 'root_orient', 'pose_body', 'betas'}
    if legacy_keys.issubset(raw_data.keys()):
        if 'mocap_frame_rate' in raw_data:
            hz = float(to_python(raw_data['mocap_frame_rate']))
        elif 'fps' in raw_data:
            hz = float(to_python(raw_data['fps']))
        else:
            hz = float(to_python(raw_data['hz']))
        body_pos = np.asarray(raw_data['joints'][:, :len(BODY_NAMES_FALLBACK), :], dtype=np.float32)
        dof_pos = np.concatenate(
            [np.asarray(raw_data['root_orient'], dtype=np.float32), np.asarray(raw_data['pose_body'], dtype=np.float32)],
            axis=1,
        ).astype(np.float32)
        return {
            'variant_name': Path(variant_path).stem,
            'body_names': BODY_NAMES_FALLBACK,
            'body_pos': body_pos,
            'body_quat': build_body_quat_from_dof_pos(dof_pos),
            'betas': np.asarray(raw_data['betas'], dtype=np.float32),
            'hz': hz,
            'body_quat_order': BODY_QUAT_ORDER,
        }

    raise KeyError(
        f'Unrecognized variant format in {variant_path}. Found keys: {sorted(raw_data.keys())}'
    )


def validate_variant(base_data, variant_data, variant_path):
    if variant_data['body_names'] != base_data['body_names']:
        raise ValueError(f'Body-name mismatch in {variant_path}')
    if variant_data['body_pos'].shape != base_data['body_pos'].shape:
        raise ValueError(
            f"Shape mismatch for 'body_pos' in {variant_path}: "
            f"{variant_data['body_pos'].shape} vs base {base_data['body_pos'].shape}"
        )
    if variant_data['body_quat'].shape != base_data['body_quat'].shape:
        raise ValueError(
            f"Shape mismatch for 'body_quat' in {variant_path}: "
            f"{variant_data['body_quat'].shape} vs base {base_data['body_quat'].shape}"
        )


def build_name_to_index(names):
    return {name: idx for idx, name in enumerate(names)}


def extract_body_series(data):
    name_to_index = build_name_to_index(data['body_names'])
    body_pos = data['body_pos']
    series = {}
    for name, spec in SELECTED_BODY_SPECS:
        kind = spec[0]
        if kind == 'joint':
            joint_name = spec[1]
            series[name] = body_pos[:, name_to_index[joint_name], :].astype(np.float32)
        elif kind == 'midpoint':
            joint_a = body_pos[:, name_to_index[spec[1]], :]
            joint_b = body_pos[:, name_to_index[spec[2]], :]
            series[name] = (0.5 * (joint_a + joint_b)).astype(np.float32)
        else:
            raise ValueError(f'Unknown body spec kind: {kind}')
    return series


def extract_orientation_series(data):
    name_to_index = build_name_to_index(data['body_names'])
    series = {}
    for name in SELECTED_ORIENTATION_BODY_NAMES:
        if name not in name_to_index:
            raise KeyError(f'Missing selected body orientation {name!r} in variant data')
        series[name] = data['body_quat'][:, name_to_index[name], :].astype(np.float32)
    return series


def compute_variant_metrics(base_data, variant_data, base_body_series, base_orientation_series):
    variant_body_series = extract_body_series(variant_data)
    variant_orientation_series = extract_orientation_series(variant_data)

    body_error = np.linalg.norm(variant_data['body_pos'] - base_data['body_pos'], axis=-1)
    orientation_error = quat_geodesic_error(variant_data['body_quat'], base_data['body_quat'])

    metrics = {
        'variant_name': variant_data['variant_name'],
        'beta_l2_distance': float(np.linalg.norm(variant_data['betas'] - base_data['betas'])),
        'mean_body_position_error': float(body_error.mean()),
        'min_body_position_error': float(body_error.min()),
        'max_body_position_error': float(body_error.max()),
        'mean_body_orientation_error': float(orientation_error.mean()),
        'min_body_orientation_error': float(orientation_error.min()),
        'max_body_orientation_error': float(orientation_error.max()),
        'selected_body_position_mean_errors': {},
        'selected_body_orientation_mean_errors': {},
    }

    for name, base_pos in base_body_series.items():
        err = np.linalg.norm(variant_body_series[name] - base_pos, axis=-1)
        metrics['selected_body_position_mean_errors'][name] = float(err.mean())

    for name, base_quat in base_orientation_series.items():
        err = quat_geodesic_error(variant_orientation_series[name], base_quat)
        metrics['selected_body_orientation_mean_errors'][name] = float(err.mean())

    return metrics, variant_body_series, variant_orientation_series


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
        ax.plot(base[:, 0], base[:, 2], color=BASE_COLOR, linewidth=BASE_LINEWIDTH)
        for idx, (_, variant_series) in enumerate(variant_series_list):
            arr = variant_series[name]
            ax.plot(
                arr[:, 0],
                arr[:, 2],
                color=variant_color(idx, len(variant_series_list)),
                linewidth=VARIANT_LINEWIDTH,
                alpha=VARIANT_ALPHA,
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
        for idx, (_, variant_series) in enumerate(variant_series_list):
            error = np.linalg.norm(variant_series[name] - base_series[name], axis=-1)
            ax.plot(
                time_axis,
                error,
                color=variant_color(idx, len(variant_series_list)),
                linewidth=VARIANT_LINEWIDTH,
                alpha=VARIANT_ALPHA,
            )
        ax.set_title(name)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('body_position_error [m]')

    finalize_grid(fig, axes, len(names))
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_orientation_error_over_time(output_path, title, time_axis, base_series, variant_series_list):
    names = list(base_series.keys())
    nrows, ncols = make_grid(len(names), max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.0 * nrows), squeeze=False)
    fig.suptitle(title)

    for ax, name in zip(axes.reshape(-1), names):
        for idx, (_, variant_series) in enumerate(variant_series_list):
            error = quat_geodesic_error(variant_series[name], base_series[name])
            ax.plot(
                time_axis,
                error,
                color=variant_color(idx, len(variant_series_list)),
                linewidth=VARIANT_LINEWIDTH,
                alpha=VARIANT_ALPHA,
            )
        ax.set_title(name)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('body_orientation_error [rad]')

    finalize_grid(fig, axes, len(names))
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_metric_bars(output_path, metrics_list):
    metric_specs = [
        ('mean_body_position_error', 'mean_body_position_error [m]'),
        ('min_body_position_error', 'min_body_position_error [m]'),
        ('max_body_position_error', 'max_body_position_error [m]'),
        ('mean_body_orientation_error', 'mean_body_orientation_error [rad]'),
        ('min_body_orientation_error', 'min_body_orientation_error [rad]'),
        ('max_body_orientation_error', 'max_body_orientation_error [rad]'),
    ]
    variant_names = [item['variant_name'] for item in metrics_list]
    x = np.arange(len(variant_names))
    colors = [variant_color(idx, len(metrics_list)) for idx in range(len(metrics_list))]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)
    fig.suptitle('Variant body_position_error and body_orientation_error metrics')

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
        'mean_body_orientation_error',
        'min_body_orientation_error',
        'max_body_orientation_error',
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
        'mean_body_orientation_error',
        'min_body_orientation_error',
        'max_body_orientation_error',
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
        'num_frames': int(base_data['body_pos'].shape[0]),
        'num_bodies': int(base_data['body_pos'].shape[1]),
        'body_names': list(base_data['body_names']),
        'body_quat_order': BODY_QUAT_ORDER,
        'selected_bodies': [name for name, _ in SELECTED_BODY_SPECS],
        'selected_body_orientations': list(SELECTED_ORIENTATION_BODY_NAMES),
        'hz': float(base_data['hz']),
        'aggregate_metrics': aggregate,
        'per_variant': metrics_list,
    }


def print_summary(metrics_list):
    print('Variant comparison summary:')
    print('variant                beta_l2   mean_body_position_error   max_body_position_error   mean_body_orientation_error   max_body_orientation_error')
    for item in metrics_list:
        print(
            f"{item['variant_name']:<20} "
            f"{item['beta_l2_distance']:>8.4f} "
            f"{item['mean_body_position_error']:>10.4f} "
            f"{item['max_body_position_error']:>9.4f} "
            f"{item['mean_body_orientation_error']:>10.6f} "
            f"{item['max_body_orientation_error']:>9.6f}"
        )


def main():
    args = parse_args()
    apply_plot_style()

    folder, base_path, variant_paths = resolve_variant_paths(args.folder)
    output_dir = folder / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    base_data = standardize_variant_data(load_npz_dict(base_path), base_path)
    base_body_series = extract_body_series(base_data)
    base_orientation_series = extract_orientation_series(base_data)
    fps = float(base_data['hz'])
    time_axis = np.arange(base_data['body_pos'].shape[0], dtype=np.float32) / fps

    metrics_list = []
    body_variant_series = []
    orientation_variant_series = []

    for variant_path in variant_paths:
        variant_data = standardize_variant_data(load_npz_dict(variant_path), variant_path)
        validate_variant(base_data, variant_data, variant_path)
        metrics, variant_body_series, variant_orientation_series = compute_variant_metrics(
            base_data,
            variant_data,
            base_body_series,
            base_orientation_series,
        )
        metrics['npz_file'] = variant_path.name
        metrics_list.append(metrics)
        body_variant_series.append((metrics['variant_name'], variant_body_series))
        orientation_variant_series.append((metrics['variant_name'], variant_orientation_series))

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
    plot_orientation_error_over_time(
        output_dir / 'selected_body_orientation_error_over_time.png',
        'Selected body_orientation_error vs Base Shape',
        time_axis,
        base_orientation_series,
        orientation_variant_series,
    )
    plot_metric_bars(output_dir / 'variant_error_metrics.png', metrics_list)

    write_csv(output_dir / 'variant_metrics.csv', metrics_list)
    summary = build_summary(folder, base_path, variant_paths, base_data, metrics_list)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    print_summary(metrics_list)
    print(f'Wrote comparison outputs to {output_dir}')


if __name__ == '__main__':
    main()
