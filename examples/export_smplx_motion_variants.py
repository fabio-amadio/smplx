#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

import smplx
from smplx.joint_names import JOINT_NAMES


AMASS_REQUIRED_KEYS = [
    'betas',
    'gender',
    'mocap_frame_rate',
    'pose_body',
    'pose_eye',
    'pose_hand',
    'pose_jaw',
    'root_orient',
    'surface_model_type',
    'trans',
]
REQUIRED_KEYS = [
    'betas',
    'gender',
    'mocap_frame_rate',
    'pose_body',
    'pose_eye',
    'pose_hand',
    'pose_jaw',
    'root_orient',
    'surface_model_type',
    'trans',
]
ROTATION_KEYS = ['root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']
LINEAR_KEYS = ['trans']
OMOMO_MOCAP_FRAME_RATE = 30.0
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ACCAD_BETA_ROOT = PROJECT_ROOT / 'ACCAD'
DEFAULT_OMOMO_BETA_BUNDLES = (
    PROJECT_ROOT.parent / 'OMOMO' / 'data' / 'train_diffusion_manip_seq_joints24.p',
    PROJECT_ROOT.parent / 'OMOMO' / 'data' / 'test_diffusion_manip_seq_joints24.p',
)

FRONT_CAMERA = {
    'azimuth_deg': 90.0,
    'elevation_deg': 92.0,
    'distance_mult': 3.0,
    'target_height': 0.38,
}
FIXED_BACKGROUND_RGB = [232, 236, 242]
FIXED_VIDEO_WIDTH = 720
FIXED_VIDEO_HEIGHT = 720
FIXED_OPENGL_PLATFORM = 'egl'

BODY_NAMES = tuple(JOINT_NAMES[:22])
BODY_PARENT_INDICES = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
)
BODY_QUAT_ORDER = 'wxyz'
OUTPUT_NPZ_FIELDS = ['body_link_names', 'body_pos_w', 'body_quat_w', 'betas', 'fps']
EMPIRICAL_BETA_POOL_CACHE = {}


def _as_python_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return value


def _normalize_motion_dict(motion_dict):
    normalized = {}
    for key, value in motion_dict.items():
        if key in ('gender', 'surface_model_type', 'seq_name', 'source_format'):
            normalized[key] = str(_as_python_scalar(value))
        elif key in ('motion_key', 'num_betas'):
            normalized[key] = int(_as_python_scalar(value))
        elif key == 'mocap_frame_rate':
            normalized[key] = float(_as_python_scalar(value))
        elif key == 'betas':
            normalized[key] = np.asarray(value, dtype=np.float32).reshape(-1)
        elif key in REQUIRED_KEYS:
            normalized[key] = np.asarray(value, dtype=np.float32)
        else:
            normalized[key] = value
    return normalized


def load_npz_motion(path):
    motion = np.load(path, allow_pickle=True)
    missing = [key for key in AMASS_REQUIRED_KEYS if key not in motion.files]
    if missing:
        raise KeyError(f'Missing keys in {path}: {missing}')

    surface_model_type = str(_as_python_scalar(motion['surface_model_type'])).lower()
    if surface_model_type != 'smplx':
        raise ValueError(
            f'Expected an SMPL-X motion file, got surface_model_type={surface_model_type!r}'
        )

    motion_dict = {
        key: motion[key]
        for key in AMASS_REQUIRED_KEYS
    }
    motion_dict['surface_model_type'] = surface_model_type
    motion_dict['source_format'] = 'smplx_npz'
    motion_dict['seq_name'] = path.stem
    if 'num_betas' in motion.files:
        motion_dict['num_betas'] = motion['num_betas']
    return _normalize_motion_dict(motion_dict)


def _import_joblib():
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            'OMOMO raw bundles require joblib. Install it in this environment with '
            '`pip install joblib` and retry.'
        ) from exc
    return joblib


def _resolve_omomo_entry(omomo_data, seq_name=None, motion_key=None):
    if seq_name is not None and motion_key is not None:
        raise ValueError('Use either seq_name or motion_key for OMOMO, not both.')

    if seq_name is not None:
        matches = [
            (key, seq)
            for key, seq in omomo_data.items()
            if str(seq.get('seq_name')) == str(seq_name)
        ]
        if not matches:
            raise KeyError(f'OMOMO sequence {seq_name!r} was not found in the bundle.')
        if len(matches) > 1:
            raise ValueError(
                f'OMOMO sequence name {seq_name!r} is ambiguous inside the bundle: '
                f'keys={[key for key, _ in matches]}'
            )
        return matches[0]

    if motion_key is not None:
        if motion_key not in omomo_data:
            raise KeyError(f'OMOMO motion key {motion_key!r} was not found in the bundle.')
        return motion_key, omomo_data[motion_key]

    raise ValueError(
        'OMOMO raw bundles contain many sequences. Provide seq_name or motion_key to choose one.'
    )


def load_omomo_motion(path, seq_name=None, motion_key=None):
    joblib = _import_joblib()
    omomo_data = joblib.load(path)
    if not isinstance(omomo_data, dict):
        raise TypeError(
            f'Expected OMOMO bundle {path} to contain a dict of sequences, got '
            f'{type(omomo_data)!r}'
        )

    resolved_key, seq = _resolve_omomo_entry(omomo_data, seq_name=seq_name, motion_key=motion_key)
    missing = [key for key in ['betas', 'gender', 'pose_body', 'root_orient', 'trans', 'seq_name'] if key not in seq]
    if missing:
        raise KeyError(f'Missing OMOMO keys in {path} for entry {resolved_key}: {missing}')

    num_frames = int(np.asarray(seq['pose_body']).shape[0])
    motion_dict = {
        'betas': np.asarray(seq['betas'], dtype=np.float32).reshape(-1),
        'gender': str(seq['gender']),
        'mocap_frame_rate': float(OMOMO_MOCAP_FRAME_RATE),
        'pose_body': np.asarray(seq['pose_body'], dtype=np.float32),
        'pose_eye': np.zeros((num_frames, 6), dtype=np.float32),
        'pose_hand': np.zeros((num_frames, 90), dtype=np.float32),
        'pose_jaw': np.zeros((num_frames, 3), dtype=np.float32),
        'root_orient': np.asarray(seq['root_orient'], dtype=np.float32),
        'surface_model_type': 'smplx',
        'trans': np.asarray(seq['trans'], dtype=np.float32),
        'source_format': 'omomo_raw_bundle',
        'seq_name': str(seq['seq_name']),
        'motion_key': int(resolved_key),
        'num_betas': int(np.asarray(seq['betas']).reshape(-1).shape[0]),
    }
    return _normalize_motion_dict(motion_dict)


def load_motion(path, seq_name=None, motion_key=None):
    suffix = path.suffix.lower()
    if suffix == '.npz':
        if seq_name is not None or motion_key is not None:
            raise ValueError('seq_name and motion_key are only valid for OMOMO raw bundles.')
        return load_npz_motion(path)
    if suffix == '.p':
        return load_omomo_motion(path, seq_name=seq_name, motion_key=motion_key)
    raise ValueError(
        f'Unsupported motion file type {suffix!r}. Expected a SMPL-X .npz or OMOMO raw .p bundle.'
    )


def resolve_frame_slice(total_frames, start_frame, num_frames):
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(
            f'start-frame must be in [0, {total_frames - 1}], got {start_frame}'
        )

    if num_frames is None:
        end_frame = total_frames
    else:
        if num_frames <= 0:
            raise ValueError(f'num-frames must be positive, got {num_frames}')
        end_frame = min(total_frames, start_frame + num_frames)

    return slice(start_frame, end_frame)


def build_model(model_folder, gender, num_betas, device):
    model = smplx.create(
        model_folder,
        model_type='smplx',
        gender=gender,
        ext='npz',
        use_pca=False,
        num_betas=num_betas,
        num_expression_coeffs=10,
    )
    model = model.to(device)
    model.eval()
    return model


def beta_key(betas, num_betas):
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas.shape[0] < num_betas:
        raise ValueError(
            f'Expected at least {num_betas} beta coefficients, got {betas.shape[0]}'
        )
    return tuple(np.round(betas[:num_betas], 6).tolist())


def collect_empirical_beta_pool(num_betas):
    cached = EMPIRICAL_BETA_POOL_CACHE.get(num_betas)
    if cached is not None:
        return cached

    unique_betas = {}
    accad_unique_keys = set()
    omomo_unique_keys = set()
    stats = {
        'sampling': 'empirical_combined_accad_omomo',
        'num_betas': int(num_betas),
        'accad_root': str(DEFAULT_ACCAD_BETA_ROOT),
        'omomo_bundle_files': [],
        'accad_motion_files_scanned': 0,
        'accad_unique_betas': 0,
        'omomo_sequences_scanned': 0,
        'omomo_unique_betas': 0,
        'combined_unique_betas': 0,
    }

    if DEFAULT_ACCAD_BETA_ROOT.exists():
        for motion_path in DEFAULT_ACCAD_BETA_ROOT.rglob('*.npz'):
            motion = np.load(motion_path, allow_pickle=True)
            if 'betas' not in motion.files:
                continue
            betas = np.asarray(motion['betas'], dtype=np.float32).reshape(-1)
            if betas.shape[0] < num_betas:
                continue
            stats['accad_motion_files_scanned'] += 1
            key = beta_key(betas, num_betas)
            accad_unique_keys.add(key)
            unique_betas.setdefault(key, betas[:num_betas].copy())

    existing_omomo_bundles = [path for path in DEFAULT_OMOMO_BETA_BUNDLES if path.exists()]
    stats['omomo_bundle_files'] = [str(path) for path in existing_omomo_bundles]
    if existing_omomo_bundles:
        joblib = _import_joblib()
        for bundle_path in existing_omomo_bundles:
            bundle = joblib.load(bundle_path)
            if not isinstance(bundle, dict):
                raise TypeError(
                    f'Expected OMOMO bundle {bundle_path} to contain a dict of sequences, '
                    f'got {type(bundle)!r}'
                )
            stats['omomo_sequences_scanned'] += len(bundle)
            for seq in bundle.values():
                betas = np.asarray(seq['betas'], dtype=np.float32).reshape(-1)
                if betas.shape[0] < num_betas:
                    continue
                key = beta_key(betas, num_betas)
                omomo_unique_keys.add(key)
                unique_betas.setdefault(key, betas[:num_betas].copy())

    stats['accad_unique_betas'] = len(accad_unique_keys)
    stats['omomo_unique_betas'] = len(omomo_unique_keys)
    stats['combined_unique_betas'] = len(unique_betas)

    if not unique_betas:
        raise RuntimeError(
            'Could not build an empirical beta pool from ACCAD and OMOMO. '
            'Check that the local datasets are available.'
        )

    beta_pool = np.stack(list(unique_betas.values()), axis=0).astype(np.float32)
    cached = {
        'beta_pool': beta_pool,
        'stats': stats,
    }
    EMPIRICAL_BETA_POOL_CACHE[num_betas] = cached
    return cached


def sample_random_betas(num_betas, num_samples, seed, exclude_betas=None):
    empirical = collect_empirical_beta_pool(num_betas)
    candidate_pool = empirical['beta_pool']

    if exclude_betas is not None:
        exclude_key = beta_key(exclude_betas, num_betas)
        filtered = np.array(
            [betas for betas in candidate_pool if beta_key(betas, num_betas) != exclude_key],
            dtype=np.float32,
        )
        if filtered.size > 0:
            candidate_pool = filtered

    if candidate_pool.shape[0] == 0:
        raise RuntimeError('The empirical beta pool is empty after excluding the source shape.')

    rng = np.random.default_rng(seed)
    replace = num_samples > candidate_pool.shape[0]
    indices = rng.choice(candidate_pool.shape[0], size=num_samples, replace=replace)
    return candidate_pool[indices].astype(np.float32), empirical['stats']


def compute_duration_seconds(num_frames, fps):
    if num_frames <= 1:
        return 0.0
    return float(num_frames - 1) / float(fps)


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


def quat_to_axis_angle(quat):
    quat = normalize_last_dim(quat)
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    w = np.clip(quat[..., :1], -1.0, 1.0)
    xyz = quat[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angles = 2.0 * np.arctan2(sin_half, w)
    scales = np.where(
        sin_half > 1e-8,
        angles / np.clip(sin_half, 1e-8, None),
        2.0 + (angles * angles) / 12.0,
    )
    return (xyz * scales).astype(np.float32)


def quat_slerp_batch(q0, q1, blend):
    q0 = normalize_last_dim(q0)
    q1 = normalize_last_dim(q1)
    blend = np.asarray(blend, dtype=np.float32).reshape(-1, 1)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    close = dot > 0.9995
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * blend
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / np.clip(sin_theta_0, 1e-8, None)
    s1 = sin_theta / np.clip(sin_theta_0, 1e-8, None)
    slerped = s0 * q0 + s1 * q1

    lerped = normalize_last_dim((1.0 - blend) * q0 + blend * q1)
    return normalize_last_dim(np.where(close, lerped, slerped)).astype(np.float32)


def build_time_axis(num_frames, fps):
    if num_frames <= 1:
        return np.zeros((num_frames,), dtype=np.float32)
    duration = compute_duration_seconds(num_frames, fps)
    return np.linspace(0.0, duration, num_frames, dtype=np.float32)


def build_interpolation_lookup(source_times, target_times):
    if source_times.shape[0] == 1:
        zeros = np.zeros(target_times.shape[0], dtype=np.int64)
        return zeros, zeros, np.zeros(target_times.shape[0], dtype=np.float32)

    idx1 = np.searchsorted(source_times, target_times, side='right')
    idx1 = np.clip(idx1, 1, source_times.shape[0] - 1)
    idx0 = idx1 - 1

    denom = source_times[idx1] - source_times[idx0]
    blend = (target_times - source_times[idx0]) / np.clip(denom, 1e-8, None)
    return idx0.astype(np.int64), idx1.astype(np.int64), blend.astype(np.float32)


def resample_linear_sequence(values, idx0, idx1, blend):
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(values.shape[0], -1)
    resampled = (1.0 - blend)[:, None] * flat[idx0] + blend[:, None] * flat[idx1]
    return resampled.reshape((blend.shape[0],) + values.shape[1:]).astype(np.float32)


def resample_axis_angle_sequence(values, idx0, idx1, blend):
    values = np.asarray(values, dtype=np.float32)
    num_channels = values.shape[-1]
    if num_channels % 3 != 0:
        raise ValueError(f'Expected axis-angle channels to be divisible by 3, got {num_channels}')

    axis_angle = values.reshape(values.shape[0], num_channels // 3, 3)
    quats = axis_angle_to_quat(axis_angle)
    q0 = quats[idx0]
    q1 = quats[idx1]
    joint_blend = np.broadcast_to(blend[:, None], (blend.shape[0], quats.shape[1]))
    resampled_quats = quat_slerp_batch(
        q0.reshape(-1, 4),
        q1.reshape(-1, 4),
        joint_blend.reshape(-1),
    ).reshape(blend.shape[0], quats.shape[1], 4)
    resampled = quat_to_axis_angle(resampled_quats)
    return resampled.reshape(blend.shape[0], num_channels).astype(np.float32)


def prepare_motion_clip(motion, frame_slice, output_fps=None):
    source_fps = float(motion['mocap_frame_rate'])
    effective_output_fps = source_fps if output_fps is None else float(output_fps)
    if effective_output_fps <= 0.0:
        raise ValueError(f'output-fps must be positive, got {effective_output_fps}')

    frame_data = {
        key: np.asarray(motion[key][frame_slice], dtype=np.float32)
        for key in LINEAR_KEYS + ROTATION_KEYS
    }
    source_num_frames = int(frame_data['trans'].shape[0])
    duration_seconds = compute_duration_seconds(source_num_frames, source_fps)
    was_resampled = False

    if source_num_frames > 1 and not np.isclose(effective_output_fps, source_fps):
        source_times = build_time_axis(source_num_frames, source_fps)
        target_num_frames = max(2, int(round(duration_seconds * effective_output_fps)) + 1)
        target_times = np.linspace(0.0, duration_seconds, target_num_frames, dtype=np.float32)
        idx0, idx1, blend = build_interpolation_lookup(source_times, target_times)

        resampled = {}
        for key in LINEAR_KEYS:
            resampled[key] = resample_linear_sequence(frame_data[key], idx0, idx1, blend)
        for key in ROTATION_KEYS:
            resampled[key] = resample_axis_angle_sequence(frame_data[key], idx0, idx1, blend)
        frame_data = resampled
        was_resampled = True

    return {
        'gender': str(motion['gender']),
        'surface_model_type': str(motion['surface_model_type']),
        'source_mocap_frame_rate': source_fps,
        'mocap_frame_rate': effective_output_fps,
        'frame_start': int(frame_slice.start),
        'frame_end': int(frame_slice.stop),
        'source_num_frames': source_num_frames,
        'num_frames': int(frame_data['trans'].shape[0]),
        'duration_seconds': duration_seconds,
        'was_resampled': bool(was_resampled),
        **frame_data,
    }


def extract_body_pos(joints):
    joints = np.asarray(joints, dtype=np.float32)
    if joints.shape[1] < len(BODY_NAMES):
        raise ValueError(
            f'Expected at least {len(BODY_NAMES)} joints, got {joints.shape[1]}'
        )
    return joints[:, :len(BODY_NAMES), :].astype(np.float32)


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
    return normalize_last_dim(product).astype(np.float32)


def build_body_quat(motion_clip):
    root_orient = np.asarray(motion_clip['root_orient'], dtype=np.float32)
    pose_body = np.asarray(motion_clip['pose_body'], dtype=np.float32)
    num_frames = root_orient.shape[0]
    expected_pose_dims = 3 * (len(BODY_NAMES) - 1)
    if pose_body.shape[1] != expected_pose_dims:
        raise ValueError(
            f'Expected pose_body to have {expected_pose_dims} channels, got {pose_body.shape[1]}'
        )

    local_axis_angle = np.zeros((num_frames, len(BODY_NAMES), 3), dtype=np.float32)
    local_axis_angle[:, 0, :] = root_orient
    local_axis_angle[:, 1:, :] = pose_body.reshape(num_frames, len(BODY_NAMES) - 1, 3)
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


def save_variant(output_path, body_pos, body_quat, betas, fps):
    np.savez_compressed(
        output_path,
        body_link_names=np.asarray(BODY_NAMES),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        betas=betas.astype(np.float32),
        fps=np.float32(fps),
    )


def replay_variant(model, motion_clip, betas, device):
    pose_hand = motion_clip['pose_hand']
    pose_eye = motion_clip['pose_eye']
    num_frames = motion_clip['num_frames']

    frame_betas = np.repeat(betas[None, :], num_frames, axis=0)

    with torch.no_grad():
        output = model(
            betas=torch.tensor(frame_betas, dtype=torch.float32, device=device),
            global_orient=torch.tensor(motion_clip['root_orient'], dtype=torch.float32, device=device),
            body_pose=torch.tensor(motion_clip['pose_body'], dtype=torch.float32, device=device),
            left_hand_pose=torch.tensor(pose_hand[:, :45], dtype=torch.float32, device=device),
            right_hand_pose=torch.tensor(pose_hand[:, 45:], dtype=torch.float32, device=device),
            jaw_pose=torch.tensor(motion_clip['pose_jaw'], dtype=torch.float32, device=device),
            leye_pose=torch.tensor(pose_eye[:, :3], dtype=torch.float32, device=device),
            reye_pose=torch.tensor(pose_eye[:, 3:], dtype=torch.float32, device=device),
            transl=torch.tensor(motion_clip['trans'], dtype=torch.float32, device=device),
            expression=torch.zeros(
                (num_frames, model.num_expression_coeffs),
                dtype=torch.float32,
                device=device,
            ),
            return_verts=True,
        )

    return output.vertices.cpu().numpy(), output.joints.cpu().numpy()


def normalize(vec, eps=1e-8):
    norm = np.linalg.norm(vec)
    if norm < eps:
        return None
    return vec / norm


def make_look_at_pose(eye, target, up):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    z_axis = eye - target
    z_axis = normalize(z_axis)
    if z_axis is None:
        raise ValueError('Camera eye and target are too close to define a view direction.')

    x_axis = np.cross(up, z_axis)
    x_axis = normalize(x_axis)
    if x_axis is None:
        raise ValueError('Camera up vector is parallel to the view direction.')

    y_axis = np.cross(z_axis, x_axis)
    y_axis = normalize(y_axis)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def estimate_world_camera_basis(joints):
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    frame = joints[0]

    left_right = 0.5 * (
        (frame[17] - frame[16]) +
        (frame[2] - frame[1])
    )
    body_right = left_right - np.dot(left_right, world_up) * world_up
    body_right = normalize(body_right)
    if body_right is None:
        body_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    face_vec = frame[55] - frame[15]
    face_proj = face_vec - np.dot(face_vec, world_up) * world_up
    body_front = normalize(face_proj)
    if body_front is None:
        body_front = np.cross(world_up, body_right)
        body_front = normalize(body_front)

    if body_front is None:
        body_front = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    body_front = body_front - np.dot(body_front, body_right) * body_right
    body_front = normalize(body_front)
    if body_front is None:
        body_front = np.cross(world_up, body_right)
        body_front = normalize(body_front)

    if body_front is None:
        body_front = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return body_front, world_up


def build_render_setup(vertices, joints):
    bounds_min = vertices.min(axis=(0, 1))
    bounds_max = vertices.max(axis=(0, 1))
    center = 0.5 * (bounds_min + bounds_max)
    extents = np.maximum(bounds_max - bounds_min, 1e-3)
    scale = float(np.max(extents))

    target = np.array([
        center[0],
        bounds_min[1] + FRONT_CAMERA['target_height'] * extents[1],
        center[2],
    ], dtype=np.float32)

    body_front, world_up = estimate_world_camera_basis(joints)
    azimuth = np.deg2rad(FRONT_CAMERA['azimuth_deg'])
    elevation = np.deg2rad(FRONT_CAMERA['elevation_deg'])
    distance = max(FRONT_CAMERA['distance_mult'] * scale, 1e-3)

    fallback_right = np.cross(body_front, world_up)
    fallback_right = normalize(fallback_right)
    if fallback_right is None:
        fallback_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    horizontal_dir = np.cos(azimuth) * body_front + np.sin(azimuth) * fallback_right
    horizontal_dir = normalize(horizontal_dir)
    if horizontal_dir is None:
        horizontal_dir = body_front

    view_dir = np.cos(elevation) * horizontal_dir + np.sin(elevation) * world_up
    view_dir = normalize(view_dir)
    if view_dir is None:
        view_dir = horizontal_dir

    eye = target + view_dir * distance
    return {
        'camera_pose': make_look_at_pose(eye, target, world_up),
        'target': target,
        'scale': scale,
    }


def render_video(video_path, vertices, joints, faces, video_fps, render_setup=None):
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg was not found on PATH, but --render-videos was requested.')

    os.environ.setdefault('PYOPENGL_PLATFORM', FIXED_OPENGL_PLATFORM)

    import pyrender
    import trimesh

    if render_setup is None:
        render_setup = build_render_setup(vertices, joints)

    camera_pose = render_setup['camera_pose']
    target = render_setup['target']
    scale = render_setup['scale']

    scene = pyrender.Scene(
        bg_color=[FIXED_BACKGROUND_RGB[0], FIXED_BACKGROUND_RGB[1], FIXED_BACKGROUND_RGB[2], 255],
        ambient_light=[0.35, 0.35, 0.35],
    )

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.2)
    scene.add(camera, pose=camera_pose)

    light_offsets = [
        np.array([0.0, 0.4 * scale, 2.4 * scale], dtype=np.float32),
        np.array([1.2 * scale, 0.8 * scale, 1.6 * scale], dtype=np.float32),
        np.array([-1.2 * scale, 0.8 * scale, 1.6 * scale], dtype=np.float32),
    ]
    for offset in light_offsets:
        light_pose = make_look_at_pose(target + offset, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
        scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=FIXED_VIDEO_HEIGHT,
        viewport_height=FIXED_VIDEO_WIDTH,
    )
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{FIXED_VIDEO_WIDTH}x{FIXED_VIDEO_HEIGHT}',
        '-r', f'{float(video_fps):.6f}',
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        str(video_path),
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    mesh_node = None

    try:
        for frame_vertices in vertices:
            if mesh_node is not None:
                scene.remove_node(mesh_node)

            vertex_colors = np.ones((frame_vertices.shape[0], 4), dtype=np.float32)
            vertex_colors[:] = [0.72, 0.72, 0.76, 1.0]
            tri_mesh = trimesh.Trimesh(
                frame_vertices,
                faces,
                vertex_colors=vertex_colors,
                process=False,
            )
            mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
            mesh_node = scene.add(mesh)

            color, _ = renderer.render(scene)
            color = np.ascontiguousarray(np.rot90(color, k=-1))
            process.stdin.write(color.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
        renderer.delete()

    if return_code != 0:
        raise RuntimeError(f'ffmpeg exited with status {return_code} while writing {video_path}')

DEFAULT_DEVICE = 'cpu'


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Replay one SMPL-X motion and save shape variants.'
    )
    parser.add_argument('--model-folder', required=True, type=str,
                        help='Folder that contains the smplx/ model files.')
    parser.add_argument('--motion-file', required=True, type=str,
                        help='Path to one SMPL-X .npz motion file or an OMOMO raw .p bundle.')
    parser.add_argument('--seq-name', default=None, type=str,
                        help='For OMOMO raw bundles: sequence name to extract, for example sub10_clothesstand_000.')
    parser.add_argument('--motion-key', default=None, type=int,
                        help='For OMOMO raw bundles: integer dict key to extract.')
    parser.add_argument('--output-dir', required=True, type=str,
                        help='Directory where the generated variant files are saved.')
    parser.add_argument('--start-frame', default=0, type=int,
                        help='First frame to process.')
    parser.add_argument('--num-frames', default=None, type=int,
                        help='Number of frames to process. Default: all remaining frames.')
    parser.add_argument('--output-fps', default=None, type=float,
                        help='Resample the sliced motion to this output rate in Hz. Default: keep the source rate.')
    parser.add_argument('--num-random-shapes', default=0, type=int,
                        help='How many random beta vectors to sample in addition to the motion shape.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed used for sampled body shapes.')
    parser.add_argument('--device', default=DEFAULT_DEVICE, type=str,
                        help='Torch device, for example cpu or cuda.')
    parser.add_argument('--render-videos', action='store_true',
                        help='Render an MP4 next to each generated npz variant.')
    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)


def resolve_motion_betas(motion):
    motion_betas = motion['betas'].astype(np.float32)
    num_betas = int(motion['num_betas']) if 'num_betas' in motion else motion_betas.shape[0]
    num_betas = min(num_betas, motion_betas.shape[0])
    return motion_betas[:num_betas], num_betas


def build_variant_specs(motion_betas, num_betas, num_random_shapes, seed):
    variants = [('motion_shape', motion_betas)]
    beta_sampling_info = None
    if num_random_shapes > 0:
        sampled, beta_sampling_info = sample_random_betas(
            num_betas=num_betas,
            num_samples=num_random_shapes,
            seed=seed,
            exclude_betas=motion_betas,
        )
        for idx, betas in enumerate(sampled):
            variants.append((f'random_shape_{idx:04d}', betas))
    return variants, beta_sampling_info


def build_manifest(
    motion_path,
    model_folder,
    motion,
    motion_clip,
    num_betas,
    variants,
    beta_sampling_info,
    render_videos,
):
    video_fps = float(motion_clip['mocap_frame_rate']) if render_videos else None
    return {
        'motion_file': str(motion_path),
        'motion_source_format': motion.get('source_format', 'smplx_npz'),
        'motion_seq_name': motion.get('seq_name'),
        'motion_key': motion.get('motion_key'),
        'model_folder': str(model_folder),
        'gender': motion_clip['gender'],
        'surface_model_type': motion_clip['surface_model_type'],
        'source_mocap_frame_rate': float(motion_clip['source_mocap_frame_rate']),
        'mocap_frame_rate': float(motion_clip['mocap_frame_rate']),
        'start_frame': int(motion_clip['frame_start']),
        'end_frame': int(motion_clip['frame_end']),
        'source_num_frames': int(motion_clip['source_num_frames']),
        'num_frames': int(motion_clip['num_frames']),
        'duration_seconds': float(motion_clip['duration_seconds']),
        'was_resampled': bool(motion_clip['was_resampled']),
        'num_betas': int(num_betas),
        'num_bodies': len(BODY_NAMES),
        'body_names': list(BODY_NAMES),
        'body_quat_order': BODY_QUAT_ORDER,
        'output_npz_fields': list(OUTPUT_NPZ_FIELDS),
        'num_variants': len(variants),
        'variants': [name for name, _ in variants],
        'random_beta_sampling': None if beta_sampling_info is None else beta_sampling_info['sampling'],
        'random_beta_pool': beta_sampling_info,
        'render_videos': bool(render_videos),
        'video_fps': video_fps,
        'video_width': FIXED_VIDEO_WIDTH if render_videos else None,
        'video_height': FIXED_VIDEO_HEIGHT if render_videos else None,
        'opengl_platform': FIXED_OPENGL_PLATFORM if render_videos else None,
        'camera': FRONT_CAMERA if render_videos else None,
        'camera_reference_variant': 'motion_shape' if render_videos else None,
        'camera_reused_across_variants': bool(render_videos),
        'background_rgb': FIXED_BACKGROUND_RGB if render_videos else None,
    }


def write_manifest(output_dir, manifest):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + chr(10))
    return manifest_path


def generate_variants(
    model_folder,
    motion_file,
    output_dir,
    seq_name=None,
    motion_key=None,
    start_frame=0,
    num_frames=None,
    output_fps=None,
    num_random_shapes=0,
    seed=0,
    device=DEFAULT_DEVICE,
    render_videos=False,
    log_fn=print,
):
    motion_path = Path(motion_file).expanduser().resolve()
    model_folder = Path(model_folder).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    motion = load_motion(motion_path, seq_name=seq_name, motion_key=motion_key)
    total_frames = motion['trans'].shape[0]
    frame_slice = resolve_frame_slice(total_frames, start_frame, num_frames)
    motion_clip = prepare_motion_clip(motion, frame_slice, output_fps=output_fps)

    motion_betas, num_betas = resolve_motion_betas(motion)
    variants, beta_sampling_info = build_variant_specs(
        motion_betas,
        num_betas,
        num_random_shapes,
        seed,
    )
    manifest = build_manifest(
        motion_path=motion_path,
        model_folder=model_folder,
        motion=motion,
        motion_clip=motion_clip,
        num_betas=num_betas,
        variants=variants,
        beta_sampling_info=beta_sampling_info,
        render_videos=render_videos,
    )
    manifest_path = write_manifest(output_dir, manifest)

    torch_device = torch.device(device)
    model = build_model(str(model_folder), str(motion['gender']), num_betas, torch_device)

    if log_fn is not None:
        label = motion.get('seq_name') if motion.get('source_format') == 'omomo_raw_bundle' else motion_path
        log_fn(f'Motion: {label}')
        if motion.get('source_format') == 'omomo_raw_bundle':
            log_fn(f'OMOMO bundle: {motion_path} (key={motion.get("motion_key")})')
        log_fn(
            f'Source frames: {motion_clip["source_num_frames"]} @ {motion_clip["source_mocap_frame_rate"]:.6f} Hz '
            f'-> output frames: {motion_clip["num_frames"]} @ {motion_clip["mocap_frame_rate"]:.6f} Hz'
        )
        log_fn(f'Duration: {motion_clip["duration_seconds"]:.6f} s')
        log_fn(f'Original frame window: {motion_clip["frame_start"]}:{motion_clip["frame_end"]}')
        log_fn(f'Output dir: {output_dir}')
        if beta_sampling_info is None:
            log_fn('Random beta sampling: disabled (num_random_shapes=0)')
        else:
            log_fn(
                'Random beta sampling: empirical combined ACCAD+OMOMO pool '
                f'with {beta_sampling_info["combined_unique_betas"]} unique shapes '
                f'(ACCAD={beta_sampling_info["accad_unique_betas"]}, '
                f'OMOMO={beta_sampling_info["omomo_unique_betas"]})'
            )
        if render_videos:
            log_fn(
                f'Video rendering enabled: {motion_clip["mocap_frame_rate"]:.6f} fps, '
                f'{FIXED_VIDEO_WIDTH}x{FIXED_VIDEO_HEIGHT}, '
                f'platform={FIXED_OPENGL_PLATFORM}, '
                f'camera={FRONT_CAMERA}, background={FIXED_BACKGROUND_RGB}'
            )

    body_quat = build_body_quat(motion_clip)
    render_setup = None
    saved_variants = []
    for variant_name, betas in variants:
        vertices, joints = replay_variant(model, motion_clip, betas, torch_device)
        body_pos = extract_body_pos(joints)
        output_path = output_dir / f'{variant_name}.npz'
        save_variant(
            output_path=output_path,
            body_pos=body_pos,
            body_quat=body_quat,
            betas=betas,
            fps=motion_clip['mocap_frame_rate'],
        )
        saved_variants.append({
            'variant_name': variant_name,
            'npz_file': output_path.name,
            'betas': betas.tolist(),
        })
        if log_fn is not None:
            log_fn(
                f'Saved {variant_name}: body_pos {body_pos.shape}, body_quat {body_quat.shape} -> {output_path}'
            )

        if render_videos:
            if render_setup is None:
                render_setup = build_render_setup(vertices, joints)
            video_path = output_dir / f'{variant_name}.mp4'
            render_video(
                video_path=video_path,
                vertices=vertices,
                joints=joints,
                faces=model.faces,
                video_fps=float(motion_clip['mocap_frame_rate']),
                render_setup=render_setup,
            )
            saved_variants[-1]['video_file'] = video_path.name
            if log_fn is not None:
                log_fn(f'Saved {variant_name} video -> {video_path}')

    return {
        'motion_file': str(motion_path),
        'motion_source_format': motion.get('source_format', 'smplx_npz'),
        'motion_seq_name': motion.get('seq_name'),
        'motion_key': motion.get('motion_key'),
        'output_dir': str(output_dir),
        'manifest_path': str(manifest_path),
        'source_num_frames': int(motion_clip['source_num_frames']),
        'num_frames': int(motion_clip['num_frames']),
        'source_mocap_frame_rate': float(motion_clip['source_mocap_frame_rate']),
        'mocap_frame_rate': float(motion_clip['mocap_frame_rate']),
        'duration_seconds': float(motion_clip['duration_seconds']),
        'was_resampled': bool(motion_clip['was_resampled']),
        'num_betas': int(num_betas),
        'num_bodies': len(BODY_NAMES),
        'render_videos': bool(render_videos),
        'variants': saved_variants,
    }


def generate_variants_from_args(args, log_fn=print):
    return generate_variants(
        model_folder=args.model_folder,
        motion_file=args.motion_file,
        output_dir=args.output_dir,
        seq_name=args.seq_name,
        motion_key=args.motion_key,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        output_fps=args.output_fps,
        num_random_shapes=args.num_random_shapes,
        seed=args.seed,
        device=args.device,
        render_videos=args.render_videos,
        log_fn=log_fn,
    )


def main():
    generate_variants_from_args(parse_args())


if __name__ == "__main__":
    main()
