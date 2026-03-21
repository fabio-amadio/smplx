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

FRONT_CAMERA = {
    'azimuth_deg': 90.0,
    'elevation_deg': 92.0,
    'distance_mult': 3.0,
    'target_height': 0.38,
}
FIXED_BACKGROUND_RGB = [232, 236, 242]
FIXED_VIDEO_FPS = 120.0
FIXED_VIDEO_WIDTH = 720
FIXED_VIDEO_HEIGHT = 720
FIXED_OPENGL_PLATFORM = 'egl'

BETA_DIST_MIN = 0.9* np.array([
    -0.3327, -0.7457, -0.2196, -0.8875,
    -4.6569, -4.0947, -1.0658, -4.2020,
    -2.1913, -2.4934, -1.2069, -3.9970,
    -3.2033, -2.1479, -0.6005, -0.1839,
], dtype=np.float32)
BETA_DIST_MAX = 1.1 * np.array([
    1.7192, 1.0801, 2.1832, 2.2377,
    2.1501, 1.4160, 2.7794, 2.3992,
    1.8828, 2.3267, 3.0854, -0.1402,
    0.4211, 2.1681, 3.5587, 3.2515,
], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Replay one AMASS/ACCAD SMPL-X motion and save shape variants.'
    )
    parser.add_argument('--model-folder', required=True, type=str,
                        help='Folder that contains the smplx/ model files.')
    parser.add_argument('--motion-file', required=True, type=str,
                        help='Path to one AMASS/ACCAD .npz motion file.')
    parser.add_argument('--output-dir', required=True, type=str,
                        help='Directory where the generated variant files are saved.')
    parser.add_argument('--start-frame', default=0, type=int,
                        help='First frame to process.')
    parser.add_argument('--num-frames', default=None, type=int,
                        help='Number of frames to process. Default: all remaining frames.')
    parser.add_argument('--num-random-shapes', default=0, type=int,
                        help='How many random beta vectors to sample in addition to the motion shape.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed used for sampled body shapes.')
    parser.add_argument('--device', default='cpu', type=str,
                        help='Torch device, for example cpu or cuda.')
    parser.add_argument('--render-videos', action='store_true',
                        help='Render an MP4 next to each generated npz variant.')
    return parser.parse_args()


def load_motion(path):
    motion = np.load(path, allow_pickle=True)
    missing = [key for key in REQUIRED_KEYS if key not in motion.files]
    if missing:
        raise KeyError(f'Missing keys in {path}: {missing}')

    surface_model_type = str(motion['surface_model_type']).lower()
    if surface_model_type != 'smplx':
        raise ValueError(
            f'Expected an SMPL-X motion file, got surface_model_type={surface_model_type!r}'
        )
    return motion


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


def get_beta_sampling_bounds(num_betas):
    if num_betas > len(BETA_DIST_MIN):
        raise ValueError(
            f'num_betas={num_betas} exceeds the hard-coded beta bounds length '
            f'{len(BETA_DIST_MIN)}'
        )
    dist_min = BETA_DIST_MIN[:num_betas]
    dist_max = BETA_DIST_MAX[:num_betas]
    if not np.all(dist_min < dist_max):
        raise ValueError('Each beta min bound must be strictly smaller than its max bound.')
    return dist_min, dist_max


def sample_random_betas(num_betas, num_samples, seed):
    dist_min, dist_max = get_beta_sampling_bounds(num_betas)
    rng = np.random.default_rng(seed)
    return rng.uniform(dist_min, dist_max, size=(num_samples, num_betas)).astype(np.float32)


def replay_variant(model, motion, frame_slice, betas, device):
    pose_hand = motion['pose_hand'][frame_slice]
    pose_eye = motion['pose_eye'][frame_slice]
    num_frames = pose_hand.shape[0]

    frame_betas = np.repeat(betas[None, :], num_frames, axis=0)

    with torch.no_grad():
        output = model(
            betas=torch.tensor(frame_betas, dtype=torch.float32, device=device),
            global_orient=torch.tensor(motion['root_orient'][frame_slice], dtype=torch.float32, device=device),
            body_pose=torch.tensor(motion['pose_body'][frame_slice], dtype=torch.float32, device=device),
            left_hand_pose=torch.tensor(pose_hand[:, :45], dtype=torch.float32, device=device),
            right_hand_pose=torch.tensor(pose_hand[:, 45:], dtype=torch.float32, device=device),
            jaw_pose=torch.tensor(motion['pose_jaw'][frame_slice], dtype=torch.float32, device=device),
            leye_pose=torch.tensor(pose_eye[:, :3], dtype=torch.float32, device=device),
            reye_pose=torch.tensor(pose_eye[:, 3:], dtype=torch.float32, device=device),
            transl=torch.tensor(motion['trans'][frame_slice], dtype=torch.float32, device=device),
            expression=torch.zeros(
                (num_frames, model.num_expression_coeffs),
                dtype=torch.float32,
                device=device,
            ),
            return_verts=True,
        )

    return output.vertices.cpu().numpy(), output.joints.cpu().numpy()


def save_variant(output_path, motion_path, motion, frame_slice, faces, betas, vertices, joints, variant_name):
    np.savez_compressed(
        output_path,
        variant_name=variant_name,
        source_motion=str(motion_path),
        gender=str(motion['gender']),
        surface_model_type=str(motion['surface_model_type']),
        mocap_frame_rate=float(motion['mocap_frame_rate']),
        frame_start=frame_slice.start,
        frame_end=frame_slice.stop,
        num_frames=frame_slice.stop - frame_slice.start,
        betas=betas.astype(np.float32),
        trans=motion['trans'][frame_slice].astype(np.float32),
        root_orient=motion['root_orient'][frame_slice].astype(np.float32),
        pose_body=motion['pose_body'][frame_slice].astype(np.float32),
        pose_hand=motion['pose_hand'][frame_slice].astype(np.float32),
        pose_jaw=motion['pose_jaw'][frame_slice].astype(np.float32),
        pose_eye=motion['pose_eye'][frame_slice].astype(np.float32),
        vertices=vertices.astype(np.float32),
        joints=joints.astype(np.float32),
        faces=faces.astype(np.int32),
    )


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


def estimate_front_camera(vertices, joints):
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
    return make_look_at_pose(eye, target, world_up), target, scale


def render_video(video_path, vertices, joints, faces):
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg was not found on PATH, but --render-videos was requested.')

    os.environ.setdefault('PYOPENGL_PLATFORM', FIXED_OPENGL_PLATFORM)

    import pyrender
    import trimesh

    camera_pose, target, scale = estimate_front_camera(vertices, joints)
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
        '-r', f'{FIXED_VIDEO_FPS:.6f}',
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
            color = np.ascontiguousarray(np.rot90(color, k=1))
            process.stdin.write(color.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
        renderer.delete()

    if return_code != 0:
        raise RuntimeError(f'ffmpeg exited with status {return_code} while writing {video_path}')


def main():
    args = parse_args()

    motion_path = Path(args.motion_file).expanduser().resolve()
    model_folder = str(Path(args.model_folder).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    motion = load_motion(motion_path)
    total_frames = motion['trans'].shape[0]
    frame_slice = resolve_frame_slice(total_frames, args.start_frame, args.num_frames)

    motion_betas = motion['betas'].astype(np.float32)
    num_betas = int(motion['num_betas']) if 'num_betas' in motion.files else motion_betas.shape[0]
    num_betas = min(num_betas, motion_betas.shape[0])
    motion_betas = motion_betas[:num_betas]
    dist_min, dist_max = get_beta_sampling_bounds(num_betas)

    device = torch.device(args.device)
    model = build_model(model_folder, str(motion['gender']), num_betas, device)

    variants = [('motion_shape', motion_betas)]
    if args.num_random_shapes > 0:
        sampled = sample_random_betas(
            num_betas=num_betas,
            num_samples=args.num_random_shapes,
            seed=args.seed,
        )
        for idx, betas in enumerate(sampled):
            variants.append((f'random_shape_{idx:04d}', betas))

    manifest = {
        'motion_file': str(motion_path),
        'model_folder': model_folder,
        'gender': str(motion['gender']),
        'surface_model_type': str(motion['surface_model_type']),
        'mocap_frame_rate': float(motion['mocap_frame_rate']),
        'start_frame': frame_slice.start,
        'end_frame': frame_slice.stop,
        'num_frames': frame_slice.stop - frame_slice.start,
        'num_betas': num_betas,
        'num_variants': len(variants),
        'variants': [name for name, _ in variants],
        'random_beta_sampling': 'uniform_per_dimension',
        'random_beta_dist_min': dist_min.tolist(),
        'random_beta_dist_max': dist_max.tolist(),
        'render_videos': args.render_videos,
        'video_fps': FIXED_VIDEO_FPS if args.render_videos else None,
        'video_width': FIXED_VIDEO_WIDTH if args.render_videos else None,
        'video_height': FIXED_VIDEO_HEIGHT if args.render_videos else None,
        'opengl_platform': FIXED_OPENGL_PLATFORM if args.render_videos else None,
        'camera': FRONT_CAMERA if args.render_videos else None,
        'background_rgb': FIXED_BACKGROUND_RGB if args.render_videos else None,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + chr(10))

    print(f'Motion: {motion_path}')
    print(f'Frames: {frame_slice.start}:{frame_slice.stop} ({frame_slice.stop - frame_slice.start} frames)')
    print(f'Output dir: {output_dir}')
    print(f'Random beta sampling: uniform per dimension in [{dist_min.tolist()}, {dist_max.tolist()}]')
    if args.render_videos:
        print(
            f'Video rendering enabled: {FIXED_VIDEO_FPS} fps, '
            f'{FIXED_VIDEO_WIDTH}x{FIXED_VIDEO_HEIGHT}, '
            f'platform={FIXED_OPENGL_PLATFORM}, '
            f'camera={FRONT_CAMERA}, background={FIXED_BACKGROUND_RGB}'
        )

    for variant_name, betas in variants:
        vertices, joints = replay_variant(model, motion, frame_slice, betas, device)
        output_path = output_dir / f'{variant_name}.npz'
        save_variant(
            output_path=output_path,
            motion_path=motion_path,
            motion=motion,
            frame_slice=frame_slice,
            faces=model.faces,
            betas=betas,
            vertices=vertices,
            joints=joints,
            variant_name=variant_name,
        )
        print(
            f'Saved {variant_name}: vertices {vertices.shape}, joints {joints.shape} -> {output_path}'
        )

        if args.render_videos:
            video_path = output_dir / f'{variant_name}.mp4'
            render_video(
                video_path=video_path,
                vertices=vertices,
                joints=joints,
                faces=model.faces,
            )
            print(f'Saved {variant_name} video -> {video_path}')


if __name__ == '__main__':
    main()
