#!/usr/bin/env python3
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

import amass_to_smplx_variants as single_motion


SUBSET_DIR = Path('/home/famadio/Workspace/RL/mjlab_playground/clamp/assets/motions/g1_motions_npz/accad_subset')
LOCAL_ACCAD_DIR = Path('/home/famadio/Workspace/smplx/ACCAD')
MODEL_FOLDER = Path('/home/famadio/Workspace/smplx/models')
OUTPUT_ROOT = Path('/home/famadio/Workspace/smplx/output/accad_subset_matched_variants')
NUM_RANDOM_SHAPES = 15
BASE_SEED = 0
DEVICE = 'cpu'
RENDER_VIDEOS = False


def normalize_motion_stem(name):
    stem = Path(name).stem.lower()
    replacements = {
        'stageii': '',
        'stagei': '',
        'subject_': '',
        'subject': '',
        'subj_': '',
        'subj': '',
        'conversation_gestures': 'conversationgestures',
        'pick_up': 'pickup',
        'same_direction': 'samedirection',
    }
    for old, new in replacements.items():
        stem = stem.replace(old, new)
    stem = re.sub(r'[^a-z0-9]+', '', stem)
    return stem


def collect_matches():
    subset_paths = sorted(SUBSET_DIR.glob('*.npz'))
    accad_paths = sorted(LOCAL_ACCAD_DIR.rglob('*.npz'))

    accad_by_norm = defaultdict(list)
    for accad_path in accad_paths:
        accad_by_norm[normalize_motion_stem(accad_path.name)].append(accad_path)

    subset_to_candidates = {}
    unmatched_subset = []
    matched_local = set()
    reverse_matches = defaultdict(list)

    for subset_path in subset_paths:
        candidates = accad_by_norm.get(normalize_motion_stem(subset_path.name), [])
        subset_to_candidates[subset_path] = candidates
        if not candidates:
            unmatched_subset.append(subset_path)
            continue
        for candidate in candidates:
            matched_local.add(candidate)
            reverse_matches[candidate].append(subset_path.name)

    return subset_paths, subset_to_candidates, sorted(matched_local), unmatched_subset, reverse_matches


def get_motion_shape_context(motion):
    total_frames = motion['trans'].shape[0]
    frame_slice = single_motion.resolve_frame_slice(total_frames, 0, None)

    motion_betas = motion['betas'].astype('float32')
    num_betas = int(motion['num_betas']) if 'num_betas' in motion.files else motion_betas.shape[0]
    num_betas = min(num_betas, motion_betas.shape[0])
    motion_betas = motion_betas[:num_betas]
    return frame_slice, motion_betas, num_betas


def get_model(model_cache, gender, num_betas, device):
    key = (str(gender), int(num_betas), str(device))
    if key not in model_cache:
        model_cache[key] = single_motion.build_model(
            str(MODEL_FOLDER),
            str(gender),
            int(num_betas),
            device,
        )
    return model_cache[key]


def build_output_dir(motion_path):
    relative_motion = motion_path.relative_to(LOCAL_ACCAD_DIR)
    return OUTPUT_ROOT / relative_motion.parent / relative_motion.stem


def make_motion_seed(motion_path):
    relative_motion = motion_path.relative_to(LOCAL_ACCAD_DIR).as_posix()
    payload = f'{BASE_SEED}:{relative_motion}'.encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], 'little') % (2 ** 32)


def write_motion_manifest(output_dir, motion_path, motion, frame_slice, num_betas, variants, motion_seed):
    dist_min, dist_max = single_motion.get_beta_sampling_bounds(num_betas)
    manifest = {
        'motion_file': str(motion_path),
        'model_folder': str(MODEL_FOLDER),
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
        'random_beta_seed': int(motion_seed),
        'random_beta_seed_strategy': 'sha256(base_seed + relative_motion_path)',
        'random_beta_dist_min': dist_min.tolist(),
        'random_beta_dist_max': dist_max.tolist(),
        'render_videos': RENDER_VIDEOS,
        'video_fps': single_motion.FIXED_VIDEO_FPS if RENDER_VIDEOS else None,
        'video_width': single_motion.FIXED_VIDEO_WIDTH if RENDER_VIDEOS else None,
        'video_height': single_motion.FIXED_VIDEO_HEIGHT if RENDER_VIDEOS else None,
        'opengl_platform': single_motion.FIXED_OPENGL_PLATFORM if RENDER_VIDEOS else None,
        'camera': single_motion.FRONT_CAMERA if RENDER_VIDEOS else None,
        'background_rgb': single_motion.FIXED_BACKGROUND_RGB if RENDER_VIDEOS else None,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + chr(10))


def process_motion(motion_path, matched_subset_files, model_cache, device):
    motion = single_motion.load_motion(motion_path)
    frame_slice, motion_betas, num_betas = get_motion_shape_context(motion)
    model = get_model(model_cache, motion['gender'], num_betas, device)
    motion_seed = make_motion_seed(motion_path)

    variants = [('motion_shape', motion_betas)]
    sampled = single_motion.sample_random_betas(
        num_betas=num_betas,
        num_samples=NUM_RANDOM_SHAPES,
        seed=motion_seed,
    )
    for idx, betas in enumerate(sampled):
        variants.append((f'random_shape_{idx:04d}', betas))

    output_dir = build_output_dir(motion_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_motion_manifest(output_dir, motion_path, motion, frame_slice, num_betas, variants, motion_seed)

    for variant_name, betas in variants:
        vertices, joints = single_motion.replay_variant(model, motion, frame_slice, betas, device)
        output_path = output_dir / f'{variant_name}.npz'
        single_motion.save_variant(
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
        if RENDER_VIDEOS:
            single_motion.render_video(
                video_path=output_dir / f'{variant_name}.mp4',
                vertices=vertices,
                joints=joints,
                faces=model.faces,
            )

    return {
        'motion_file': str(motion_path),
        'matched_subset_files': list(matched_subset_files),
        'output_dir': str(output_dir),
        'num_frames': int(frame_slice.stop - frame_slice.start),
        'num_betas': int(num_betas),
        'num_variants': len(variants),
        'random_beta_seed': int(motion_seed),
        'render_videos': RENDER_VIDEOS,
    }


def write_batch_manifest(output_root, subset_paths, subset_to_candidates, matched_local_paths, unmatched_subset, processed, failed):
    manifest = {
        'subset_dir': str(SUBSET_DIR),
        'local_accad_dir': str(LOCAL_ACCAD_DIR),
        'model_folder': str(MODEL_FOLDER),
        'output_root': str(output_root),
        'num_subset_files': len(subset_paths),
        'num_unique_local_matches': len(matched_local_paths),
        'num_unmatched_subset_files': len(unmatched_subset),
        'num_processed_motions': len(processed),
        'num_failed_motions': len(failed),
        'num_random_shapes': NUM_RANDOM_SHAPES,
        'base_seed': BASE_SEED,
        'random_beta_seed_strategy': 'sha256(base_seed + relative_motion_path)',
        'device': DEVICE,
        'render_videos': RENDER_VIDEOS,
        'unmatched_subset_files': [path.name for path in unmatched_subset],
        'subset_match_counts': {
            subset_path.name: len(candidates)
            for subset_path, candidates in subset_to_candidates.items()
        },
        'processed_motions': processed,
        'failed_motions': failed,
    }
    (output_root / 'batch_manifest.json').write_text(json.dumps(manifest, indent=2) + chr(10))


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    subset_paths, subset_to_candidates, matched_local_paths, unmatched_subset, reverse_matches = collect_matches()
    if unmatched_subset:
        missing_names = ', '.join(path.name for path in unmatched_subset[:10])
        raise RuntimeError(
            f'Found {len(unmatched_subset)} unmatched subset files. First entries: {missing_names}'
        )

    print(f'Subset files: {len(subset_paths)}')
    print(f'Unique local ACCAD matches: {len(matched_local_paths)}')
    print(f'Output root: {OUTPUT_ROOT}')
    print(f'Random shape variations per motion: {NUM_RANDOM_SHAPES}')
    print(f'Render videos: {RENDER_VIDEOS}')

    processed = []
    failed = []
    model_cache = {}

    for index, motion_path in enumerate(matched_local_paths, start=1):
        relative_motion = motion_path.relative_to(LOCAL_ACCAD_DIR)
        print(f'[{index}/{len(matched_local_paths)}] Processing {relative_motion}')
        try:
            result = process_motion(
                motion_path=motion_path,
                matched_subset_files=reverse_matches.get(motion_path, []),
                model_cache=model_cache,
                device=device,
            )
            processed.append(result)
        except Exception as exc:
            failed.append({
                'motion_file': str(motion_path),
                'error': repr(exc),
            })
            print(f'  FAILED: {exc}')

    write_batch_manifest(
        output_root=OUTPUT_ROOT,
        subset_paths=subset_paths,
        subset_to_candidates=subset_to_candidates,
        matched_local_paths=matched_local_paths,
        unmatched_subset=unmatched_subset,
        processed=processed,
        failed=failed,
    )

    print(f'Processed motions: {len(processed)}')
    print(f'Failed motions: {len(failed)}')
    print(f'Batch manifest: {OUTPUT_ROOT / "batch_manifest.json"}')


if __name__ == '__main__':
    main()
