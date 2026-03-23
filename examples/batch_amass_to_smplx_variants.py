#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import yaml

import amass_to_smplx_variants as variants_lib


BATCH_OPTION_KEYS = (
    'start_frame',
    'num_frames',
    'output_fps',
    'num_random_shapes',
    'seed',
    'device',
    'render_videos',
)
BATCH_DEFAULTS = {
    'start_frame': 0,
    'num_frames': None,
    'output_fps': None,
    'num_random_shapes': 0,
    'seed': 0,
    'device': variants_lib.DEFAULT_DEVICE,
    'render_videos': False,
}
REQUIRED_CONFIG_KEYS = ('model_folder', 'output_root', 'motions')
MOTION_ENTRY_KEYS = set(BATCH_OPTION_KEYS) | {'path', 'motion_file', 'output_subdir', 'output_dir'}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Generate SMPL-X shape variants for a batch of motions listed in a YAML file.'
    )
    parser.add_argument('config', type=str, help='Path to the batch YAML config file.')
    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)


def load_yaml_config(config_path):
    with Path(config_path).open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return data



def resolve_path(raw_path, base_dir):
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def derive_motion_seed(base_seed, motion_path):
    payload = f'{int(base_seed)}:{motion_path}'.encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], 'little') % (2 ** 32)


def normalize_motion_entry(entry):
    if isinstance(entry, str):
        return {'path': entry}
    if not isinstance(entry, dict):
        raise TypeError(f'Each motion entry must be a string or a mapping, got {type(entry)!r}.')
    unknown_keys = set(entry) - MOTION_ENTRY_KEYS
    if unknown_keys:
        raise KeyError(f'Unsupported motion-entry keys: {sorted(unknown_keys)}')
    if 'path' not in entry and 'motion_file' not in entry:
        raise KeyError('Each motion entry must define path or motion_file.')
    return dict(entry)


def default_output_subdir(raw_motion_value, motion_path, motion_root):
    if motion_root is not None:
        try:
            return motion_path.relative_to(motion_root).with_suffix('')
        except ValueError:
            pass

    raw_path = Path(str(raw_motion_value))
    if not raw_path.is_absolute() and '..' not in raw_path.parts:
        return raw_path.with_suffix('')

    parent_name = motion_path.parent.name or 'motion'
    return Path(parent_name) / motion_path.stem


def resolve_output_dir(entry, output_root, motion_path, motion_root):
    if entry.get('output_dir') is not None:
        return resolve_path(entry['output_dir'], Path.cwd())
    if entry.get('output_subdir') is not None:
        return (output_root / Path(str(entry['output_subdir']))).resolve()
    raw_motion_value = entry.get('path', entry.get('motion_file'))
    return (output_root / default_output_subdir(raw_motion_value, motion_path, motion_root)).resolve()


def merge_motion_options(config, entry, motion_path):
    merged = {key: config.get(key, default) for key, default in BATCH_DEFAULTS.items()}
    for key in BATCH_OPTION_KEYS:
        if key in entry and entry[key] is not None:
            merged[key] = entry[key]

    if entry.get('seed') is None:
        merged['seed'] = derive_motion_seed(merged['seed'], motion_path)

    return merged


def validate_config(config):
    if not isinstance(config, dict):
        raise TypeError(f'Batch config must be a mapping, got {type(config)!r}.')
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(f'Missing required config keys: {missing}')
    if not isinstance(config['motions'], list) or not config['motions']:
        raise ValueError('Config key motions must be a non-empty list.')


def build_batch_manifest(config_path, config, processed, failed):
    return {
        'config_file': str(config_path),
        'model_folder': str(config['model_folder']),
        'output_root': str(config['output_root']),
        'motion_root': str(config['motion_root']) if config.get('motion_root') is not None else None,
        'num_requested_motions': len(config['motions']),
        'num_processed_motions': len(processed),
        'num_failed_motions': len(failed),
        'default_batch_options': {
            key: config.get(key, BATCH_DEFAULTS[key])
            for key in BATCH_OPTION_KEYS
        },
        'processed_motions': processed,
        'failed_motions': failed,
    }


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml_config(config_path)
    validate_config(config)

    config_dir = config_path.parent
    model_folder = resolve_path(config['model_folder'], config_dir)
    output_root = resolve_path(config['output_root'], config_dir)
    motion_root = None
    if config.get('motion_root') is not None:
        motion_root = resolve_path(config['motion_root'], config_dir)

    normalized_config = dict(config)
    normalized_config['model_folder'] = str(model_folder)
    normalized_config['output_root'] = str(output_root)
    normalized_config['motion_root'] = str(motion_root) if motion_root is not None else None

    output_root.mkdir(parents=True, exist_ok=True)

    print(f'Config: {config_path}')
    print(f'Model folder: {model_folder}')
    print(f'Output root: {output_root}')
    print(f'Motion root: {motion_root if motion_root is not None else "<not set>"}')
    print(f'Motions to process: {len(config["motions"])}')

    processed = []
    failed = []
    used_output_dirs = set()

    for index, raw_entry in enumerate(config['motions'], start=1):
        entry = normalize_motion_entry(raw_entry)
        raw_motion_value = entry.get('path', entry.get('motion_file'))
        motion_base_dir = motion_root if motion_root is not None else config_dir
        motion_path = resolve_path(raw_motion_value, motion_base_dir)
        output_dir = resolve_output_dir(entry, output_root, motion_path, motion_root)

        if output_dir in used_output_dirs:
            raise ValueError(f'Duplicate output directory resolved for {motion_path}: {output_dir}')
        used_output_dirs.add(output_dir)

        options = merge_motion_options(config, entry, motion_path)
        print(f'[{index}/{len(config["motions"])}] {motion_path}')
        print(f'  output_dir={output_dir}')
        print(
            f'  start_frame={options["start_frame"]}, num_frames={options["num_frames"]}, '
            f'output_fps={options["output_fps"]}, num_random_shapes={options["num_random_shapes"]}, '
            f'seed={options["seed"]}, device={options["device"]}, render_videos={options["render_videos"]}'
        )

        try:
            summary = variants_lib.generate_variants(
                model_folder=model_folder,
                motion_file=motion_path,
                output_dir=output_dir,
                start_frame=options['start_frame'],
                num_frames=options['num_frames'],
                output_fps=options['output_fps'],
                num_random_shapes=options['num_random_shapes'],
                seed=options['seed'],
                device=options['device'],
                render_videos=options['render_videos'],
                log_fn=print,
            )
            summary['resolved_seed'] = int(options['seed'])
            processed.append(summary)
        except Exception as exc:
            failed.append({
                'motion_file': str(motion_path),
                'output_dir': str(output_dir),
                'error': repr(exc),
            })
            print(f'  FAILED: {exc}')

    batch_manifest = build_batch_manifest(config_path, normalized_config, processed, failed)
    batch_manifest_path = output_root / 'batch_manifest.json'
    batch_manifest_path.write_text(json.dumps(batch_manifest, indent=2) + '\n')
    print(f'Wrote batch manifest -> {batch_manifest_path}')


if __name__ == '__main__':
    main()
