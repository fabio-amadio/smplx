# Examples

This folder contains the main scripts for generating SMPL-X motion variants and inspecting the outputs.

## Single Motion Export

Generate one motion with the original shape plus random beta variants:

```bash
cd /home/famadio/Workspace/smplx
./.venv/bin/python examples/amass_to_smplx_variants.py \
  --model-folder models \
  --motion-file "ACCAD/Female1General_c3d/A10_-_lie_to_crouch_stageii.npz" \
  --output-dir output/a10_lie_to_crouch_variants \
  --num-random-shapes 15 \
  --output-fps 50 \
  --device cpu
```

Optional video rendering:

```bash
cd /home/famadio/Workspace/smplx
./.venv/bin/python examples/amass_to_smplx_variants.py \
  --model-folder models \
  --motion-file "ACCAD/Female1General_c3d/A10_-_lie_to_crouch_stageii.npz" \
  --output-dir output/a10_lie_to_crouch_variants_video \
  --num-random-shapes 15 \
  --output-fps 50 \
  --render-videos \
  --device cpu
```

Saved `.npz` fields:
- `body_names`
- `body_pos`
- `body_quat`
- `betas`
- `hz`

## Batch Export

Batch export is configured through a YAML file. Example config:
[batch_amass_to_smplx_variants.example.yaml](/home/famadio/Workspace/smplx/examples/batch_amass_to_smplx_variants.example.yaml)

Run it with:

```bash
cd /home/famadio/Workspace/smplx
./.venv/bin/python examples/batch_amass_to_smplx_variants.py \
  examples/batch_amass_to_smplx_variants.example.yaml
```

The YAML has top-level defaults plus a `motions:` list. Per-motion entries can override:
- `path` or `motion_file`
- `output_subdir` or `output_dir`
- `start_frame`
- `num_frames`
- `output_fps`
- `num_random_shapes`
- `seed`
- `device`
- `render_videos`

## Variant Comparison Plots

Plot the difference between `motion_shape.npz` and the random variants inside one generated folder:

```bash
cd /home/famadio/Workspace/smplx
./.venv/bin/python examples/plot_smplx_variant_folder_comparison.py \
  output/a10_lie_to_crouch_variants
```

This writes plots and metrics into a `comparison/` subfolder inside the selected output directory.
