# PA-MagCapPoseVLM

Official code for **PA-MagCapPoseVLM: Multifunctional Photoacoustic
Vision-Language Perception for Magnetically Steerable Capsule Pose
Estimation**.

PA-MagCapPoseVLM combines photoacoustic (PA) appearance, robot pose, and a
three-bit surface code for continuous six-degree-of-freedom capsule pose
estimation. A vision-language branch performs three related image-level
functions: spatial-validity screening, code reading, and their joint decision.
A fixed consistency gate releases a code to the pose estimator only when the
three prompt forms agree.

This repository is a compact research release. It deliberately excludes PA
data, trained parameters, generated splits, predictions, logs, and the internal
experiment-orchestration scripts used during development.

## Method at a glance

| Component | Inputs | Output | Implementation |
|---|---|---|---|
| Grouped data split | PA image/pose records | train/validation/test manifests | `scripts/build_grouped_split.py` |
| Continuous pose model | PA image + robot pose + 3-bit code | translation and rotation | `scripts/train_pose.py` |
| Multifunctional VLM | PA image + task prompt | validity and/or 3-bit code | `scripts/train_vlm.py` |
| Constrained inference | VLM checkpoint + PA image | schema-valid JSON | `scripts/evaluate_vlm.py` |
| Consistency gate | 3 tasks × 3 prompt forms | released code or rejection | `scripts/apply_consistency_gate.py` |
| End-to-end evaluation | released VLM code + pose network | selective pose metrics | `scripts/evaluate_pose_with_vlm.py` |

The continuous pose model uses a ResNet-18 image encoder, separate robot-pose
encoders, code embeddings, translation and continuous-6D rotation heads, and a
fixed linear robot-translation prior. The optimization objective combines
Smooth L1 translation loss with geodesic rotation loss on SO(3).

## Repository layout

```text
PA-MagCapPoseVLM/
├── docs/data_format.md              # input schemas and evaluation layout
├── scripts/
│   ├── build_grouped_split.py       # keep all eight codes of a pose together
│   ├── build_vlm_manifests.py       # construct QC/code/joint task records
│   ├── train_pose.py                # pose model and comparison baselines
│   ├── train_vlm.py                 # QLoRA fine-tuning
│   ├── evaluate_vlm.py              # finite-state JSON decoding
│   ├── apply_consistency_gate.py    # fixed prompt-agreement policy
│   └── evaluate_pose_with_vlm.py    # VLM-to-pose evaluation
├── src/pa_magcapposevlm/prompts.py  # canonical prompts and legal codes
└── tests/                           # lightweight release checks
```

## Installation

Python 3.10 or later and a CUDA-capable PyTorch installation are recommended.
Create an isolated environment, install PyTorch for the CUDA version on the
target computer, and then install this repository:

```bash
git clone https://github.com/Mechanic707/PA-MagCapPoseVLM.git
cd PA-MagCapPoseVLM
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The VLM scripts load the base model from a local path and use 4-bit NF4
quantization. Confirm that the installed PyTorch, CUDA, and bitsandbytes builds
are mutually compatible before a full run.

## Data preparation

The pose dataset contains eight code observations for each physical pose. Build
the split at pose-group level so the eight observations stay together:

```bash
python scripts/build_grouped_split.py \
  --dataset-root /path/to/pose_dataset \
  --output-dir splits/seed42 \
  --seed 42
```

This writes `grouped_split.json`, `nearest_train.csv`, and
`split_audit.json`. The audit verifies the eight-code group structure, zero
group overlap, class counts, file hashes, and nearest-training-pose distances.
See [docs/data_format.md](docs/data_format.md) for the source JSON schema.

For VLM training, create a JSONL file containing accepted and rejected images,
their fixed partition, and the code for accepted images. Then generate all three
tasks and prompt forms:

```bash
python scripts/build_vlm_manifests.py \
  --examples /path/to/vlm_examples.jsonl \
  --output-dir manifests/vlm
```

The script never copies images or changes the supplied partition. Training and
validation examples are expanded under the `standard`, `concise`, and
`reordered` prompts; the test examples are kept as a single frozen set.

## Continuous pose estimation

Train the complete multimodal estimator:

```bash
python scripts/train_pose.py \
  --model multimodal \
  --split splits/seed42/grouped_split.json \
  --nearest-train splits/seed42/nearest_train.csv \
  --output-dir outputs/pose/seed7 \
  --seed 7 \
  --device cuda:0
```

The paper configuration uses 224 × 224 images, AdamW with learning rate
`3e-4` and weight decay `1e-4`, batch size 64, at most 40 epochs, an early
stopping patience of 10, and a rotation-loss weight of 0.25. Descriptive
alternatives accepted by `--model` are printed by `python
scripts/train_pose.py --help`.

## Multifunctional VLM

Fine-tune Qwen3-VL-2B-Instruct with QLoRA:

```bash
python scripts/train_vlm.py \
  --model-path /path/to/Qwen3-VL-2B-Instruct \
  --train-manifest manifests/vlm/train.jsonl \
  --val-manifest manifests/vlm/val.jsonl \
  --manifest-audit manifests/vlm/manifest_audit.json \
  --output-dir outputs/vlm/seed7 \
  --seed 7 \
  --device cuda:0 \
  --gradient-checkpointing
```

Defaults reproduce the public training configuration: one epoch, 224 × 224
image area, micro-batch size 2, gradient accumulation 8, learning rate `1e-5`,
weight decay `1e-4`, warm-up ratio 0.03, LoRA rank 8, and LoRA alpha 16 on
`q_proj` and `v_proj`.

The reported manuscript run starts from a compatible capsule-domain LoRA
initialization. Trained parameters are not stored in this repository. For exact
numerical reproduction, obtain that adapter from the authors and pass
`--initial-adapter /path/to/adapter`. Omitting the option creates a fresh LoRA
adapter from the base model; the code path is complete, but the initialization
and therefore the numerical result differ from the manuscript run.

Run the full configuration with seeds `7`, `42`, and `2026`.

## Constrained inference and consistency gating

Evaluate each of the three tasks (`qc`, `code`, `joint`) under each of the three
prompt forms (`standard`, `concise`, `reordered`). One invocation is:

```bash
python scripts/evaluate_vlm.py \
  --model-path /path/to/Qwen3-VL-2B-Instruct \
  --adapter-dir outputs/vlm/seed7/best_adapter \
  --manifest manifests/vlm/test.jsonl \
  --mode joint \
  --prompt-variant standard \
  --output-dir outputs/evaluation/seed7/joint/standard \
  --device cuda:0
```

The decoder permits only complete legal JSON objects. Repeat for all nine
task/prompt combinations using the directory layout in
[docs/data_format.md](docs/data_format.md), and apply the fixed gate:

```bash
python scripts/apply_consistency_gate.py \
  --evaluation-root outputs/evaluation/seed7 \
  --output-dir outputs/gate/seed7
```

A normal image is released only if all three QC decisions and all three joint
validity decisions are `VALID`, and the three code-only plus three joint code
outputs are non-null and identical. The policy is applied independently to each
checkpoint; it is not a cross-seed ensemble.

Finally, measure the pose result on the released subset:

```bash
python scripts/evaluate_pose_with_vlm.py \
  --split splits/seed42/grouped_split.json \
  --pose-checkpoint outputs/pose/seed7/best_model.pt \
  --gated-predictions outputs/gate/seed7/gated_predictions.jsonl \
  --output outputs/gate/seed7/pose_integration.json \
  --device cuda:0
```

The output reports coverage, released-code accuracy, and pose metrics with the
predicted codes versus reference codes on exactly the same released samples.

## Reproducibility and scope

- Random seeds used for repeated training are `7`, `42`, and `2026`.
- All scripts refuse to overwrite a non-empty output directory.
- Manifest, checkpoint, and prediction provenance is recorded with SHA-256
  hashes where applicable.
- VLM outputs are machine-parseable finite-state JSON rather than free text.
- This code is intended for research evaluation. It is not a clinical device or
  a substitute for operator review.

Data and trained parameters are omitted because they contain experiment assets
and are large. They can be made available by the authors on reasonable request,
subject to the applicable institutional and data-sharing conditions.

## Citation

Please cite the accompanying manuscript. A publication-ready BibTeX entry will
be added after bibliographic details are assigned.

## License

The source code is released under the [MIT License](LICENSE).
