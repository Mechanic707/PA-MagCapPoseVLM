# Data formats

The repository intentionally contains no PA images, robot records, split files,
or trained parameters. Paths in generated manifests point to files on the local
machine and are not copied by any script.

## Pose source directory

`build_grouped_split.py` expects a flat directory containing one image and one
pose JSON file per observation. A pose file is named
`YYYYMMDD_HHMMSS_<index>_pose.json`; its image has the same prefix and one of
`.png`, `.jpg`, `.jpeg`, `.tif`, or `.tiff`.

The JSON object must contain:

```json
{
  "Binary_encoding": "010",
  "pact_robot_modify": {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "euler_angles": {"rx": 0.0, "ry": 0.0, "rz": 0.0}
  },
  "teacher_capsule_modify": {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "euler_angles": {"rx": 0.0, "ry": 0.0, "rz": 0.0}
  }
}
```

Eight consecutive observations form one pose group and must contain all codes
from `000` through `111`. The split builder validates this invariant and keeps a
pose group entirely in one partition.

## VLM examples manifest

`build_vlm_manifests.py` accepts JSON Lines. Each line is an independent image:

```json
{"example_id":"normal-0001","partition":"train","image_path":"D:/data/normal-0001.png","label":"ACCEPT","true_code":"010","pose_group_id":"group-0001"}
{"example_id":"reject-0001","partition":"test","image_path":"D:/data/reject-0001.png","label":"REJECT","true_code":null}
```

Required fields are `example_id`, `partition`, `image_path`, and `label`.
Accepted examples additionally require `true_code`. Optional provenance fields
are preserved. Partitions are supplied by the researcher; the script does not
move images or silently reassign examples.

Training and validation examples are expanded into the three fixed,
semantically equivalent prompt forms. Test examples remain single records and
are evaluated independently under each prompt form.

## VLM evaluation layout

Run `evaluate_vlm.py` nine times for one checkpoint and store outputs as:

```text
evaluation_root/
  qc/{standard,concise,reordered}/predictions.jsonl
  code/{standard,concise,reordered}/predictions.jsonl
  joint/{standard,concise,reordered}/predictions.jsonl
```

This is the input layout for `apply_consistency_gate.py`.
