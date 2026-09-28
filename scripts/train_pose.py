#!/usr/bin/env python3
"""Train and evaluate capsule-pose estimators on a pose-group split.

The script contains the multimodal PA-MagCapPose model and the comparison
baselines used in the paper.  It uses training-only normalization, continuous
6D rotation outputs, and geodesic SO(3) evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


MODEL_DESCRIPTIONS = {
    "B0": "training-set mean pose",
    "B1": "analytic rigid robot-to-capsule transform",
    "B2": "robot-pose Ridge regression",
    "B3": "robot-pose MLP",
    "B4": "semantic-code-only MLP",
    "B5": "robot-pose plus semantic-code MLP",
    "B6": "PA-image-only ResNet-18",
    "B7": "PA image plus robot pose plus semantic code",
}
ANALYTIC_MODELS = {"B0", "B1", "B2"}
IMAGE_MODELS = {"B6", "B7"}
NEURAL_MODELS = set(MODEL_DESCRIPTIONS) - ANALYTIC_MODELS
MODEL_ALIASES = {
    "mean-pose": "B0",
    "rigid-transform": "B1",
    "ridge-robot": "B2",
    "robot-mlp": "B3",
    "code-mlp": "B4",
    "robot-code-mlp": "B5",
    "image-resnet": "B6",
    "multimodal": "B7",
}
AXES = ("x", "y", "z", "rx", "ry", "rz")


@dataclass(frozen=True)
class Record:
    partition: str
    pose_group_id: str
    sample_index: int
    code: int
    img_path: str
    json_path: str
    robot_pose: tuple[float, float, float, float, float, float]
    target_pose: tuple[float, float, float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ALIASES))
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--nearest-train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rotation-loss-weight", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as stream:
        stream.write(text)
        temporary = stream.name
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def parse_pose(payload: dict[str, Any], field: str) -> tuple[float, ...]:
    pose = payload[field]
    position = pose["position"]
    angles = pose["euler_angles"]
    return (
        float(position["x"]),
        float(position["y"]),
        float(position["z"]),
        float(angles["rx"]),
        float(angles["ry"]),
        float(angles["rz"]),
    )


def parse_code(raw: Any) -> int:
    if isinstance(raw, str):
        text = raw.strip()
        return int(text, 2) if set(text) <= {"0", "1"} else int(text)
    return int(raw)


def load_records(split_path: Path) -> dict[str, list[Record]]:
    with split_path.open("r", encoding="utf-8") as stream:
        split = json.load(stream)
    output: dict[str, list[Record]] = {}
    for partition in ("train", "val", "test"):
        rows: list[Record] = []
        for item in split[partition]:
            json_path = Path(item["json_path"])
            with json_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            code = parse_code(payload["Binary_encoding"])
            if code != int(item["semantic_code"], 2):
                raise ValueError(f"code mismatch for {json_path}")
            rows.append(
                Record(
                    partition=partition,
                    pose_group_id=str(item["pose_group_id"]),
                    sample_index=int(item["sample_index"]),
                    code=code,
                    img_path=str(item["img_path"]),
                    json_path=str(json_path),
                    robot_pose=parse_pose(payload, "pact_robot_modify"),
                    target_pose=parse_pose(payload, "teacher_capsule_modify"),
                )
            )
        output[partition] = rows
    return output


def load_nearest_distance(path: Path) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["pose_source"] != "pact_pose":
                continue
            result[(row["partition"], row["pose_group_id"])] = float(
                row["distance_mm"]
            )
    return result


def arrays(records: Sequence[Record]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot = np.asarray([row.robot_pose for row in records], dtype=np.float64)
    target = np.asarray([row.target_pose for row in records], dtype=np.float64)
    code = np.asarray([row.code for row in records], dtype=np.int64)
    return robot, target, code


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def rotation_6d_to_matrix_numpy(value: np.ndarray) -> np.ndarray:
    a1 = value[..., 0:3]
    a2 = value[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    projection = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - projection * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rotation_6d_to_matrix_torch(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    a1 = value[..., 0:3]
    a2 = value[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(
        a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1,
        dim=-1,
        eps=1e-6,
    )
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack([b1, b2, b3], dim=-1)


def robot_features(robot_pose: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(robot_pose[:, 3:6])
    return np.concatenate(
        [robot_pose[:, 0:3], np.sin(radians), np.cos(radians)], axis=1
    )


def fit_analytic(
    model_id: str, train_records: Sequence[Record]
) -> dict[str, Any]:
    robot, target, _ = arrays(train_records)
    target_rotation = Rotation.from_euler("xyz", target[:, 3:6], degrees=True)
    if model_id == "B0":
        return {
            "model_id": model_id,
            "translation": target[:, 0:3].mean(axis=0),
            "rotation_matrix": target_rotation.mean().as_matrix(),
        }

    if model_id == "B1":
        x_center = robot[:, 0:3].mean(axis=0)
        y_center = target[:, 0:3].mean(axis=0)
        covariance = (robot[:, 0:3] - x_center).T @ (
            target[:, 0:3] - y_center
        )
        u, _, vt = np.linalg.svd(covariance)
        rigid_rotation = vt.T @ u.T
        if np.linalg.det(rigid_rotation) < 0:
            vt[-1, :] *= -1
            rigid_rotation = vt.T @ u.T
        rigid_translation = y_center - rigid_rotation @ x_center
        robot_rotation = Rotation.from_euler("xyz", robot[:, 3:6], degrees=True)
        relative_rotation = target_rotation * robot_rotation.inv()
        return {
            "model_id": model_id,
            "translation_rotation": rigid_rotation,
            "translation_offset": rigid_translation,
            "rotation_offset_matrix": relative_rotation.mean().as_matrix(),
        }

    if model_id == "B2":
        scaler = StandardScaler().fit(robot_features(robot))
        features = scaler.transform(robot_features(robot))
        translation_model = Ridge(alpha=1e-3).fit(features, target[:, 0:3])
        target_rotation_6d = matrix_to_rotation_6d(target_rotation.as_matrix())
        rotation_model = Ridge(alpha=1e-3).fit(features, target_rotation_6d)
        return {
            "model_id": model_id,
            "scaler": scaler,
            "translation_model": translation_model,
            "rotation_model": rotation_model,
        }
    raise ValueError(model_id)


def predict_analytic(
    fitted: dict[str, Any], records: Sequence[Record]
) -> tuple[np.ndarray, np.ndarray]:
    model_id = fitted["model_id"]
    robot, _, _ = arrays(records)
    if model_id == "B0":
        translation = np.repeat(
            np.asarray(fitted["translation"])[None, :], len(records), axis=0
        )
        rotation_matrix = np.repeat(
            np.asarray(fitted["rotation_matrix"])[None, :, :], len(records), axis=0
        )
        return translation, rotation_matrix
    if model_id == "B1":
        translation = (
            np.asarray(fitted["translation_rotation"]) @ robot[:, 0:3].T
        ).T + np.asarray(fitted["translation_offset"])
        robot_rotation = Rotation.from_euler(
            "xyz", robot[:, 3:6], degrees=True
        )
        offset = Rotation.from_matrix(fitted["rotation_offset_matrix"])
        return translation, (offset * robot_rotation).as_matrix()
    features = fitted["scaler"].transform(robot_features(robot))
    translation = fitted["translation_model"].predict(features)
    rotation_6d = fitted["rotation_model"].predict(features)
    return translation, rotation_6d_to_matrix_numpy(rotation_6d)


class PoseDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Record],
        robot_mean: np.ndarray,
        robot_std: np.ndarray,
        target_translation_mean: np.ndarray,
        target_translation_std: np.ndarray,
        image_size: int,
        use_image: bool,
    ) -> None:
        self.records = list(records)
        self.robot_mean = robot_mean.astype(np.float32)
        self.robot_std = target_safe_std(robot_std).astype(np.float32)
        self.target_translation_mean = target_translation_mean.astype(np.float32)
        self.target_translation_std = target_safe_std(
            target_translation_std
        ).astype(np.float32)
        self.image_size = image_size
        self.use_image = use_image

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        robot = np.asarray(row.robot_pose, dtype=np.float32)
        target = np.asarray(row.target_pose, dtype=np.float32)
        output: dict[str, Any] = {
            "robot": torch.from_numpy((robot - self.robot_mean) / self.robot_std),
            "code": torch.tensor(row.code, dtype=torch.long),
            "target_translation_norm": torch.from_numpy(
                (target[:3] - self.target_translation_mean)
                / self.target_translation_std
            ),
            "target_rotation": torch.from_numpy(
                Rotation.from_euler("xyz", target[3:6], degrees=True)
                .as_matrix()
                .astype(np.float32)
            ),
            "target_pose": torch.from_numpy(target),
            "partition": row.partition,
            "pose_group_id": row.pose_group_id,
            "sample_index": row.sample_index,
            "img_path": row.img_path,
            "json_path": row.json_path,
        }
        if self.use_image:
            image = cv2.imread(row.img_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(row.img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
            )
            image = image.astype(np.float32) / 255.0
            image = (image - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
                [0.229, 0.224, 0.225], np.float32
            )
            output["image"] = torch.from_numpy(image.transpose(2, 0, 1).copy())
        return output


def target_safe_std(value: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(value) < 1e-8, 1.0, value)


class PoseRegressor(nn.Module):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id = model_id
        self.code_embedding_trans = nn.Embedding(8, 16)
        self.code_embedding_rot = nn.Embedding(8, 16)
        self.robot_encoder_trans = self._make_robot_encoder()
        self.robot_encoder_rot = self._make_robot_encoder()
        self.robot_translation_skip = nn.Linear(6, 3)
        nn.init.zeros_(self.robot_translation_skip.weight)
        nn.init.zeros_(self.robot_translation_skip.bias)
        self.image_encoder: nn.Module | None = None
        feature_dim: int
        if model_id == "B3":
            feature_dim = 64
        elif model_id == "B4":
            feature_dim = 16
        elif model_id == "B5":
            feature_dim = 80
        elif model_id in IMAGE_MODELS:
            backbone = resnet18(weights=None)
            backbone.fc = nn.Identity()
            self.image_encoder = backbone
            feature_dim = 512 if model_id == "B6" else 592
        else:
            raise ValueError(model_id)
        self.translation_head = self._make_head(feature_dim, 3)
        self.rotation_head = self._make_head(feature_dim, 6)
        final_rotation = self.rotation_head[-1]
        assert isinstance(final_rotation, nn.Linear)
        with torch.no_grad():
            final_rotation.bias[:] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    @staticmethod
    def _make_robot_encoder() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

    @staticmethod
    def _make_head(feature_dim: int, output_dim: int) -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )
        final = head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        return head

    def set_fixed_robot_translation_prior(
        self, weight: np.ndarray, bias: np.ndarray
    ) -> None:
        with torch.no_grad():
            self.robot_translation_skip.weight.copy_(
                torch.as_tensor(weight, dtype=torch.float32)
            )
            self.robot_translation_skip.bias.copy_(
                torch.as_tensor(bias, dtype=torch.float32)
            )
        self.robot_translation_skip.requires_grad_(False)

    def forward(
        self,
        robot: torch.Tensor,
        code: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.model_id == "B3":
            translation_feature = self.robot_encoder_trans(robot)
            rotation_feature = self.robot_encoder_rot(robot)
        elif self.model_id == "B4":
            translation_feature = self.code_embedding_trans(code)
            rotation_feature = self.code_embedding_rot(code)
        elif self.model_id == "B5":
            translation_feature = torch.cat(
                [self.robot_encoder_trans(robot), self.code_embedding_trans(code)],
                dim=1,
            )
            rotation_feature = torch.cat(
                [self.robot_encoder_rot(robot), self.code_embedding_rot(code)], dim=1
            )
        elif self.model_id == "B6":
            assert self.image_encoder is not None and image is not None
            translation_feature = self.image_encoder(image)
            rotation_feature = translation_feature
        else:
            assert self.image_encoder is not None and image is not None
            image_feature = self.image_encoder(image)
            translation_feature = torch.cat(
                [
                    image_feature,
                    self.robot_encoder_trans(robot),
                    self.code_embedding_trans(code),
                ],
                dim=1,
            )
            rotation_feature = torch.cat(
                [
                    image_feature,
                    self.robot_encoder_rot(robot),
                    self.code_embedding_rot(code),
                ],
                dim=1,
            )
        translation = self.translation_head(translation_feature)
        if self.model_id in {"B3", "B5", "B7"}:
            translation = translation + self.robot_translation_skip(robot)
        return torch.cat([translation, self.rotation_head(rotation_feature)], dim=1)


def neural_loss(
    prediction: torch.Tensor,
    target_translation_norm: torch.Tensor,
    target_rotation: torch.Tensor,
    rotation_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    translation_loss = F.smooth_l1_loss(
        prediction[:, :3], target_translation_norm
    )
    predicted_rotation = rotation_6d_to_matrix_torch(prediction[:, 3:])
    relative = predicted_rotation.transpose(1, 2) @ target_rotation
    trace = relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2]
    cosine = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    skew_vector = torch.stack(
        [
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ],
        dim=1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew_vector, dim=1)
    rotation_loss = torch.atan2(sine, cosine).mean()
    total = translation_loss + rotation_weight * rotation_loss
    return total, translation_loss, rotation_loss


def loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def train_neural(
    args: argparse.Namespace,
    records: dict[str, list[Record]],
    output_dir: Path,
    device: torch.device,
) -> tuple[PoseRegressor, dict[str, np.ndarray], list[dict[str, float]]]:
    train_robot, train_target, _ = arrays(records["train"])
    stats = {
        "robot_mean": train_robot.mean(axis=0).astype(np.float32),
        "robot_std": target_safe_std(train_robot.std(axis=0)).astype(np.float32),
        "target_translation_mean": train_target[:, :3].mean(axis=0).astype(np.float32),
        "target_translation_std": target_safe_std(
            train_target[:, :3].std(axis=0)
        ).astype(np.float32),
    }
    use_image = args.model in IMAGE_MODELS
    datasets = {
        partition: PoseDataset(
            rows,
            stats["robot_mean"],
            stats["robot_std"],
            stats["target_translation_mean"],
            stats["target_translation_std"],
            args.image_size,
            use_image,
        )
        for partition, rows in records.items()
    }
    train_loader = loader(
        datasets["train"],
        args.batch_size,
        args.num_workers,
        True,
        device,
        args.seed,
    )
    val_loader = loader(
        datasets["val"],
        args.batch_size,
        args.num_workers,
        False,
        device,
        args.seed,
    )
    model = PoseRegressor(args.model).to(device)
    if args.model in {"B3", "B5", "B7"}:
        normalized_robot = (train_robot - stats["robot_mean"]) / stats["robot_std"]
        normalized_target_translation = (
            train_target[:, :3] - stats["target_translation_mean"]
        ) / stats["target_translation_std"]
        design = np.concatenate(
            [normalized_robot, np.ones((len(normalized_robot), 1))], axis=1
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            design, normalized_target_translation, rcond=None
        )
        model.set_fixed_robot_translation_prior(
            coefficients[:6, :].T,
            coefficients[6, :],
        )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    # The RTX 5060 Ti supports AMP, but full FP32 is intentionally used here:
    # mixed precision caused non-finite gradients in the 6D-to-SO(3) projection.
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_translation_losses: list[float] = []
        train_rotation_losses: list[float] = []
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            robot = batch["robot"].to(device, non_blocking=True)
            code = batch["code"].to(device, non_blocking=True)
            image = (
                batch["image"].to(device, non_blocking=True) if use_image else None
            )
            target_translation = batch["target_translation_norm"].to(
                device, non_blocking=True
            )
            target_rotation = batch["target_rotation"].to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model(robot, code, image)
                loss, translation_loss, rotation_loss = neural_loss(
                    prediction,
                    target_translation,
                    target_rotation,
                    args.rotation_loss_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch={epoch}, batch={batch_index}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            train_translation_losses.append(float(translation_loss.detach().cpu()))
            train_rotation_losses.append(float(rotation_loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        val_translation_losses: list[float] = []
        val_rotation_losses: list[float] = []
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
                    break
                robot = batch["robot"].to(device, non_blocking=True)
                code = batch["code"].to(device, non_blocking=True)
                image = (
                    batch["image"].to(device, non_blocking=True)
                    if use_image
                    else None
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    prediction = model(robot, code, image)
                    loss, translation_loss, rotation_loss = neural_loss(
                        prediction,
                        batch["target_translation_norm"].to(device),
                        batch["target_rotation"].to(device),
                        args.rotation_loss_weight,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite validation loss at epoch={epoch}, batch={batch_index}"
                    )
                val_losses.append(float(loss.detach().cpu()))
                val_translation_losses.append(float(translation_loss.detach().cpu()))
                val_rotation_losses.append(float(rotation_loss.detach().cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_translation_loss": float(np.mean(train_translation_losses)),
                "train_rotation_loss_rad": float(np.mean(train_rotation_losses)),
                "val_translation_loss": float(np.mean(val_translation_losses)),
                "val_rotation_loss_rad": float(np.mean(val_rotation_losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1
        scheduler.step()
        minimum_epochs = max(20, args.epochs // 2)
        if (
            epoch >= minimum_epochs
            and patience_count >= args.patience
            and not args.smoke
        ):
            print(f"early_stop epoch={epoch} best_val={best_val:.6f}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("no neural checkpoint was produced")
    model.load_state_dict(best_state)
    checkpoint = {
        "model_id": args.model,
        "model_state_dict": best_state,
        "normalization": {key: value.tolist() for key, value in stats.items()},
        "image_size": args.image_size,
        "best_val_loss": best_val,
    }
    temporary = output_dir / "best_model.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, output_dir / "best_model.pt")
    return model, stats, history


def predict_neural(
    args: argparse.Namespace,
    model: PoseRegressor,
    stats: dict[str, np.ndarray],
    records: Sequence[Record],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = PoseDataset(
        records,
        stats["robot_mean"],
        stats["robot_std"],
        stats["target_translation_mean"],
        stats["target_translation_std"],
        args.image_size,
        args.model in IMAGE_MODELS,
    )
    data_loader = loader(
        dataset,
        args.batch_size,
        args.num_workers,
        False,
        device,
        args.seed,
    )
    predicted_translation: list[np.ndarray] = []
    predicted_rotation: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
                break
            robot = batch["robot"].to(device, non_blocking=True)
            code = batch["code"].to(device, non_blocking=True)
            image = (
                batch["image"].to(device, non_blocking=True)
                if args.model in IMAGE_MODELS
                else None
            )
            prediction = model(robot, code, image)
            trans_norm = prediction[:, :3].detach().cpu().numpy()
            translation = (
                trans_norm * stats["target_translation_std"]
                + stats["target_translation_mean"]
            )
            rotation = (
                rotation_6d_to_matrix_torch(prediction[:, 3:])
                .detach()
                .cpu()
                .numpy()
            )
            predicted_translation.append(translation)
            predicted_rotation.append(rotation)
    return np.concatenate(predicted_translation), np.concatenate(predicted_rotation)


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compute_metrics(
    target_pose: np.ndarray,
    predicted_translation: np.ndarray,
    predicted_rotation: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    target_rotation = Rotation.from_euler(
        "xyz", target_pose[:, 3:6], degrees=True
    )
    prediction_rotation = Rotation.from_matrix(predicted_rotation)
    translation_delta = predicted_translation - target_pose[:, :3]
    translation_error = np.linalg.norm(translation_delta, axis=1)
    rotation_error = np.rad2deg(
        (prediction_rotation.inv() * target_rotation).magnitude()
    )
    predicted_euler = prediction_rotation.as_euler("xyz", degrees=True)
    circular_euler_delta = (
        predicted_euler - target_pose[:, 3:6] + 180.0
    ) % 360.0 - 180.0
    metrics: dict[str, Any] = {
        "translation_euclidean_mm": percentile_summary(translation_error),
        "rotation_geodesic_deg": percentile_summary(rotation_error),
        "translation_axis_mae_mm": {
            axis: float(np.mean(np.abs(translation_delta[:, index])))
            for index, axis in enumerate(AXES[:3])
        },
        "translation_axis_rmse_mm": {
            axis: float(np.sqrt(np.mean(np.square(translation_delta[:, index]))))
            for index, axis in enumerate(AXES[:3])
        },
        "rotation_euler_axis_mae_deg": {
            axis: float(np.mean(np.abs(circular_euler_delta[:, index])))
            for index, axis in enumerate(AXES[3:])
        },
    }
    return metrics, translation_error, rotation_error, predicted_euler


def evaluate_and_write(
    output_dir: Path,
    records: Sequence[Record],
    predicted_translation: np.ndarray,
    predicted_rotation: np.ndarray,
    nearest: dict[tuple[str, str], float],
) -> dict[str, Any]:
    evaluated_records = list(records[: len(predicted_translation)])
    target_pose = np.asarray(
        [row.target_pose for row in evaluated_records], dtype=np.float64
    )
    metrics, trans_error, rot_error, predicted_euler = compute_metrics(
        target_pose, predicted_translation, predicted_rotation
    )
    code_metrics: dict[str, Any] = {}
    for code in range(8):
        mask = np.asarray([row.code == code for row in evaluated_records])
        code_metrics[f"{code:03b}"] = compute_metrics(
            target_pose[mask],
            predicted_translation[mask],
            predicted_rotation[mask],
        )[0]
    metrics["per_code"] = code_metrics

    distances = np.asarray(
        [nearest[(row.partition, row.pose_group_id)] for row in evaluated_records]
    )
    distance_bins = {
        "lt_3mm": distances < 3.0,
        "3_to_4mm": (distances >= 3.0) & (distances < 4.0),
        "ge_4mm": distances >= 4.0,
    }
    metrics["by_nearest_train_pact_distance"] = {}
    for label, mask in distance_bins.items():
        if not np.any(mask):
            continue
        metrics["by_nearest_train_pact_distance"][label] = compute_metrics(
            target_pose[mask],
            predicted_translation[mask],
            predicted_rotation[mask],
        )[0]

    prediction_path = output_dir / "test_predictions.csv"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        delete=False,
        dir=output_dir,
    ) as stream:
        fieldnames = [
            "partition",
            "pose_group_id",
            "sample_index",
            "semantic_code",
            "json_path",
            "img_path",
            *[f"gt_{axis}" for axis in AXES],
            *[f"pred_{axis}" for axis in AXES],
            "translation_error_mm",
            "rotation_geodesic_error_deg",
            "nearest_train_pact_distance_mm",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(evaluated_records):
            gt = target_pose[index]
            prediction = np.concatenate(
                [predicted_translation[index], predicted_euler[index]]
            )
            values: dict[str, Any] = {
                "partition": row.partition,
                "pose_group_id": row.pose_group_id,
                "sample_index": row.sample_index,
                "semantic_code": f"{row.code:03b}",
                "json_path": row.json_path,
                "img_path": row.img_path,
                "translation_error_mm": trans_error[index],
                "rotation_geodesic_error_deg": rot_error[index],
                "nearest_train_pact_distance_mm": distances[index],
            }
            values.update({f"gt_{axis}": gt[i] for i, axis in enumerate(AXES)})
            values.update(
                {f"pred_{axis}": prediction[i] for i, axis in enumerate(AXES)}
            )
            writer.writerow(values)
        temporary = stream.name
    os.replace(temporary, prediction_path)
    metrics["prediction_file"] = str(prediction_path)
    metrics["prediction_sha256"] = sha256_file(prediction_path)
    return metrics


def environment_info(device: torch.device) -> dict[str, Any]:
    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision_model": "resnet18(weights=None)",
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "selected_device": str(device),
        "selected_gpu_name": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def main() -> int:
    args = parse_args()
    args.model = MODEL_ALIASES[args.model]
    if args.smoke:
        args.epochs = 1
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_batches = args.max_eval_batches or 2
        args.num_workers = 0
    set_reproducibility(args.seed)

    split_path = args.split.expanduser().resolve()
    nearest_path = args.nearest_train.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(
        f"model={args.model} description={MODEL_DESCRIPTIONS[args.model]}",
        flush=True,
    )
    print(f"split={split_path}", flush=True)
    records = load_records(split_path)
    nearest = load_nearest_distance(nearest_path)
    if len(nearest) != 360:
        raise RuntimeError(f"expected 360 val/test group distances, found {len(nearest)}")

    requested_device = torch.device(args.device)
    if args.model in ANALYTIC_MODELS:
        device = torch.device("cpu")
    else:
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = requested_device

    history: list[dict[str, float]] = []
    if args.model in ANALYTIC_MODELS:
        fitted = fit_analytic(args.model, records["train"])
        joblib.dump(fitted, output_dir / "analytic_model.joblib")
        predicted_translation, predicted_rotation = predict_analytic(
            fitted, records["test"]
        )
    else:
        model, stats, history = train_neural(args, records, output_dir, device)
        predicted_translation, predicted_rotation = predict_neural(
            args, model, stats, records["test"], device
        )

    metrics = evaluate_and_write(
        output_dir,
        records["test"],
        predicted_translation,
        predicted_rotation,
        nearest,
    )
    metrics.update(
        {
            "model_id": args.model,
            "description": MODEL_DESCRIPTIONS[args.model],
            "smoke_test": bool(args.smoke),
            "evaluated_test_samples": len(predicted_translation),
            "elapsed_seconds": time.time() - start,
        }
    )
    run_config = vars(args).copy()
    run_config["split"] = str(split_path)
    run_config["nearest_train"] = str(nearest_path)
    run_config["output_dir"] = str(output_dir)
    run_config["split_sha256"] = sha256_file(split_path)
    run_config["script_sha256"] = sha256_file(Path(__file__).resolve())
    run_config["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(output_dir / "run_config.json", run_config)
    atomic_write_json(output_dir / "environment.json", environment_info(device))
    atomic_write_json(output_dir / "training_history.json", history)
    atomic_write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"RESULT_DIR={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
