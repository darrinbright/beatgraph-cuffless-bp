import argparse
import logging
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import (
    ABP_WAVE_NORM_MEAN,
    ABP_WAVE_NORM_STD,
    BATCH_SIZE,
    BEAT_LEN,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    INPUT_MODES,
    IW_BANDWIDTH,
    IW_MAX_WEIGHT,
    LEARNING_RATE,
    MODEL_TYPES,
    NUM_WORKERS,
    TRAIN_SPLIT,
    VALIDATE_EVERY,
    VAL_SPLIT,
    WARMUP_EPOCHS,
)
from .dataset import (
    VitalDBBeatSequenceDataset,
    read_subjects,
    select_cignn_subjects,
    select_records_by_subjects,
    split_records,
    split_records_stratified_by_bp,
    split_subjects_loocv,
)
from .metrics import aggregate_predictions_by_subject, format_metrics, regression_metrics
from .model import build_model, count_parameters
from .paths import AAMI_CAL_MAT, DEFAULT_TRAIN_OUTPUT, TRAIN_MAT

logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_batch(samples):
    batch_size = len(samples)
    channels = int(samples[0]["inputs"].shape[0])
    beat_counts = [len(sample["beat_boundaries"]) for sample in samples]
    max_beats = max(beat_counts)
    beat_inputs = torch.zeros(batch_size, max_beats, channels, BEAT_LEN, dtype=torch.float32)
    beat_mask = torch.zeros(batch_size, max_beats, dtype=torch.bool)
    beat_features = torch.zeros(batch_size, max_beats, 4, dtype=torch.float32)

    for sample_index, sample in enumerate(samples):
        signal = sample["inputs"]
        boundaries = sample["beat_boundaries"]
        features = sample["beat_features"]
        feature_count = int(features.shape[0]) if features.ndim > 0 else 0
        for beat_index, (start, end) in enumerate(boundaries):
            segment = signal[:, start:end].unsqueeze(0)
            beat_inputs[sample_index, beat_index] = F.interpolate(
                segment, size=BEAT_LEN, mode="linear", align_corners=False,
            ).squeeze(0)
            beat_mask[sample_index, beat_index] = True
            if beat_index < feature_count:
                beat_features[sample_index, beat_index] = features[beat_index]

    return {
        "inputs": torch.stack([sample["inputs"] for sample in samples], dim=0),
        "blood_pressure": torch.stack([sample["blood_pressure"] for sample in samples], dim=0),
        "abp_waveform": torch.stack([sample["abp_waveform"] for sample in samples], dim=0),
        "demographics": torch.stack([sample["demographics"] for sample in samples], dim=0),
        "record_index": torch.stack([sample["record_index"] for sample in samples], dim=0),
        "subject_id": [sample["subject_id"] for sample in samples],
        "beat_boundaries": [sample["beat_boundaries"] for sample in samples],
        "beat_features": beat_features,
        "beat_inputs": beat_inputs,
        "beat_mask": beat_mask,
    }


def forward_model(model, batch, device, demographics):
    return model(
        batch["inputs"].to(device, non_blocking=True),
        batch["beat_boundaries"],
        batch["beat_features"].to(device, non_blocking=True),
        batch["beat_inputs"].to(device, non_blocking=True),
        batch["beat_mask"].to(device, non_blocking=True),
        demographics,
    )


def input_spec(input_mode):
    mapping = {
        "ppg_d1_d2": {"channels": 3, "ppg_index": 0, "beat_channels": (0, 1, 2)},
        "ecg_ppg_d1_d2": {"channels": 4, "ppg_index": 1, "beat_channels": (0, 1)},
    }
    return mapping[input_mode]


def make_loader(dataset, batch_size, shuffle, num_workers, sampler=None):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_batch,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
        kwargs["timeout"] = 120
    return DataLoader(dataset, **kwargs)


def compute_target_stats(dataset):
    targets = np.asarray(dataset.targets, dtype=np.float32)
    return {
        "mean": targets.mean(axis=0),
        "std": np.clip(targets.std(axis=0), 1e-6, None),
    }


def compute_demographics_stats(dataset):
    demographics = np.asarray(dataset.demos, dtype=np.float32)
    mean = np.nanmean(demographics, axis=0)
    demographics = np.where(np.isfinite(demographics), demographics, mean)
    std = np.clip(demographics.std(axis=0), 1e-6, None)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def make_balanced_sampler(dataset, num_bins=10):
    targets = np.asarray(dataset.targets, dtype=np.float32)
    systolic = targets[:, 0]
    edges = np.linspace(float(systolic.min()), float(systolic.max()), num_bins + 1)
    bin_index = np.clip(np.digitize(systolic, edges[1:-1]), 0, num_bins - 1)
    counts = np.bincount(bin_index, minlength=num_bins).astype(np.float64)
    weights = 1.0 / np.clip(counts[bin_index], 1.0, None)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def make_importance_weighted_sampler(train_dataset, aami_cal_file, max_weight, bandwidth):
    if not Path(aami_cal_file).is_file():
        raise FileNotFoundError(aami_cal_file)
    with h5py.File(aami_cal_file, "r") as handle:
        aami_systolic = np.asarray(handle["Subset"]["SBP"][0, :], dtype=np.float32)

    train_systolic = np.asarray(train_dataset.targets, dtype=np.float32)[:, 0]
    train_systolic = train_systolic[np.isfinite(train_systolic)]
    aami_systolic = aami_systolic[np.isfinite(aami_systolic)]
    systolic_min = float(min(train_systolic.min(), aami_systolic.min()))
    systolic_max = float(max(train_systolic.max(), aami_systolic.max()))
    systolic_range = systolic_max - systolic_min + 1e-6
    kde_bandwidth = bandwidth * systolic_range
    num_bins = 512
    edges = np.linspace(systolic_min, systolic_max, num_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    sigma_bins = max(kde_bandwidth / bin_width, 1e-6)

    train_counts, _ = np.histogram(train_systolic, bins=edges, density=True)
    target_counts, _ = np.histogram(aami_systolic, bins=edges, density=True)
    train_grid = gaussian_filter1d(train_counts.astype(np.float64), sigma=sigma_bins, mode="nearest")
    target_grid = gaussian_filter1d(target_counts.astype(np.float64), sigma=sigma_bins, mode="nearest")

    full_train_systolic = np.asarray(train_dataset.targets, dtype=np.float32)[:, 0]
    train_density = np.interp(full_train_systolic, centers, train_grid, left=train_grid[0], right=train_grid[-1])
    target_density = np.interp(full_train_systolic, centers, target_grid, left=target_grid[0], right=target_grid[-1])
    weights = np.clip(target_density / np.clip(train_density, 1e-10, None), 0.0, max_weight)
    weights = weights / weights.mean()
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def denormalize_targets(targets, target_mean, target_std):
    return targets * target_std + target_mean


def normalize_targets(targets, target_mean, target_std):
    return (targets - target_mean) / target_std


def normalize_demographics(demographics, demo_mean, demo_std):
    demographics = torch.where(torch.isfinite(demographics), demographics, demo_mean)
    return (demographics - demo_mean) / demo_std


def apply_group_calibration(predictions, targets, subject_ids, calibration_samples):
    calibrated = np.asarray(predictions, dtype=np.float32).copy()
    targets = np.asarray(targets, dtype=np.float32)
    subject_ids = np.asarray(subject_ids)
    evaluation_mask = np.ones(len(calibrated), dtype=bool)
    for subject in np.unique(subject_ids):
        subject_mask = np.flatnonzero(subject_ids == subject)
        if subject_mask.size == 0:
            continue
        calibration_index = subject_mask[: min(int(calibration_samples), subject_mask.size)]
        bias = np.mean(calibrated[calibration_index] - targets[calibration_index], axis=0, keepdims=True)
        calibrated[subject_mask] -= bias
        evaluation_mask[calibration_index] = False
    return calibrated, evaluation_mask


def waveform_loss(predicted_waveform, true_waveform, device):
    true_normalized = (true_waveform.to(device) - ABP_WAVE_NORM_MEAN) / ABP_WAVE_NORM_STD
    return F.l1_loss(predicted_waveform, true_normalized)


def run_train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    target_mean,
    target_std,
    demo_mean,
    demo_std,
    accumulate_steps,
    wave_loss_weight,
    scalar_loss_weight,
):
    model.train()
    total_mae = 0.0
    total_count = 0
    total_density = 0.0
    amp_enabled = device.type == "cuda"

    for batch_index, batch in enumerate(loader):
        targets = batch["blood_pressure"].to(device, non_blocking=True)
        demographics = normalize_demographics(
            batch["demographics"].to(device, non_blocking=True), demo_mean, demo_std
        )
        normalized_targets = normalize_targets(targets, target_mean, target_std)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            predicted_bp, predicted_waveform = forward_model(model, batch, device, demographics)
            loss = (
                wave_loss_weight * waveform_loss(predicted_waveform, batch["abp_waveform"], device)
                + scalar_loss_weight * F.l1_loss(predicted_bp, normalized_targets)
            ) / accumulate_steps

        scaler.scale(loss).backward()
        if (batch_index + 1) % accumulate_steps == 0 or (batch_index + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = targets.size(0)
        predicted_mmhg = denormalize_targets(predicted_bp.detach(), target_mean, target_std)
        total_mae += float(F.l1_loss(predicted_mmhg, targets.detach()).cpu()) * batch_size
        total_count += batch_size
        total_density += float(model.graph_density) * batch_size

    return {
        "mae": total_mae / max(1, total_count),
        "graph_density": total_density / max(1, total_count),
    }


def evaluate_loader(model, loader, device, target_mean, target_std, demo_mean, demo_std):
    model.eval()
    predictions = []
    labels = []
    subject_ids = []
    total_count = 0
    total_density = 0.0

    with torch.no_grad():
        for batch in loader:
            targets = batch["blood_pressure"].to(device, non_blocking=True)
            demographics = normalize_demographics(
                batch["demographics"].to(device, non_blocking=True), demo_mean, demo_std
            )
            predicted_bp, _ = forward_model(model, batch, device, demographics)
            predicted_bp = denormalize_targets(predicted_bp, target_mean, target_std)
            predictions.append(predicted_bp.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())
            subject_ids.extend(batch["subject_id"])
            total_count += targets.size(0)
            total_density += float(model.graph_density) * targets.size(0)

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    subject_ids = np.asarray(subject_ids)

    window_metrics = regression_metrics(predictions, labels)
    aggregated_predictions, aggregated_labels, unique_subjects = aggregate_predictions_by_subject(
        predictions, labels, subject_ids, reducer="median"
    )
    subject_metrics = regression_metrics(aggregated_predictions, aggregated_labels)

    metrics = {}
    for key, value in window_metrics.items():
        metrics[f"window_{key}"] = value
    for key, value in subject_metrics.items():
        metrics[f"subject_{key}"] = value
    metrics["mae"] = window_metrics["mae"]
    metrics["aami_pass"] = window_metrics["aami_pass"]
    metrics["subjects"] = int(len(unique_subjects))
    metrics["graph_density"] = total_density / max(1, total_count)
    return metrics


def format_metric_block(metrics, prefix):
    block = {
        key: metrics[f"{prefix}_{key}"]
        for key in ("sbp_mae", "sbp_me", "sbp_std", "dbp_mae", "dbp_me", "dbp_std", "mae", "aami_pass")
    }
    return format_metrics(block, prefix=prefix)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    args,
    target_mean,
    target_std,
    demo_mean,
    demo_std,
    channel_mean,
    channel_std,
    scheduler=None,
    scaler=None,
):
    def to_numpy(value):
        if torch.is_tensor(value):
            value = value.detach().cpu()
        return np.asarray(value, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
        "target_mean": to_numpy(target_mean),
        "target_std": to_numpy(target_std),
        "demo_mean": to_numpy(demo_mean),
        "demo_std": to_numpy(demo_std),
        "channel_mean": to_numpy(channel_mean),
        "channel_std": to_numpy(channel_std),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def build_subject_protocol(args):
    cohort_subjects = None
    with h5py.File(args.file, "r") as handle:
        total_records = int(handle["Subset"]["Signals"].shape[2])
    if args.max_records is not None:
        total_records = min(total_records, int(args.max_records))
    all_indices = np.arange(total_records, dtype=np.int64)

    if args.val_split > 0:
        rng = np.random.default_rng(args.seed)
        shuffled = all_indices.copy()
        rng.shuffle(shuffled)
        val_count = int(0.20 * len(shuffled))
        val_indices = np.sort(shuffled[:val_count])
        train_indices = np.sort(shuffled[val_count:])
    else:
        train_indices = all_indices
        val_indices = np.empty((0,), dtype=np.int64)

    test_indices = np.empty((0,), dtype=np.int64)
    return train_indices, val_indices, test_indices, cohort_subjects


def train(args):
    configure_logging()
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    train_indices, val_indices, test_indices, cohort_subjects = build_subject_protocol(args)
    logger.info(
        "protocol_ready train_records=%d val_records=%d test_records=%d split_mode=%s",
        len(train_indices),
        len(val_indices),
        len(test_indices),
        args.split_mode,
    )

    train_set = VitalDBBeatSequenceDataset(
        args.file,
        input_mode=args.input_mode,
        indices=train_indices,
        max_records=args.max_records,
        cache_workers=args.cache_workers,
    )
    channel_mean = train_set.channel_mean
    channel_std = train_set.channel_std
    logger.info(
        "train_cache_ready samples=%d channel_mean=%s channel_std=%s",
        len(train_set),
        np.asarray(channel_mean).tolist() if channel_mean is not None else None,
        np.asarray(channel_std).tolist() if channel_std is not None else None,
    )

    val_set = (
        VitalDBBeatSequenceDataset(
            args.file,
            input_mode=args.input_mode,
            indices=val_indices,
            max_records=args.max_records,
            channel_mean=channel_mean,
            channel_std=channel_std,
            cache_workers=args.cache_workers,
        )
        if len(val_indices)
        else None
    )
    logger.info("val_cache_ready samples=%d", 0 if val_set is None else len(val_set))

    target_stats = compute_target_stats(train_set)
    demo_stats = compute_demographics_stats(train_set)
    target_mean = torch.as_tensor(target_stats["mean"], dtype=torch.float32, device=device)
    target_std = torch.as_tensor(target_stats["std"], dtype=torch.float32, device=device)
    demo_mean = torch.as_tensor(demo_stats["mean"], dtype=torch.float32, device=device)
    demo_std = torch.as_tensor(demo_stats["std"], dtype=torch.float32, device=device)
    logger.info(
        "stats_ready target_mean=%s target_std=%s demo_mean=%s demo_std=%s",
        np.asarray(target_stats["mean"]).tolist(),
        np.asarray(target_stats["std"]).tolist(),
        np.asarray(demo_stats["mean"]).tolist(),
        np.asarray(demo_stats["std"]).tolist(),
    )

    if args.importance_weighted_sampler:
        logger.info("sampler_build_start type=importance_weighted aami_cal_file=%s", args.aami_cal_file)
        train_sampler = make_importance_weighted_sampler(
            train_set, args.aami_cal_file, IW_MAX_WEIGHT, IW_BANDWIDTH
        )
        logger.info("sampler_ready type=importance_weighted")
    elif args.balanced_sampler:
        logger.info("sampler_build_start type=balanced")
        train_sampler = make_balanced_sampler(train_set)
        logger.info("sampler_ready type=balanced")
    else:
        train_sampler = None
        logger.info("sampler_ready type=shuffle")

    train_loader = make_loader(
        train_set, args.batch_size, train_sampler is None, args.num_workers, sampler=train_sampler
    )
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers) if val_set is not None else None

    spec = input_spec(args.input_mode)
    model = build_model(
        args.model,
        spec["channels"],
        ppg_channel_index=spec["ppg_index"],
        beat_channel_indices=spec["beat_channels"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    warmup_epochs = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=1,
        eta_min=args.lr_min,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.model}_{args.input_mode}"
    if args.split_mode == "loocv":
        run_name = f"{run_name}_loocv_{args.heldout_subject}"
    best_path = output_dir / f"{run_name}_best.pt"
    last_path = output_dir / f"{run_name}_last.pt"
    start_epoch = 1
    best_score = float("inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_path.is_file():
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            best_score = float(best_checkpoint["metrics"].get("window_mae", float("inf")))

    logger.info(
        "model=%s input_mode=%s params=%d train_records=%d val_records=%d test_records=%d "
        "train_samples=%d val_samples=%d cohort_subjects=%d device=%s",
        args.model,
        args.input_mode,
        count_parameters(model),
        len(train_indices),
        len(val_indices),
        len(test_indices),
        len(train_set),
        0 if val_set is None else len(val_set),
        0 if cohort_subjects is None else len(cohort_subjects),
        device,
    )

    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = run_train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            target_mean,
            target_std,
            demo_mean,
            demo_std,
            args.accumulate_steps,
            args.wave_loss_weight,
            args.scalar_loss_weight,
        )
        scheduler.step(epoch)

        if val_loader is not None and (epoch % args.validate_every == 0 or epoch == 1):
            val_metrics = evaluate_loader(model, val_loader, device, target_mean, target_std, demo_mean, demo_std)
            score = val_metrics["window_mae"]
            logger.info(
                "epoch=%d train_mae=%.5f val_window_mae=%.5f graph_density=%.4f lr=%.8f %s %s",
                epoch,
                train_stats["mae"],
                val_metrics["window_mae"],
                val_metrics["graph_density"],
                optimizer.param_groups[0]["lr"],
                format_metric_block(val_metrics, "window"),
                format_metric_block(val_metrics, "subject"),
            )
            if score < best_score:
                best_score = score
                epochs_without_improvement = 0
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    val_metrics,
                    args,
                    target_mean,
                    target_std,
                    demo_mean,
                    demo_std,
                    channel_mean,
                    channel_std,
                    scheduler=scheduler,
                    scaler=scaler,
                )
            elif epoch > max(warmup_epochs, 100):
                epochs_without_improvement += 1
                if args.patience > 0 and epochs_without_improvement >= args.patience:
                    break
        else:
            logger.info(
                "epoch=%d train_mae=%.5f graph_density=%.4f lr=%.8f",
                epoch,
                train_stats["mae"],
                train_stats["graph_density"],
                optimizer.param_groups[0]["lr"],
            )

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            {},
            args,
            target_mean,
            target_std,
            demo_mean,
            demo_std,
            channel_mean,
            channel_std,
            scheduler=scheduler,
            scaler=scaler,
        )

    if val_loader is not None:
        logger.info("best_checkpoint=%s", best_path)
    logger.info("last_checkpoint=%s", last_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=TRAIN_MAT)
    parser.add_argument("--model", choices=MODEL_TYPES, default="bpm_graph_unet")
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="ecg_ppg_d1_d2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--validate-every", type=int, default=VALIDATE_EVERY)
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--record-split", action="store_true")
    parser.add_argument("--split-mode", choices=("subject", "loocv"), default="subject")
    parser.add_argument("--heldout-subject", default=None)
    parser.add_argument("--use-cignn-cohort", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--cache-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--importance-weighted-sampler", action="store_true")
    parser.add_argument("--aami-cal-file", type=Path, default=AAMI_CAL_MAT)
    parser.add_argument("--wave-loss-weight", type=float, default=1.0)
    parser.add_argument("--scalar-loss-weight", type=float, default=1.0)
    parser.add_argument("--accumulate-steps", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    args.file = Path(args.file)
    args.output_dir = Path(args.output_dir)
    args.aami_cal_file = Path(args.aami_cal_file)
    train(args)


if __name__ == "__main__":
    main()
