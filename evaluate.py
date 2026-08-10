import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import BATCH_SIZE, NUM_WORKERS
from .dataset import VitalDBBeatSequenceDataset
from .metrics import aggregate_predictions_by_subject, format_metrics, regression_metrics
from .model import build_model, count_parameters
from .train import collate_batch, forward_model, input_spec, normalize_demographics

logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
    target_mean,
    target_std,
    demo_mean,
    demo_std,
):
    model.eval()
    predictions = []
    targets = []
    subject_ids = []
    demo_mean = torch.as_tensor(demo_mean, dtype=torch.float32, device=device)
    demo_std = torch.as_tensor(demo_std, dtype=torch.float32, device=device)
    for batch in loader:
        demographics = normalize_demographics(
            batch["demographics"].to(device, non_blocking=True), demo_mean, demo_std
        )
        predicted_bp, _ = forward_model(model, batch, device, demographics)
        mean = torch.as_tensor(target_mean, dtype=predicted_bp.dtype, device=predicted_bp.device)
        std = torch.as_tensor(target_std, dtype=predicted_bp.dtype, device=predicted_bp.device)
        predicted_bp = predicted_bp * std + mean
        predictions.append(predicted_bp.detach().cpu().numpy())
        targets.append(batch["blood_pressure"].numpy())
        subject_ids.extend(batch["subject_id"])
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.asarray(subject_ids),
    )


def apply_subject_calibration(predictions, targets, subject_ids, calibration_samples):
    calibrated = np.asarray(predictions, dtype=np.float32).copy()
    targets = np.asarray(targets, dtype=np.float32)
    subject_ids = np.asarray(subject_ids)
    evaluation_mask = np.ones(len(calibrated), dtype=bool)

    for subject in np.unique(subject_ids):
        subject_mask = np.flatnonzero(subject_ids == subject)
        calibration_index = subject_mask[: min(int(calibration_samples), subject_mask.size)]
        bias = np.mean(calibrated[calibration_index] - targets[calibration_index], axis=0, keepdims=True)
        calibrated[subject_mask] -= bias
        evaluation_mask[calibration_index] = False

    return calibrated, evaluation_mask


def apply_aami_calibration(
    test_predictions,
    test_targets,
    test_subject_ids,
    cal_predictions,
    cal_targets,
    cal_subject_ids,
    calibration_samples,
):
    calibrated = np.asarray(test_predictions, dtype=np.float32).copy()
    cal_predictions = np.asarray(cal_predictions, dtype=np.float32)
    cal_targets = np.asarray(cal_targets, dtype=np.float32)
    test_subject_ids = np.asarray(test_subject_ids)
    cal_subject_ids = np.asarray(cal_subject_ids)
    matched_biases = []
    matched_subjects = 0
    unmatched_subjects = 0

    for subject in np.unique(test_subject_ids):
        cal_mask = np.flatnonzero(cal_subject_ids == subject)
        if cal_mask.size == 0:
            unmatched_subjects += 1
            continue
        matched_subjects += 1
        calibration_index = cal_mask[: min(int(calibration_samples), cal_mask.size)]
        bias = np.mean(
            cal_predictions[calibration_index] - cal_targets[calibration_index],
            axis=0,
            keepdims=True,
        )
        matched_biases.append(bias)
        test_mask = np.flatnonzero(test_subject_ids == subject)
        calibrated[test_mask] -= bias

    if matched_biases:
        population_bias = np.mean(np.concatenate(matched_biases, axis=0), axis=0, keepdims=True)
        for subject in np.unique(test_subject_ids):
            cal_mask = np.flatnonzero(cal_subject_ids == subject)
            if cal_mask.size != 0:
                continue
            test_mask = np.flatnonzero(test_subject_ids == subject)
            calibrated[test_mask] -= population_bias
    else:
        population_bias = None

    logger.info(
        "aami_calibration matched_subjects=%d unmatched_subjects=%d calibration_samples=%d population_bias=%s",
        matched_subjects,
        unmatched_subjects,
        int(calibration_samples),
        None if population_bias is None else np.asarray(population_bias).reshape(-1).tolist(),
    )
    return calibrated


def make_loader(file_path, input_mode, channel_mean, channel_std, cache_workers=1):
    dataset = VitalDBBeatSequenceDataset(
        file_path,
        input_mode=input_mode,
        channel_mean=channel_mean,
        channel_std=channel_std,
        cache_workers=cache_workers,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    return dataset, loader


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = ("model", "target_mean", "target_std", "demo_mean", "demo_std", "channel_mean", "channel_std", "args")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint missing keys: {missing}")
    state_dict = checkpoint["model"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key[len("_orig_mod."):]: value for key, value in state_dict.items()}
    return checkpoint, state_dict


def is_legacy_graph_checkpoint(state_dict):
    return "encoder1.se.se.2.weight" not in state_dict


def evaluate(
    checkpoint_path,
    file_path,
    aami_cal_file=None,
    calibration_samples=0,
    output_path=None,
    device="cuda",
):
    configure_logging()
    device = torch.device(device)

    checkpoint, state_dict = load_checkpoint(checkpoint_path, device)
    train_args = checkpoint["args"]
    input_mode = train_args["input_mode"]
    model_type = train_args["model"]

    target_mean = checkpoint["target_mean"]
    target_std = checkpoint["target_std"]
    demo_mean = checkpoint["demo_mean"]
    demo_std = checkpoint["demo_std"]
    channel_mean = checkpoint["channel_mean"]
    channel_std = checkpoint["channel_std"]
    cache_workers = int(train_args.get("cache_workers", 1))

    spec = input_spec(input_mode)
    model = build_model(
        model_type,
        spec["channels"],
        ppg_channel_index=spec["ppg_index"],
        beat_channel_indices=spec["beat_channels"],
        legacy=is_legacy_graph_checkpoint(state_dict),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)

    test_dataset, test_loader = make_loader(file_path, input_mode, channel_mean, channel_std, cache_workers=cache_workers)
    test_predictions, test_targets, test_subject_ids = collect_predictions(
        model,
        test_loader,
        device,
        target_mean,
        target_std,
        demo_mean,
        demo_std,
    )

    if aami_cal_file is not None:
        _, cal_loader = make_loader(aami_cal_file, input_mode, channel_mean, channel_std, cache_workers=cache_workers)
        cal_predictions, cal_targets, cal_subject_ids = collect_predictions(
            model,
            cal_loader,
            device,
            target_mean,
            target_std,
            demo_mean,
            demo_std,
        )
        predictions = apply_aami_calibration(
            test_predictions,
            test_targets,
            test_subject_ids,
            cal_predictions,
            cal_targets,
            cal_subject_ids,
            calibration_samples,
        )
        evaluation_targets = test_targets
        evaluation_subject_ids = test_subject_ids
    elif calibration_samples > 0:
        predictions, evaluation_mask = apply_subject_calibration(
            test_predictions, test_targets, test_subject_ids, calibration_samples
        )
        predictions = predictions[evaluation_mask]
        evaluation_targets = test_targets[evaluation_mask]
        evaluation_subject_ids = test_subject_ids[evaluation_mask]
    else:
        predictions = test_predictions
        evaluation_targets = test_targets
        evaluation_subject_ids = test_subject_ids

    window_metrics = regression_metrics(predictions, evaluation_targets)

    subject_predictions, subject_targets, _ = aggregate_predictions_by_subject(
        predictions, evaluation_targets, evaluation_subject_ids, reducer="median"
    )
    subject_metrics = regression_metrics(subject_predictions, subject_targets)

    logger.info(
        "file=%s samples=%d params=%d %s %s",
        file_path,
        len(test_dataset),
        count_parameters(model),
        format_metrics(window_metrics, prefix="window"),
        format_metrics(subject_metrics, prefix="subject"),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            window_predictions=predictions,
            window_targets=evaluation_targets,
            subject_predictions=subject_predictions,
            subject_targets=subject_targets,
            window_metrics=np.asarray(window_metrics, dtype=object),
            subject_metrics=np.asarray(subject_metrics, dtype=object),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--aami-cal-file", type=Path, default=None)
    parser.add_argument("--calibration-samples", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.file,
        aami_cal_file=args.aami_cal_file,
        calibration_samples=args.calibration_samples,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
