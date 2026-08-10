import numpy as np


def regression_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    errors = predictions - targets
    abs_errors = np.abs(errors)
    metrics = {}
    for idx, name in enumerate(("sbp", "dbp")):
        metrics[f"{name}_mae"] = float(np.mean(abs_errors[:, idx]))
        metrics[f"{name}_std"] = float(np.std(errors[:, idx]))
        metrics[f"{name}_me"] = float(np.mean(errors[:, idx]))
    metrics["mae"] = float(np.mean(abs_errors))
    metrics["aami_pass"] = bool(
        abs(metrics["sbp_me"]) <= 5.0
        and abs(metrics["dbp_me"]) <= 5.0
        and metrics["sbp_std"] <= 8.0
        and metrics["dbp_std"] <= 8.0
    )
    return metrics


def aggregate_predictions_by_subject(predictions, targets, subject_ids, reducer="median"):
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    subject_ids = np.asarray(subject_ids)
    unique_subjects = np.unique(subject_ids)
    agg_preds = []
    agg_targets = []
    reduce_fn = np.median if reducer == "median" else np.mean
    for subject_id in unique_subjects:
        mask = subject_ids == subject_id
        agg_preds.append(reduce_fn(predictions[mask], axis=0))
        agg_targets.append(reduce_fn(targets[mask], axis=0))
    return np.asarray(agg_preds, dtype=np.float32), np.asarray(agg_targets, dtype=np.float32), unique_subjects


def format_metrics(metrics, prefix=""):
    parts = []
    for key in ("sbp_mae", "sbp_me", "sbp_std", "dbp_mae", "dbp_me", "dbp_std", "mae"):
        name = f"{prefix}_{key}" if prefix else key
        parts.append(f"{name} {metrics[key]:.4f}")
    parts.append(f"{prefix + '_' if prefix else ''}aami_pass {int(metrics['aami_pass'])}")
    return " ".join(parts)
