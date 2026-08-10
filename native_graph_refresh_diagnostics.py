import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import VitalDBBeatSequenceDataset
from .model import batch_graph_density, build_adjacency, build_model, splat_graph_context
from .paths import CACHE_DIR
from .train import collate_batch, input_spec


def normalize_demographics(demographics, demo_mean, demo_std):
    demographics = torch.where(torch.isfinite(demographics), demographics, demo_mean)
    return (demographics - demo_mean) / demo_std


def denormalize_targets(targets, target_mean, target_std):
    return targets * target_std + target_mean


def make_loader(dataset, batch_size, num_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": collate_batch,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
        kwargs["timeout"] = 120
    return DataLoader(dataset, **kwargs)


def pearson_r(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x_std = x.std()
    y_std = y.std()
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def window_clustering(adjacency, beat_mask):
    adjacency = (adjacency > 0).to(torch.float32)
    batch_size = adjacency.size(0)
    clustering = np.zeros(batch_size, dtype=np.float32)
    for batch_index in range(batch_size):
        valid = beat_mask[batch_index].detach().cpu().numpy().astype(bool)
        node_count = int(valid.sum())
        if node_count < 3:
            clustering[batch_index] = 0.0
            continue
        sub_adj = adjacency[batch_index][valid][:, valid].detach().cpu().numpy()
        np.fill_diagonal(sub_adj, 0.0)
        coeffs = []
        for node in range(node_count):
            neighbors = np.flatnonzero(sub_adj[node] > 0)
            degree = neighbors.size
            if degree < 2:
                coeffs.append(0.0)
                continue
            neighbor_graph = sub_adj[np.ix_(neighbors, neighbors)]
            edges = neighbor_graph.sum() / 2.0
            coeffs.append(float((2.0 * edges) / (degree * (degree - 1))))
        clustering[batch_index] = float(np.mean(coeffs)) if coeffs else 0.0
    return clustering


def extract_graph_state(model, batch, device, demo_mean, demo_std):
    inputs = batch["inputs"].to(device, non_blocking=True)
    demographics = normalize_demographics(
        batch["demographics"].to(device, non_blocking=True),
        demo_mean,
        demo_std,
    )
    beat_features = batch["beat_features"].to(device, non_blocking=True)
    beat_inputs = batch["beat_inputs"].to(device, non_blocking=True)
    beat_mask = batch["beat_mask"].to(device, non_blocking=True)

    bottleneck, skips = model.encode(inputs)

    beat_inputs_selected = beat_inputs[:, :, list(model.beat_channel_indices), :]
    batch_size, max_beats, channels, beat_len = beat_inputs_selected.shape
    flat_beats = beat_inputs_selected.reshape(batch_size * max_beats, channels, beat_len)
    beat_embeddings = model.beat_encoder(flat_beats).view(batch_size, max_beats, model.embed_dim)
    projected_features = model.beat_feature_projection(beat_features)
    beat_embeddings = beat_embeddings + projected_features * beat_mask.unsqueeze(-1).to(projected_features.dtype)

    adjacency = build_adjacency(beat_embeddings, beat_mask, model.morph_threshold)
    beat_tokens = model.beat_graph(beat_embeddings, adjacency)

    pool_logits = model.graph_pool_attention(beat_tokens).squeeze(-1)
    pool_logits = pool_logits.masked_fill(~beat_mask, -1e4)
    pool_weights = torch.softmax(pool_logits, dim=1)
    graph_pool = torch.sum(beat_tokens * pool_weights.unsqueeze(-1), dim=1)

    graph_context = splat_graph_context(
        beat_tokens,
        beat_mask,
        batch["beat_boundaries"],
        window_len=inputs.size(-1),
    )
    decoded = model.decode(bottleneck, skips, graph_context)
    encoder_pool = bottleneck.mean(dim=-1)
    decoder_pool = torch.cat([decoded.mean(dim=-1), decoded.max(dim=-1).values], dim=1)
    demographics_embed = model.demographics_encoder(demographics)
    global_features = torch.cat([encoder_pool, decoder_pool, graph_pool, demographics_embed], dim=1)
    blood_pressure = model.blood_pressure_head(global_features)

    return {
        "inputs": inputs,
        "demographics_embed": demographics_embed,
        "bottleneck": bottleneck,
        "skips": skips,
        "adjacency": adjacency,
        "beat_mask": beat_mask,
        "graph_context": graph_context,
        "graph_pool": graph_pool,
        "blood_pressure": blood_pressure,
        "beat_tokens": beat_tokens,
        "pool_weights": pool_weights,
        "decoder_pool": decoder_pool,
        "encoder_pool": encoder_pool,
        "decoded": decoded,
    }


def predict_variant(model, state, variant):
    if variant == "full":
        graph_context = state["graph_context"]
        graph_pool = state["graph_pool"]
    elif variant == "zero_graph_pool":
        graph_context = state["graph_context"]
        graph_pool = torch.zeros_like(state["graph_pool"])
    elif variant == "zero_graph_context":
        graph_context = torch.zeros_like(state["graph_context"])
        graph_pool = state["graph_pool"]
    elif variant == "zero_graph_branch":
        graph_context = torch.zeros_like(state["graph_context"])
        graph_pool = torch.zeros_like(state["graph_pool"])
    else:
        raise KeyError(variant)

    decoded = model.decode(state["bottleneck"], state["skips"], graph_context)
    decoder_pool = torch.cat([decoded.mean(dim=-1), decoded.max(dim=-1).values], dim=1)
    global_features = torch.cat(
        [state["encoder_pool"], decoder_pool, graph_pool, state["demographics_embed"]],
        dim=1,
    )
    return model.blood_pressure_head(global_features)


def beat_hemodynamics(abp_wave, boundaries):
    sbp = []
    dbp = []
    pp = []
    map_values = []
    for start, end in boundaries:
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        beat = np.asarray(abp_wave[start:end], dtype=np.float32)
        if beat.size == 0:
            continue
        beat_sbp = float(np.max(beat))
        beat_dbp = float(np.min(beat))
        beat_pp = beat_sbp - beat_dbp
        beat_map = float(np.mean(beat))
        sbp.append(beat_sbp)
        dbp.append(beat_dbp)
        pp.append(beat_pp)
        map_values.append(beat_map)
    return {
        "sbp": np.asarray(sbp, dtype=np.float32),
        "dbp": np.asarray(dbp, dtype=np.float32),
        "pp": np.asarray(pp, dtype=np.float32),
        "map": np.asarray(map_values, dtype=np.float32),
    }


def std_and_sdsd(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(values))
    if values.size < 2:
        return std, float("nan")
    diffs = np.diff(values)
    sdsd = float(np.sqrt(np.mean(np.square(diffs))))
    return std, sdsd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    spec = input_spec(train_args["input_mode"])

    dataset = VitalDBBeatSequenceDataset(
        file_path=str(args.file),
        input_mode=train_args["input_mode"],
        channel_mean=checkpoint["channel_mean"],
        channel_std=checkpoint["channel_std"],
        cache_dir=CACHE_DIR,
        cache_workers=args.cache_workers,
    )
    loader = make_loader(dataset, args.batch_size, args.num_workers)

    device = torch.device(args.device)
    model = build_model(
        train_args["model"],
        spec["channels"],
        ppg_channel_index=spec["ppg_index"],
        beat_channel_indices=spec["beat_channels"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    target_mean = torch.tensor(checkpoint["target_mean"], dtype=torch.float32, device=device)
    target_std = torch.tensor(checkpoint["target_std"], dtype=torch.float32, device=device)
    demo_mean = torch.tensor(checkpoint["demo_mean"], dtype=torch.float32, device=device)
    demo_std = torch.tensor(checkpoint["demo_std"], dtype=torch.float32, device=device)

    densities = []
    clusterings = []
    dbp_sdsd = []
    pp_sdsd = []
    map_sdsd = []
    internal_low_dbp_dev = []
    internal_high_dbp_dev = []
    internal_low_pp_dev = []
    internal_high_pp_dev = []
    internal_dbp_low_gt_high = 0
    internal_pp_low_gt_high = 0
    internal_windows = 0

    variant_errors = {
        "full": {"sbp": [], "dbp": [], "mae": []},
        "zero_graph_pool": {"sbp": [], "dbp": [], "mae": []},
        "zero_graph_context": {"sbp": [], "dbp": [], "mae": []},
        "zero_graph_branch": {"sbp": [], "dbp": [], "mae": []},
    }

    with torch.no_grad():
        for batch in loader:
            state = extract_graph_state(model, batch, device, demo_mean, demo_std)
            density_batch = batch_graph_density(state["adjacency"], state["beat_mask"]).detach().cpu().numpy()
            clustering_batch = window_clustering(state["adjacency"], state["beat_mask"])
            densities.extend(density_batch.tolist())
            clusterings.extend(clustering_batch.tolist())

            targets = batch["blood_pressure"].to(device, non_blocking=True)
            for variant in variant_errors:
                predicted = predict_variant(model, state, variant)
                predicted = denormalize_targets(predicted, target_mean, target_std)
                abs_err = torch.abs(predicted - targets).detach().cpu().numpy()
                variant_errors[variant]["sbp"].extend(abs_err[:, 0].tolist())
                variant_errors[variant]["dbp"].extend(abs_err[:, 1].tolist())
                variant_errors[variant]["mae"].extend(abs_err.mean(axis=1).tolist())

            adjacency_np = state["adjacency"].detach().cpu().numpy()
            beat_mask_np = state["beat_mask"].detach().cpu().numpy()
            abp_waves_np = batch["abp_waveform"].cpu().numpy()

            for sample_index, boundaries in enumerate(batch["beat_boundaries"]):
                valid_count = int(beat_mask_np[sample_index].sum())
                boundaries = list(boundaries[:valid_count])
                hemo = beat_hemodynamics(abp_waves_np[sample_index], boundaries)
                dbp_std_value, dbp_sdsd_value = std_and_sdsd(hemo["dbp"])
                pp_std_value, pp_sdsd_value = std_and_sdsd(hemo["pp"])
                map_std_value, map_sdsd_value = std_and_sdsd(hemo["map"])
                dbp_sdsd.append(dbp_sdsd_value)
                pp_sdsd.append(pp_sdsd_value)
                map_sdsd.append(map_sdsd_value)

                if valid_count < 4:
                    continue
                degrees = adjacency_np[sample_index, :valid_count, :valid_count].sum(axis=-1)
                internal_slice = slice(1, valid_count - 1)
                internal_degrees = degrees[internal_slice]
                if internal_degrees.size == 0:
                    continue
                internal_dbp = hemo["dbp"][1:-1]
                internal_pp = hemo["pp"][1:-1]
                if internal_dbp.size == 0 or internal_pp.size == 0:
                    continue
                low_pos = int(np.argmin(internal_degrees))
                high_pos = int(np.argmax(internal_degrees))
                median_dbp = float(np.median(hemo["dbp"]))
                median_pp = float(np.median(hemo["pp"]))
                low_dbp_dev = float(abs(internal_dbp[low_pos] - median_dbp))
                high_dbp_dev = float(abs(internal_dbp[high_pos] - median_dbp))
                low_pp_dev = float(abs(internal_pp[low_pos] - median_pp))
                high_pp_dev = float(abs(internal_pp[high_pos] - median_pp))
                internal_low_dbp_dev.append(low_dbp_dev)
                internal_high_dbp_dev.append(high_dbp_dev)
                internal_low_pp_dev.append(low_pp_dev)
                internal_high_pp_dev.append(high_pp_dev)
                internal_dbp_low_gt_high += int(low_dbp_dev > high_dbp_dev)
                internal_pp_low_gt_high += int(low_pp_dev > high_pp_dev)
                internal_windows += 1

    densities_np = np.asarray(densities, dtype=np.float32)
    hist_counts, hist_edges = np.histogram(densities_np, bins=20)

    density_vs_dbp = pearson_r(densities, dbp_sdsd)
    clustering_vs_dbp = pearson_r(clusterings, dbp_sdsd)
    density_vs_pp = pearson_r(densities, pp_sdsd)
    density_vs_map = pearson_r(densities, map_sdsd)

    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "dataset": str(args.file),
        "windows_analyzed": int(len(densities)),
        "run1_density_distribution": {
            "mean": float(np.mean(densities_np)),
            "std": float(np.std(densities_np)),
            "min": float(np.min(densities_np)),
            "max": float(np.max(densities_np)),
            "histogram_counts": hist_counts.tolist(),
            "histogram_edges": hist_edges.tolist(),
        },
        "run2_hemodynamic_stability": {
            "density_vs_dbp_sdsd_r": density_vs_dbp,
            "clustering_vs_dbp_sdsd_r": clustering_vs_dbp,
            "density_vs_pp_sdsd_r": density_vs_pp,
            "density_vs_map_sdsd_r": density_vs_map,
            "clustering_mean": float(np.mean(clusterings)),
            "clustering_std": float(np.std(clusterings)),
        },
        "run3_internal_outliers": {
            "windows_used": int(internal_windows),
            "low_degree_internal_dbp_dev_mean": float(np.mean(internal_low_dbp_dev)) if internal_low_dbp_dev else float("nan"),
            "high_degree_internal_dbp_dev_mean": float(np.mean(internal_high_dbp_dev)) if internal_high_dbp_dev else float("nan"),
            "low_degree_internal_pp_dev_mean": float(np.mean(internal_low_pp_dev)) if internal_low_pp_dev else float("nan"),
            "high_degree_internal_pp_dev_mean": float(np.mean(internal_high_pp_dev)) if internal_high_pp_dev else float("nan"),
            "dbp_low_gt_high_rate": float(internal_dbp_low_gt_high / max(1, internal_windows)),
            "pp_low_gt_high_rate": float(internal_pp_low_gt_high / max(1, internal_windows)),
        },
        "run4_run5_ablations": {},
    }

    for variant, metrics in variant_errors.items():
        sbp = np.asarray(metrics["sbp"], dtype=np.float32)
        dbp = np.asarray(metrics["dbp"], dtype=np.float32)
        mae = np.asarray(metrics["mae"], dtype=np.float32)
        results["run4_run5_ablations"][variant] = {
            "window_sbp_mae": float(np.mean(sbp)),
            "window_dbp_mae": float(np.mean(dbp)),
            "window_mae": float(np.mean(mae)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
