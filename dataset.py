from pathlib import Path
import hashlib
import multiprocessing as mp
import pickle
from functools import lru_cache

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .paths import CACHE_DIR
from .config import (
    ABP_CHANNEL,
    CACHE_VERSION,
    COHORT_HYPERTENSIVE_SUBJECTS,
    COHORT_NORMOTENSIVE_SUBJECTS,
    ECG_CHANNEL,
    FS,
    HYPERTENSION_DBP_THRESHOLD,
    HYPERTENSION_SBP_THRESHOLD,
    MAX_DBP,
    MAX_DBP_STD,
    MAX_PULSE_PRESSURE,
    MAX_SBP,
    MAX_SBP_STD,
    MIN_DBP,
    MIN_PPG_RANGE,
    MIN_PULSE_PRESSURE,
    MIN_SBP,
    PPG_CHANNEL,
    RESAMPLED_LENGTH,
)
from .preprocess import (
    compute_beat_rr_ptt_features,
    detect_abp_feet,
    detect_r_peaks,
    ecg_based_beat_boundaries,
    filter_abp,
    filter_ppg,
)


_CACHE_WORKER_HANDLE = None
_CACHE_WORKER_SIGNALS = None
_CACHE_WORKER_SBP = None
_CACHE_WORKER_DBP = None
_CACHE_WORKER_CONFIG = None


def input_channel_count(input_mode):
    mapping = {
        "ppg_d1_d2": 3,
        "ecg_ppg_d1_d2": 4,
    }
    return mapping[input_mode]


def _init_cache_worker(file_path, ecg_channel, ppg_channel, abp_channel, input_mode):
    global _CACHE_WORKER_HANDLE, _CACHE_WORKER_SIGNALS, _CACHE_WORKER_SBP, _CACHE_WORKER_DBP, _CACHE_WORKER_CONFIG
    _CACHE_WORKER_HANDLE = h5py.File(file_path, "r")
    subset = _CACHE_WORKER_HANDLE["Subset"]
    _CACHE_WORKER_SIGNALS = subset["Signals"]
    _CACHE_WORKER_SBP = np.asarray(subset["SBP"][0], dtype=np.float32)
    _CACHE_WORKER_DBP = np.asarray(subset["DBP"][0], dtype=np.float32)
    _CACHE_WORKER_CONFIG = {
        "ecg_channel": int(ecg_channel),
        "ppg_channel": int(ppg_channel),
        "abp_channel": int(abp_channel),
        "input_mode": input_mode,
    }


def _process_native_record(record_idx):
    record_idx = int(record_idx)
    ecg = np.asarray(
        _CACHE_WORKER_SIGNALS[:RESAMPLED_LENGTH, _CACHE_WORKER_CONFIG["ecg_channel"], record_idx],
        dtype=np.float32,
    )
    ppg = np.asarray(
        _CACHE_WORKER_SIGNALS[:RESAMPLED_LENGTH, _CACHE_WORKER_CONFIG["ppg_channel"], record_idx],
        dtype=np.float32,
    )
    abp = np.asarray(
        _CACHE_WORKER_SIGNALS[:RESAMPLED_LENGTH, _CACHE_WORKER_CONFIG["abp_channel"], record_idx],
        dtype=np.float32,
    )
    sbp = float(_CACHE_WORKER_SBP[record_idx])
    dbp = float(_CACHE_WORKER_DBP[record_idx])

    dppg = np.gradient(ppg).astype(np.float32) * FS
    ddppg = np.gradient(dppg).astype(np.float32) * FS
    segment_stack = (
        np.stack([ppg, dppg, ddppg], axis=0)
        if _CACHE_WORKER_CONFIG["input_mode"] == "ppg_d1_d2"
        else np.stack([ecg, ppg, dppg, ddppg], axis=0)
    ).astype(np.float32)

    r_peaks = detect_r_peaks(ecg)
    boundaries = ecg_based_beat_boundaries(r_peaks, window_len=RESAMPLED_LENGTH)
    beat_feats = compute_beat_rr_ptt_features(r_peaks, ppg, FS)
    target = np.asarray([sbp, dbp], dtype=np.float32)
    return record_idx, segment_stack, target, boundaries, beat_feats, abp


def _read_matlab_string(file_handle, ref):
    data = file_handle[ref][()]
    data = np.asarray(data).squeeze()
    if data.dtype.kind in {"u", "i"}:
        return "".join(chr(int(x)) for x in data.flat if int(x) != 0)
    return str(data)


@lru_cache(maxsize=64)
def _read_subjects_cached(file_path, max_records):
    with h5py.File(file_path, "r") as handle:
        subset = handle["Subset"]
        if "Subject" not in subset:
            return None
        refs = np.asarray(subset["Subject"][0])
        if max_records is not None:
            refs = refs[: int(max_records)]
        return np.asarray([_read_matlab_string(handle, ref) for ref in refs])


def read_subjects(file_path, max_records=None):
    max_records = None if max_records is None else int(max_records)
    subjects = _read_subjects_cached(str(file_path), max_records)
    if subjects is None:
        return None
    return subjects.copy()


@lru_cache(maxsize=64)
def _read_demographics_cached(file_path, max_records):
    with h5py.File(file_path, "r") as handle:
        subset = handle["Subset"]
        total = int(subset["Signals"].shape[2])
        if max_records is not None:
            total = min(total, int(max_records))

        age = np.asarray(subset["Age"][0, :total], dtype=np.float32)
        height = np.asarray(subset["Height"][0, :total], dtype=np.float32)
        weight = np.asarray(subset["Weight"][0, :total], dtype=np.float32)
        bmi = np.asarray(subset["BMI"][0, :total], dtype=np.float32)

        gender_refs = np.asarray(subset["Gender"][0, :total])
        gender = []
        for ref in gender_refs:
            value = _read_matlab_string(handle, ref).strip().lower()
            if value in {"m", "male"}:
                gender.append(1.0)
            elif value in {"f", "female"}:
                gender.append(0.0)
            else:
                gender.append(np.nan)
        gender = np.asarray(gender, dtype=np.float32)
    return np.stack([age, gender, height, weight, bmi], axis=1).astype(np.float32)


def read_demographics(file_path, max_records=None):
    max_records = None if max_records is None else int(max_records)
    return _read_demographics_cached(str(file_path), max_records).copy()


def valid_segment_targets(ppg_segment, abp_segment, cycle_peaks=None, start=None, end=None):
    if not np.all(np.isfinite(ppg_segment)) or not np.all(np.isfinite(abp_segment)):
        return None
    ppg_clean = filter_ppg(ppg_segment)
    if float(np.percentile(ppg_clean, 95.0) - np.percentile(ppg_clean, 5.0)) < MIN_PPG_RANGE:
        return None

    abp_clean = filter_abp(abp_segment)
    feet = detect_abp_feet(abp_clean)
    if len(feet) < 2:
        return None

    sbp_per_beat = []
    dbp_per_beat = []
    for foot_start, foot_end in zip(feet[:-1], feet[1:]):
        beat = abp_clean[int(foot_start):int(foot_end)]
        if beat.size <= 5:
            continue
        sbp_per_beat.append(float(np.max(beat)))
        dbp_per_beat.append(float(beat[0]))
    if not sbp_per_beat or not dbp_per_beat:
        return None

    sbp_arr = np.asarray(sbp_per_beat, dtype=np.float32)
    dbp_arr = np.asarray(dbp_per_beat, dtype=np.float32)
    pulse_pressure = sbp_arr - dbp_arr
    if float(np.median(pulse_pressure)) < MIN_PULSE_PRESSURE or float(np.median(pulse_pressure)) > MAX_PULSE_PRESSURE:
        return None
    if float(np.std(sbp_arr)) > MAX_SBP_STD or float(np.std(dbp_arr)) > MAX_DBP_STD:
        return None

    sbp = float(np.median(sbp_arr))
    dbp = float(np.median(dbp_arr))
    if sbp <= dbp:
        return None
    if sbp < MIN_SBP or sbp > MAX_SBP or dbp < MIN_DBP or dbp > MAX_DBP:
        return None
    return sbp, dbp


def load_subject_metadata(file_path, max_records=None):
    subjects = read_subjects(file_path, max_records=max_records)
    if subjects is None:
        return None
    file_path = str(file_path)
    max_records = None if max_records is None else int(max_records)
    key = (file_path, max_records)
    cached = _load_subject_targets_cached(*key)
    if cached is None:
        return None
    cached_subjects, cached_sbp, cached_dbp = cached
    if len(cached_subjects) != len(subjects):
        return None
    return cached_subjects.copy(), cached_sbp.copy(), cached_dbp.copy()


@lru_cache(maxsize=64)
def _load_subject_targets_cached(file_path, max_records):
    with h5py.File(file_path, "r") as handle:
        subset = handle["Subset"]
        total = int(subset["SBP"].shape[1])
        if max_records is not None:
            total = min(total, int(max_records))
        sbp = np.asarray(subset["SBP"][0, :total], dtype=np.float32)
        dbp = np.asarray(subset["DBP"][0, :total], dtype=np.float32)
    subjects = _read_subjects_cached(file_path, max_records)
    if subjects is None:
        return None
    return subjects, sbp, dbp


def subject_level_bp(file_path, max_records=None):
    metadata = load_subject_metadata(file_path, max_records=max_records)
    if metadata is None:
        return None
    subjects, sbp, dbp = metadata
    summary = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        summary.append((subject, float(np.mean(sbp[mask])), float(np.mean(dbp[mask]))))
    return summary


def split_subjects_by_bp(file_path, seed=42, max_records=None):
    summary = subject_level_bp(file_path, max_records=max_records)
    if summary is None:
        return None, None

    normotensive = []
    hypertensive = []
    for subject, sbp, dbp in summary:
        if sbp >= HYPERTENSION_SBP_THRESHOLD or dbp >= HYPERTENSION_DBP_THRESHOLD:
            hypertensive.append(subject)
        else:
            normotensive.append(subject)

    rng = np.random.default_rng(seed)
    normotensive = np.asarray(normotensive, dtype=str)
    hypertensive = np.asarray(hypertensive, dtype=str)
    rng.shuffle(normotensive)
    rng.shuffle(hypertensive)
    return normotensive, hypertensive


def select_cignn_subjects(file_path, seed=42, max_records=None):
    normotensive, hypertensive = split_subjects_by_bp(file_path, seed=seed, max_records=max_records)
    if normotensive is None or hypertensive is None:
        return None

    selected = np.concatenate(
        [
            normotensive[:COHORT_NORMOTENSIVE_SUBJECTS],
            hypertensive[:COHORT_HYPERTENSIVE_SUBJECTS],
        ]
    )
    return np.sort(selected.astype(str))


def select_records_by_subjects(file_path, subject_ids, max_records=None):
    subjects = read_subjects(file_path, max_records=max_records)
    if subjects is None:
        return None
    wanted = set(np.asarray(subject_ids).astype(str).tolist())
    return np.flatnonzero(np.asarray([subject in wanted for subject in subjects], dtype=bool)).astype(np.int64)


def split_subjects_loocv(file_path, heldout_subject, cohort_subjects=None, max_records=None):
    subjects = read_subjects(file_path, max_records=max_records)
    if subjects is None:
        raise ValueError("Subject metadata is required for LOOCV")

    if cohort_subjects is not None:
        allowed = set(np.asarray(cohort_subjects).astype(str).tolist())
        allowed_mask = np.asarray([subject in allowed for subject in subjects], dtype=bool)
    else:
        allowed_mask = np.ones_like(subjects, dtype=bool)

    heldout_mask = np.asarray([subject == heldout_subject for subject in subjects], dtype=bool)
    train_indices = np.flatnonzero(allowed_mask & ~heldout_mask).astype(np.int64)
    test_indices = np.flatnonzero(allowed_mask & heldout_mask).astype(np.int64)
    return train_indices, test_indices


def split_records(file_path, train_fraction=0.9, seed=42, max_records=None, by_subject=True, cache_dir=None, allowed_subjects=None):
    with h5py.File(file_path, "r") as handle:
        total = int(handle["Subset"]["Signals"].shape[2])
    if max_records is not None:
        total = min(total, int(max_records))
    all_indices = np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)

    if by_subject:
        subjects = read_subjects(file_path, max_records=total)
        if subjects is not None:
            if allowed_subjects is not None:
                allowed = set(np.asarray(allowed_subjects).astype(str).tolist())
                subject_mask = np.asarray([subject in allowed for subject in subjects], dtype=bool)
                subjects = subjects[subject_mask]
                all_indices = all_indices[subject_mask]
            unique_subjects = np.unique(subjects)
            rng.shuffle(unique_subjects)
            train_subject_count = int(round(train_fraction * unique_subjects.size))
            train_subjects = set(unique_subjects[:train_subject_count])
            train_mask = np.asarray([subject in train_subjects for subject in subjects], dtype=bool)
            return all_indices[train_mask], all_indices[~train_mask]

    if allowed_subjects is not None:
        selected = select_records_by_subjects(file_path, allowed_subjects, max_records=total)
        all_indices = selected if selected is not None else all_indices
    shuffled = all_indices.copy()
    rng.shuffle(shuffled)
    train_count = int(round(train_fraction * len(shuffled)))
    return shuffled[:train_count], shuffled[train_count:]


def split_records_stratified_by_bp(file_path, train_fraction=0.9, seed=42, max_records=None, allowed_subjects=None):
    subjects = read_subjects(file_path, max_records=max_records)
    if subjects is None:
        raise ValueError("Subject metadata is required for stratified subject split")

    normotensive, hypertensive = split_subjects_by_bp(file_path, seed=seed, max_records=max_records)
    if normotensive is None or hypertensive is None:
        raise ValueError("Unable to derive subject-level BP strata")

    if allowed_subjects is not None:
        allowed = set(np.asarray(allowed_subjects).astype(str).tolist())
        normotensive = np.asarray([subject for subject in normotensive if subject in allowed], dtype=str)
        hypertensive = np.asarray([subject for subject in hypertensive if subject in allowed], dtype=str)

    rng = np.random.default_rng(seed)
    rng.shuffle(normotensive)
    rng.shuffle(hypertensive)

    def split_group(group):
        if len(group) <= 1:
            return group, group[:0]
        cut = int(round(train_fraction * len(group)))
        cut = min(max(cut, 1), len(group) - 1)
        return group[:cut], group[cut:]

    train_norm, val_norm = split_group(normotensive)
    train_hyp, val_hyp = split_group(hypertensive)

    train_subjects = set(np.concatenate([train_norm, train_hyp]).astype(str).tolist())
    val_subjects = set(np.concatenate([val_norm, val_hyp]).astype(str).tolist())

    all_indices = np.arange(len(subjects), dtype=np.int64)
    train_mask = np.asarray([subject in train_subjects for subject in subjects], dtype=bool)
    val_mask = np.asarray([subject in val_subjects for subject in subjects], dtype=bool)
    return all_indices[train_mask], all_indices[val_mask]


class VitalDBBeatSequenceDataset(Dataset):
    def __init__(
        self,
        file_path,
        input_mode="ecg_ppg_d1_d2",
        indices=None,
        max_records=None,
        ecg_channel=ECG_CHANNEL,
        ppg_channel=PPG_CHANNEL,
        abp_channel=ABP_CHANNEL,
        cache_dir=None,
        channel_mean=None,
        channel_std=None,
        cache_workers=1,
    ):
        self.file_path = str(file_path)
        self.input_mode = input_mode
        self.ecg_channel = ecg_channel
        self.ppg_channel = ppg_channel
        self.abp_channel = abp_channel
        if cache_dir is None:
            cache_dir = CACHE_DIR
        self.cache_dir = Path(cache_dir)
        self.cache_workers = max(1, int(cache_workers))
        self._ext_channel_mean = np.asarray(channel_mean, dtype=np.float32) if channel_mean is not None else None
        self._ext_channel_std = np.asarray(channel_std, dtype=np.float32) if channel_std is not None else None

        with h5py.File(self.file_path, "r") as handle:
            total_records = int(handle["Subset"]["Signals"].shape[2])
        if max_records is not None:
            total_records = min(total_records, int(max_records))

        if indices is None:
            record_indices = np.arange(total_records, dtype=np.int64)
        else:
            record_indices = np.asarray(indices, dtype=np.int64)
            record_indices = record_indices[record_indices < total_records]

        cache_paths = self._cache_paths(record_indices, max_records)

        if cache_paths["meta"].exists():
            self._load_cache(cache_paths)
            return

        self.subjects_by_record = read_subjects(self.file_path, max_records=total_records)
        self.demographics_by_record = read_demographics(self.file_path, max_records=total_records)
        self._build_cache(record_indices, cache_paths)
        self._load_cache(cache_paths)

    @property
    def channel_mean(self):
        if self._ext_channel_mean is not None:
            return self._ext_channel_mean
        return getattr(self, "_pop_channel_mean", None)

    @property
    def channel_std(self):
        if self._ext_channel_std is not None:
            return self._ext_channel_std
        return getattr(self, "_pop_channel_std", None)

    def _build_cache(self, record_indices, cache_paths):
        channels = input_channel_count(self.input_mode)
        pop_mean = None if self._ext_channel_mean is None else np.asarray(self._ext_channel_mean, dtype=np.float32)
        pop_std = None if self._ext_channel_std is None else np.asarray(self._ext_channel_std, dtype=np.float32)

        worker_count = max(1, self.cache_workers)
        chunksize = max(1, min(256, len(record_indices) // max(worker_count * 8, 1)))

        def iterate_processed_records():
            if worker_count == 1:
                _init_cache_worker(self.file_path, self.ecg_channel, self.ppg_channel, self.abp_channel, self.input_mode)
                for record_idx in record_indices:
                    yield _process_native_record(record_idx)
                return
            start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
            ctx = mp.get_context(start_method)
            with ctx.Pool(
                processes=worker_count,
                initializer=_init_cache_worker,
                initargs=(self.file_path, self.ecg_channel, self.ppg_channel, self.abp_channel, self.input_mode),
            ) as pool:
                yield from pool.imap(_process_native_record, record_indices.tolist(), chunksize=chunksize)

        if pop_mean is None or pop_std is None:
            sum_channels = np.zeros(channels, dtype=np.float64)
            sumsq_channels = np.zeros(channels, dtype=np.float64)
            count_channels = 0
            for _, segment_stack, _, _, _, _ in iterate_processed_records():
                sum_channels += segment_stack.sum(axis=1).astype(np.float64)
                sumsq_channels += (segment_stack ** 2).sum(axis=1).astype(np.float64)
                count_channels += segment_stack.shape[1]
            denom = float(max(count_channels, 1))
            pop_mean = (sum_channels / denom).astype(np.float32)
            pop_var = (sumsq_channels / denom) - (pop_mean.astype(np.float64) ** 2)
            pop_std = np.sqrt(np.clip(pop_var, 1e-12, None)).astype(np.float32)
            pop_std = np.maximum(pop_std, 1e-6).astype(np.float32)

        cache_paths["dir"].mkdir(parents=True, exist_ok=True)
        num_samples = len(record_indices)

        inputs = np.lib.format.open_memmap(
            cache_paths["inputs"], mode="w+", dtype=np.float32,
            shape=(num_samples, channels, RESAMPLED_LENGTH),
        )
        targets = np.lib.format.open_memmap(
            cache_paths["targets"], mode="w+", dtype=np.float32, shape=(num_samples, 2),
        )
        record_ids = np.lib.format.open_memmap(
            cache_paths["record_idx"], mode="w+", dtype=np.int64, shape=(num_samples,)
        )
        starts_arr = np.lib.format.open_memmap(
            cache_paths["start"], mode="w+", dtype=np.int64, shape=(num_samples,)
        )
        ends_arr = np.lib.format.open_memmap(
            cache_paths["end"], mode="w+", dtype=np.int64, shape=(num_samples,)
        )
        demos = np.lib.format.open_memmap(
            cache_paths["demo"], mode="w+", dtype=np.float32, shape=(num_samples, 5)
        )
        abp_waves = np.lib.format.open_memmap(
            cache_paths["abp_wave"], mode="w+", dtype=np.float32,
            shape=(num_samples, RESAMPLED_LENGTH),
        )
        boundaries_list = []
        beat_features_list = []
        mean = np.asarray(pop_mean, dtype=np.float32)[:, None]
        std = np.asarray(pop_std, dtype=np.float32)[:, None]

        for idx, (record_idx, segment_stack, target, beat_boundaries, beat_feats, abp_wave) in enumerate(iterate_processed_records()):
            inputs[idx] = ((segment_stack - mean) / std).astype(np.float32)
            targets[idx] = np.asarray(target, dtype=np.float32)
            record_ids[idx] = record_idx
            starts_arr[idx] = 0
            ends_arr[idx] = RESAMPLED_LENGTH
            demo = (
                np.zeros(5, dtype=np.float32)
                if self.demographics_by_record is None
                else self.demographics_by_record[record_idx]
            )
            demos[idx] = np.asarray(demo, dtype=np.float32)
            abp_waves[idx] = np.asarray(abp_wave, dtype=np.float32)
            boundaries_list.append(beat_boundaries)
            beat_features_list.append(beat_feats)

        inputs.flush()
        targets.flush(); record_ids.flush(); starts_arr.flush()
        ends_arr.flush(); demos.flush(); abp_waves.flush()

        pop_stats = {"channel_mean": pop_mean, "channel_std": pop_std}
        with cache_paths["pop_stats"].open("wb") as fh:
            pickle.dump(pop_stats, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with cache_paths["boundaries"].open("wb") as fh:
            pickle.dump(boundaries_list, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with cache_paths["beat_features"].open("wb") as fh:
            pickle.dump(beat_features_list, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with cache_paths["meta"].open("wb") as fh:
            pickle.dump({"num_samples": num_samples, "channels": channels}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        sample_subjects = np.asarray(
            [
                "" if self.subjects_by_record is None
                else str(self.subjects_by_record[int(record_ids[i])])
                for i in range(num_samples)
            ],
            dtype=object,
        )
        with cache_paths["subjects"].open("wb") as fh:
            pickle.dump(sample_subjects, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def _cache_paths(self, record_indices, max_records):
        cache_indices = np.sort(np.asarray(record_indices, dtype=np.int64))
        digest = hashlib.sha1(np.ascontiguousarray(cache_indices).tobytes()).hexdigest()[:12]
        stats_digest = ""
        if self._ext_channel_mean is not None and self._ext_channel_std is not None:
            stats_payload = np.concatenate([self._ext_channel_mean, self._ext_channel_std]).astype(np.float32)
            stats_digest = "_stats" + hashlib.sha1(np.ascontiguousarray(stats_payload).tobytes()).hexdigest()[:12]
        stem = Path(self.file_path).stem
        name = (
            f"{stem}_{self.input_mode}_{CACHE_VERSION}_max{max_records}_n{len(record_indices)}_{digest}{stats_digest}"
        )
        cache_root = self.cache_dir / name
        return {
            "dir":           cache_root,
            "meta":          cache_root / "meta.pkl",
            "inputs":        cache_root / "inputs.npy",
            "targets":       cache_root / "targets.npy",
            "record_idx":    cache_root / "record_idx.npy",
            "start":         cache_root / "start.npy",
            "end":           cache_root / "end.npy",
            "demo":          cache_root / "demo.npy",
            "abp_wave":      cache_root / "abp_wave.npy",
            "boundaries":    cache_root / "boundaries.pkl",
            "beat_features": cache_root / "beat_features.pkl",
            "pop_stats":     cache_root / "pop_stats.pkl",
            "subjects":      cache_root / "subjects.pkl",
        }

    def _load_cache(self, cache_paths):
        self.inputs = np.load(cache_paths["inputs"], mmap_mode="r")
        self.targets    = np.load(cache_paths["targets"],    mmap_mode="r")
        self.record_ids = np.load(cache_paths["record_idx"], mmap_mode="r")
        self.starts     = np.load(cache_paths["start"],      mmap_mode="r")
        self.ends       = np.load(cache_paths["end"],        mmap_mode="r")
        self.demos      = np.load(cache_paths["demo"],       mmap_mode="r")
        self.abp_waves  = np.load(cache_paths["abp_wave"],   mmap_mode="r")

        with cache_paths["boundaries"].open("rb") as fh:
            self.boundaries = pickle.load(fh)
        with cache_paths["beat_features"].open("rb") as fh:
            self.beat_features = pickle.load(fh)

        if cache_paths["subjects"].exists():
            with cache_paths["subjects"].open("rb") as fh:
                self._sample_subjects = pickle.load(fh)
        else:
            self._sample_subjects = None

        if cache_paths["pop_stats"].exists():
            with cache_paths["pop_stats"].open("rb") as fh:
                pop = pickle.load(fh)
            self._pop_channel_mean = pop["channel_mean"]
            self._pop_channel_std = pop["channel_std"]
        else:
            self._pop_channel_mean = None
            self._pop_channel_std = None

        self.samples = [None] * len(self.targets)

    def __len__(self):
        if hasattr(self, "targets"):
            return len(self.targets)
        return len(self.samples)

    def __getitem__(self, idx):
        x = np.asarray(self.inputs[idx], dtype=np.float32)
        sbp, dbp = np.asarray(self.targets[idx], dtype=np.float32)
        record_idx = int(self.record_ids[idx])
        boundaries = self.boundaries[idx]
        abp_wave = np.asarray(self.abp_waves[idx], dtype=np.float32)

        return {
            "inputs": torch.tensor(x, dtype=torch.float32),
            "blood_pressure": torch.tensor([sbp, dbp], dtype=torch.float32),
            "abp_waveform": torch.tensor(abp_wave, dtype=torch.float32),
            "record_index": torch.tensor(record_idx, dtype=torch.long),
            "subject_id": str(record_idx if self._sample_subjects is None else self._sample_subjects[idx]),
            "beat_boundaries": boundaries,
            "beat_features": torch.tensor(self.beat_features[idx], dtype=torch.float32),
            "demographics": torch.tensor(np.asarray(self.demos[idx], dtype=np.float32), dtype=torch.float32),
        }
