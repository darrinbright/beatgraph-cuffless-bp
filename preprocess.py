import numpy as np
from scipy.signal import butter, find_peaks, resample, sosfiltfilt

from .config import (
    ABP_FILTER_ORDER,
    ABP_HIGH_CUTOFF_HZ,
    ABP_LOW_CUTOFF_HZ,
    ECG_FILTER_ORDER,
    ECG_HIGH_CUTOFF_HZ,
    ECG_LOW_CUTOFF_HZ,
    FS,
    INPUT_MODES,
    MAX_BPM,
    MIN_BPM,
    PPG_FILTER_ORDER,
    PPG_HIGH_CUTOFF_HZ,
    PPG_LOW_CUTOFF_HZ,
    RESAMPLED_LENGTH,
)


def bandpass_filter(signal, fs, low_hz, high_hz, order):
    signal = np.asarray(signal, dtype=np.float32)
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def lowpass_filter(signal, fs, high_hz, order):
    signal = np.asarray(signal, dtype=np.float32)
    sos = butter(order, high_hz, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def filter_ecg(signal, fs=FS):
    return bandpass_filter(signal, fs=fs, low_hz=ECG_LOW_CUTOFF_HZ, high_hz=ECG_HIGH_CUTOFF_HZ, order=ECG_FILTER_ORDER)


def filter_ppg(signal, fs=FS):
    return bandpass_filter(signal, fs=fs, low_hz=PPG_LOW_CUTOFF_HZ, high_hz=PPG_HIGH_CUTOFF_HZ, order=PPG_FILTER_ORDER)


def filter_abp(signal, fs=FS):
    return lowpass_filter(signal, fs=fs, high_hz=ABP_HIGH_CUTOFF_HZ, order=ABP_FILTER_ORDER)


def zscore_normalize(signal, eps=1e-6):
    signal = np.asarray(signal, dtype=np.float32)
    return ((signal - float(np.mean(signal))) / (float(np.std(signal)) + eps)).astype(np.float32)


def detect_abp_feet(abp_clean, fs=FS, min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    abp_clean = np.asarray(abp_clean, dtype=np.float32)
    min_dist = max(1, int(fs * 60.0 / max_bpm))
    prominence = max(0.1, 0.05 * float(np.max(abp_clean) - np.min(abp_clean)))
    feet, _ = find_peaks(-abp_clean, distance=min_dist, prominence=prominence)
    return feet.astype(np.int64)


def resample_signal(signal, length=RESAMPLED_LENGTH):
    return resample(np.asarray(signal, dtype=np.float32), length).astype(np.float32)


def detect_r_peaks(ecg_signal, fs=FS, min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    signal = zscore_normalize(filter_ecg(ecg_signal, fs=fs))
    min_dist = max(1, int(fs * 60.0 / max_bpm))
    prominence = max(0.2, 0.1 * float(np.max(signal) - np.min(signal)))
    peaks, _ = find_peaks(signal, distance=min_dist, prominence=prominence)
    return peaks.astype(np.int64)


def valid_rr_intervals(r_peaks, fs=FS, min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    if len(r_peaks) < 2:
        return False
    rr = np.diff(r_peaks).astype(np.float32) / float(fs)
    bpm = 60.0 / np.clip(rr, 1e-6, None)
    return bool(np.all((bpm >= min_bpm) & (bpm <= max_bpm)))


def segment_cycles(r_peaks, sequence_cycles, stride_cycles):
    segments = []
    needed = sequence_cycles + 1
    for start_idx in range(0, max(0, len(r_peaks) - needed + 1), stride_cycles):
        start = int(r_peaks[start_idx])
        end = int(r_peaks[start_idx + sequence_cycles])
        if end > start:
            segments.append((start, end, r_peaks[start_idx : start_idx + needed]))
    return segments


def prepare_segment_signals(ecg_raw, ppg_raw, start, end, fs=FS, output_length=RESAMPLED_LENGTH):
    ecg_seg = resample_signal(filter_ecg(ecg_raw[start:end], fs=fs), output_length)
    ppg_filt = filter_ppg(ppg_raw[start:end], fs=fs)
    dppg_orig = np.gradient(ppg_filt).astype(np.float32) * fs
    ddppg_orig = np.gradient(dppg_orig).astype(np.float32) * fs
    ppg_seg = resample_signal(ppg_filt, output_length)
    dppg = resample_signal(dppg_orig, output_length)
    ddppg = resample_signal(ddppg_orig, output_length)
    return ecg_seg, ppg_seg, dppg, ddppg

def ecg_based_beat_boundaries(r_peaks_resampled, window_len=RESAMPLED_LENGTH):
    r_peaks_resampled = np.asarray(r_peaks_resampled, dtype=np.int64)
    if r_peaks_resampled.size < 2:
        return []
    beats = []
    for idx in range(len(r_peaks_resampled) - 1):
        rr = int(r_peaks_resampled[idx + 1] - r_peaks_resampled[idx])
        start = max(0, int(r_peaks_resampled[idx] - int(0.25 * rr)))
        end = min(int(r_peaks_resampled[idx + 1] - int(0.05 * rr)), window_len)
        if end - start > 10:
            beats.append((start, end))
    return beats if len(beats) >= 2 else []


def preprocess_segment(
    ecg_raw,
    ppg_raw,
    start,
    end,
    input_mode,
    fs=FS,
    output_length=RESAMPLED_LENGTH,
    r_peaks_resampled=None,
    beat_boundaries=None,
    channel_mean=None,
    channel_std=None,
):
    if input_mode not in INPUT_MODES:
        raise KeyError(input_mode)

    ecg_seg, ppg_seg, dppg, ddppg = prepare_segment_signals(ecg_raw, ppg_raw, start, end, fs=fs, output_length=output_length)

    if input_mode == "ppg_d1_d2":
        channels = [ppg_seg, dppg, ddppg]
    else:
        channels = [ecg_seg, ppg_seg, dppg, ddppg]
    stack = np.stack(channels, axis=0).astype(np.float32)
    if channel_mean is not None and channel_std is not None:
        mean = np.asarray(channel_mean, dtype=np.float32)[:, None]
        std = np.asarray(channel_std, dtype=np.float32)[:, None]
        return ((stack - mean) / std).astype(np.float32)
    return np.stack([zscore_normalize(channel) for channel in channels], axis=0).astype(np.float32)


def compute_beat_rr_ptt_features(r_peaks_resampled, ppg_resampled, fs_effective):
    r_peaks = np.asarray(r_peaks_resampled, dtype=np.int64)
    ppg = np.asarray(ppg_resampled, dtype=np.float32)
    if len(r_peaks) < 2:
        return np.zeros((0, 4), dtype=np.float32)

    rr = np.diff(r_peaks).astype(np.float32) / float(fs_effective)
    delta_rr = np.concatenate([[0.0], np.diff(rr)]).astype(np.float32)
    hr_norm = (60.0 / np.clip(rr, 1e-3, None) - 70.0) / 40.0

    ptt = np.zeros(len(rr), dtype=np.float32)
    look_forward = max(1, int(0.3 * float(fs_effective)))
    for i in range(len(rr)):
        r_pos = int(r_peaks[i])
        s = r_pos
        e = min(len(ppg) - 1, r_pos + look_forward)
        if e > s:
            foot = s + int(np.argmin(ppg[s:e]))
            ptt[i] = float(foot - r_pos) / float(fs_effective)
        else:
            ptt[i] = 0.15

    return np.stack([rr, delta_rr, hr_norm, ptt], axis=1).astype(np.float32)
