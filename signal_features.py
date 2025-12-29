import argparse
import os

import numpy as np
import pandas as pd

try:
    import pywt
except ImportError as exc:
    raise ImportError(
        "pywt is required. Install with: pip install PyWavelets"
    ) from exc

INPUT_FEATURES = ['dx_m', 'dy_m', 'speed_m_s', 'turning_angle_deg']
SIGNAL_COLUMNS = ['speed_m_s', 'turning_angle_deg']


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    if 'timestamp' in df.columns:
        return pd.to_datetime(df['timestamp'], errors='coerce')
    if {'date', 'time'}.issubset(df.columns):
        return pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str),
            errors='coerce',
        )
    raise KeyError('Missing timestamp or date/time columns')


def add_hour_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '_timestamp' in df.columns:
        ts = pd.to_datetime(df['_timestamp'], errors='coerce')
    else:
        ts = build_timestamp(df)
    hour = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    return df


def wavelet_energy(series: np.ndarray, wavelet: str, level: int) -> list[float]:
    max_level = pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len)
    if max_level < level:
        return [np.nan] * level

    coeffs = pywt.wavedec(series, wavelet, level=level)
    details = coeffs[1:]
    energies = [float(np.mean(c ** 2)) for c in details]
    return energies


def fft_power_bins(series: np.ndarray, bins: int) -> list[float]:
    if len(series) < 2:
        return [np.nan] * bins

    series = series - np.mean(series)
    power = np.abs(np.fft.rfft(series)) ** 2
    bins_available = power.shape[0] - 1
    if bins_available <= 0:
        return [np.nan] * bins

    take = min(bins, bins_available)
    values = power[1:1 + take].tolist()
    if take < bins:
        values.extend([np.nan] * (bins - take))
    return [float(v) for v in values]


def compute_signal_features(
    df: pd.DataFrame,
    window_steps: int = 72,
    wavelet: str = 'db4',
    wavelet_level: int = 5,
    fft_bins: int = 10,
) -> pd.DataFrame:
    missing = set(INPUT_FEATURES) - set(df.columns)
    if missing:
        raise KeyError(f'Missing required columns: {sorted(missing)}')

    df = df.copy()
    df['_timestamp'] = build_timestamp(df)
    df = df.dropna(subset=['_timestamp'])
    df = df.sort_values('_timestamp')

    speed = df['speed_m_s'].to_numpy(dtype=float)
    angle = df['turning_angle_deg'].to_numpy(dtype=float)

    wavelet_cols = [f'wavelet_L{i}' for i in range(1, wavelet_level + 1)]
    fft_cols = [f'fft_bin_{i}' for i in range(1, fft_bins + 1)]
    wavelet_features = np.full((len(df), wavelet_level), np.nan, dtype=float)
    fft_features = np.full((len(df), fft_bins), np.nan, dtype=float)

    for end in range(window_steps - 1, len(df)):
        start = end - window_steps + 1
        speed_window = speed[start:end + 1]
        angle_window = angle[start:end + 1]

        if not np.isfinite(speed_window).all() or not np.isfinite(angle_window).all():
            continue

        wavelet_speed = wavelet_energy(speed_window, wavelet, wavelet_level)
        wavelet_angle = wavelet_energy(angle_window, wavelet, wavelet_level)
        wavelet_features[end] = (np.array(wavelet_speed) + np.array(wavelet_angle)) / 2.0

        fft_speed = fft_power_bins(speed_window, fft_bins)
        fft_angle = fft_power_bins(angle_window, fft_bins)
        fft_features[end] = (np.array(fft_speed) + np.array(fft_angle)) / 2.0

    df[wavelet_cols] = wavelet_features
    df[fft_cols] = fft_features
    df = add_hour_encodings(df)
    return df


def process_file(
    input_path: str,
    output_path: str,
    window_steps: int,
    wavelet: str,
    wavelet_level: int,
    fft_bins: int,
) -> None:
    df = load_csv(input_path)
    df = compute_signal_features(
        df,
        window_steps=window_steps,
        wavelet=wavelet,
        wavelet_level=wavelet_level,
        fft_bins=fft_bins,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def process_dir(
    input_dir: str,
    output_dir: str,
    window_steps: int,
    wavelet: str,
    wavelet_level: int,
    fft_bins: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(input_dir):
        if not name.lower().endswith('.csv'):
            continue
        input_path = os.path.join(input_dir, name)
        output_path = os.path.join(output_dir, name)
        try:
            process_file(
                input_path,
                output_path,
                window_steps,
                wavelet,
                wavelet_level,
                fft_bins,
            )
            print(f'Wrote {output_path}')
        except Exception as exc:
            print(f'Skipped {input_path}: {exc}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract wavelet/FFT features for speed and turning angle.'
    )
    parser.add_argument('--input', required=True, help='Input CSV file or directory.')
    parser.add_argument('--output', required=True, help='Output CSV file or directory.')
    parser.add_argument('--window-steps', type=int, default=72, help='Window size in steps.')
    parser.add_argument('--wavelet', default='db4', help='Wavelet name.')
    parser.add_argument('--wavelet-level', type=int, default=5, help='Wavelet decomposition level.')
    parser.add_argument('--fft-bins', type=int, default=10, help='Number of FFT bins.')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_dir(
            args.input,
            args.output,
            args.window_steps,
            args.wavelet,
            args.wavelet_level,
            args.fft_bins,
        )
    else:
        output_path = args.output
        if os.path.isdir(args.output):
            output_path = os.path.join(args.output, os.path.basename(args.input))
        process_file(
            args.input,
            output_path,
            args.window_steps,
            args.wavelet,
            args.wavelet_level,
            args.fft_bins,
        )


if __name__ == '__main__':
    main()
