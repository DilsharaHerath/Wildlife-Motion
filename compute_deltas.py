import argparse
import os

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371008.8


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


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    required = {'location-lat', 'location-long'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing required columns: {sorted(missing)}')

    df = df.copy()
    df['timestamp'] = build_timestamp(df)
    df = df.dropna(subset=['timestamp', 'location-lat', 'location-long'])
    df = df.sort_values('timestamp')

    lat = np.radians(df['location-lat'].to_numpy())
    lon = np.radians(df['location-long'].to_numpy())

    dlat = np.diff(lat, prepend=np.nan)
    dlon = np.diff(lon, prepend=np.nan)
    mean_lat = (lat + np.roll(lat, 1)) / 2.0
    mean_lat[0] = np.nan

    dx = EARTH_RADIUS_M * dlon * np.cos(mean_lat)
    dy = EARTH_RADIUS_M * dlat
    distance = np.sqrt(dx ** 2 + dy ** 2)

    dt_seconds = (
        df['timestamp'].diff().dt.total_seconds().to_numpy()
    )
    speed = distance / dt_seconds

    vx = np.roll(dx, 1)
    vy = np.roll(dy, 1)
    dot = vx * dx + vy * dy
    cross = vx * dy - vy * dx
    turning_angle = np.degrees(np.arctan2(cross, dot))
    turning_angle[0] = np.nan

    df['dx_m'] = dx
    df['dy_m'] = dy
    df['distance_m'] = distance
    df['dt_seconds'] = dt_seconds
    df['speed_m_s'] = speed
    df['turning_angle_deg'] = turning_angle
    return df


def process_file(input_path: str, output_path: str) -> None:
    df = load_csv(input_path)
    df = compute_deltas(df)
    df.to_csv(output_path, index=False)


def process_dir(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(input_dir):
        if not name.lower().endswith('.csv'):
            continue
        if name.endswith('summary.csv'):
            continue
        input_path = os.path.join(input_dir, name)
        output_path = os.path.join(output_dir, name)
        try:
            process_file(input_path, output_path)
            print(f'Wrote {output_path}')
        except Exception as exc:
            print(f'Skipped {input_path}: {exc}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute dx, dy, speed, and turning angle.'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file or directory.',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output CSV file or directory.',
    )
    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_dir(args.input, args.output)
    else:
        output_path = args.output
        if os.path.isdir(args.output):
            output_path = os.path.join(args.output, os.path.basename(args.input))
        process_file(args.input, output_path)


if __name__ == '__main__':
    main()
