import os
import re

import pandas as pd
import numpy as np

DATA_PATH = 'data/African elephants in Etosha National Park (data from Tsalyuk et al. 2018).csv'
OUTPUT_DIR = 'data/individuals'

COLUMNS_TO_DROP = [
    'event-id',
    'visible',
    'manually-marked-outlier',
    'sensor-type',
    'individual-taxon-canonical-name',
    'tag-local-identifier',
    'study-name',
]

EXPECTED_INTERVAL = pd.Timedelta(minutes=20)
INTERVAL_TOLERANCE = pd.Timedelta(minutes=2)
MAX_SPEED_KMH = 40.0


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.drop(columns=columns, errors='ignore')


def sanitize_filename(value: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('._-')
    return safe or 'unknown'


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df.dropna(subset=['timestamp'])


def filter_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['location-lat', 'location-long'])
    mask = df['location-lat'].between(-90, 90) & df['location-long'].between(-180, 180)
    return df[mask]


def sort_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('timestamp')
    return df.drop_duplicates(subset=['timestamp', 'location-lat', 'location-long'])


def filter_interval(df: pd.DataFrame) -> pd.DataFrame:
    diffs = df['timestamp'].diff()
    keep = diffs.isna() | (diffs.sub(EXPECTED_INTERVAL).abs() <= INTERVAL_TOLERANCE)
    return df[keep]


def haversine_km(lat1, lon1, lat2, lon2) -> pd.Series:
    radius_km = 6371.0088
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius_km * c


def filter_speed(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2:
        return df

    dist_km = haversine_km(
        df['location-lat'].shift(),
        df['location-long'].shift(),
        df['location-lat'],
        df['location-long'],
    )
    dt_hours = df['timestamp'].diff().dt.total_seconds() / 3600
    speed_kmh = dist_km / dt_hours
    keep = speed_kmh.isna() | (speed_kmh <= MAX_SPEED_KMH)
    return df[keep]


def preprocess_individual(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_timestamps(df)
    df = filter_coordinates(df)
    df = sort_and_dedup(df)
    df = filter_interval(df)
    df = filter_speed(df)
    return df


def save_by_individual(df: pd.DataFrame, output_dir: str) -> None:
    if 'individual-local-identifier' not in df.columns:
        raise KeyError('Missing required column: individual-local-identifier')

    os.makedirs(output_dir, exist_ok=True)
    for identifier, group in df.groupby('individual-local-identifier', dropna=False):
        identifier_str = 'unknown' if pd.isna(identifier) else str(identifier)
        filename = f'{sanitize_filename(identifier_str)}.csv'
        output_path = os.path.join(output_dir, filename)
        cleaned = preprocess_individual(group)
        cleaned.to_csv(output_path, index=False)


def main() -> None:
    df = load_data(DATA_PATH)
    df = drop_columns(df, COLUMNS_TO_DROP)
    print(df.info())
    save_by_individual(df, OUTPUT_DIR)


if __name__ == '__main__':
    main()
