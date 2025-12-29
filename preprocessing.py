import os
import re

import pandas as pd

DATA_PATH = 'data/African elephants in Etosha National Park (data from Tsalyuk et al. 2018).csv'
OUTPUT_DIR = 'data/individuals1'
INTERVAL_SUMMARY_PATH = 'data/individuals1/interval-summary.csv'
MISSING_POINTS_PATH = 'data/individuals1/missing-points.csv'
DAY_FILTER_SUMMARY_PATH = 'data/individuals1/day-filter-summary.csv'

COLUMNS_TO_DROP = [
    'event-id',
    'visible',
    'manually-marked-outlier',
    'sensor-type',
    'individual-taxon-canonical-name',
    'tag-local-identifier',
    'study-name',
]

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.drop(columns=columns, errors='ignore')


def sanitize_filename(value: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('._-')
    return safe or 'unknown'


def add_date_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'timestamp' not in df.columns:
        raise KeyError('Missing required column: timestamp')

    df = df.copy()
    timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = timestamps.dt.date
    df['time'] = timestamps.dt.time
    df = df.drop(columns=['timestamp'])
    return df


def normalize_timestamps(df: pd.DataFrame, interval: str = '20min') -> pd.DataFrame:
    if 'timestamp' not in df.columns:
        raise KeyError('Missing required column: timestamp')

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.floor(interval)

    if 'individual-local-identifier' in df.columns:
        df = df.drop_duplicates(subset=['individual-local-identifier', 'timestamp'])
    else:
        df = df.drop_duplicates(subset=['timestamp'])

    return df


def save_by_individual(df: pd.DataFrame, output_dir: str) -> None:
    if 'individual-local-identifier' not in df.columns:
        raise KeyError('Missing required column: individual-local-identifier')

    os.makedirs(output_dir, exist_ok=True)
    for identifier, group in df.groupby('individual-local-identifier', dropna=False):
        identifier_str = 'unknown' if pd.isna(identifier) else str(identifier)
        filename = f'{sanitize_filename(identifier_str)}.csv'
        output_path = os.path.join(output_dir, filename)
        group.to_csv(output_path, index=False)


def summarize_intervals(df: pd.DataFrame) -> pd.DataFrame:
    if 'individual-local-identifier' not in df.columns:
        raise KeyError('Missing required column: individual-local-identifier')
    if 'timestamp' not in df.columns:
        raise KeyError('Missing required column: timestamp')

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    summaries = []
    for identifier, group in df.groupby('individual-local-identifier', dropna=False):
        identifier_str = 'unknown' if pd.isna(identifier) else str(identifier)
        group = group.dropna(subset=['timestamp']).sort_values('timestamp')
        diffs = group['timestamp'].diff().dt.total_seconds() / 60.0
        diffs = diffs.dropna()
        is_20min = diffs == 20.0

        summaries.append(
            {
                'individual-local-identifier': identifier_str,
                'points': int(group.shape[0]),
                'intervals': int(diffs.shape[0]),
                'interval_min': diffs.min() if not diffs.empty else None,
                'interval_median': diffs.median() if not diffs.empty else None,
                'interval_mean': diffs.mean() if not diffs.empty else None,
                'interval_max': diffs.max() if not diffs.empty else None,
                'interval_20min_count': int(is_20min.sum()) if not diffs.empty else 0,
                'interval_20min_pct': (is_20min.mean() * 100.0) if not diffs.empty else None,
            }
        )

    return pd.DataFrame(summaries)


def summarize_missing_points(df: pd.DataFrame, expected_per_day: int = 72) -> pd.DataFrame:
    required_columns = {'individual-local-identifier', 'date'}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f'Missing required columns: {sorted(missing)}')

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df = df.dropna(subset=['date'])

    counts = (
        df.groupby(['individual-local-identifier', 'date'])
        .size()
        .reset_index(name='points')
    )
    counts['expected_points'] = expected_per_day
    counts['missing_points'] = (expected_per_day - counts['points']).clip(lower=0)
    return counts


def filter_days_by_availability(
    df: pd.DataFrame,
    min_points_per_day: int = 65,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {'individual-local-identifier', 'date'}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f'Missing required columns: {sorted(missing)}')

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df = df.dropna(subset=['date'])

    counts = (
        df.groupby(['individual-local-identifier', 'date'])
        .size()
        .reset_index(name='points')
    )
    counts['kept'] = counts['points'] >= min_points_per_day
    counts['min_points_required'] = min_points_per_day

    kept_days = counts[counts['kept']]
    filtered_df = df.merge(
        kept_days[['individual-local-identifier', 'date']],
        on=['individual-local-identifier', 'date'],
        how='inner',
    )

    return filtered_df, counts


def main() -> None:
    df = load_data(DATA_PATH)
    df = drop_columns(df, COLUMNS_TO_DROP)
    df = normalize_timestamps(df)
    interval_summary = summarize_intervals(df)
    df = add_date_time_columns(df)
    df, day_filter_summary = filter_days_by_availability(df, min_points_per_day=65)
    print(df.info())
    save_by_individual(df, OUTPUT_DIR)
    interval_summary.to_csv(INTERVAL_SUMMARY_PATH, index=False)
    missing_points = summarize_missing_points(df)
    missing_points.to_csv(MISSING_POINTS_PATH, index=False)
    day_filter_summary.to_csv(DAY_FILTER_SUMMARY_PATH, index=False)


if __name__ == '__main__':
    main()
