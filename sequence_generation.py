import argparse
import os

import numpy as np
import pandas as pd

INPUT_FEATURES = ['dx_m', 'dy_m', 'speed_m_s', 'turning_angle_deg']
TARGET_FEATURES = ['dx_m', 'dy_m']


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    missing = set(INPUT_FEATURES + TARGET_FEATURES) - set(df.columns)
    if missing:
        raise KeyError(f'Missing required columns: {sorted(missing)}')


def build_sequences(
    df: pd.DataFrame,
    history_steps: int = 48,
    future_steps: int = 12,
    stride: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    validate_columns(df)
    df = df.dropna(subset=INPUT_FEATURES + TARGET_FEATURES)

    x = df[INPUT_FEATURES].to_numpy(dtype=float)
    y = df[TARGET_FEATURES].to_numpy(dtype=float)

    total = history_steps + future_steps
    max_start = x.shape[0] - total
    if max_start < 0:
        return np.empty((0, history_steps, len(INPUT_FEATURES))), np.empty((0, future_steps, len(TARGET_FEATURES)))

    xs = []
    ys = []
    for start in range(0, max_start + 1, stride):
        xs.append(x[start:start + history_steps])
        ys.append(y[start + history_steps:start + total])

    return np.stack(xs), np.stack(ys)


def save_sequences(output_path: str, x: np.ndarray, y: np.ndarray) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, X=x, y=y)


def process_file(
    input_path: str,
    output_path: str,
    history_steps: int,
    future_steps: int,
    stride: int,
) -> tuple[int, int]:
    df = load_csv(input_path)
    x, y = build_sequences(df, history_steps, future_steps, stride)
    save_sequences(output_path, x, y)
    return x.shape[0], x.shape[1]


def process_dir(
    input_dir: str,
    output_dir: str,
    history_steps: int,
    future_steps: int,
    stride: int,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for name in os.listdir(input_dir):
        if not name.lower().endswith('.csv'):
            continue
        input_path = os.path.join(input_dir, name)
        base = os.path.splitext(name)[0]
        output_path = os.path.join(output_dir, f'{base}.npz')
        try:
            count, hist = process_file(
                input_path,
                output_path,
                history_steps,
                future_steps,
                stride,
            )
            rows.append(
                {
                    'file': name,
                    'sequences': count,
                    'history_steps': hist,
                    'future_steps': future_steps,
                    'stride': stride,
                }
            )
            print(f'Wrote {output_path} ({count} sequences)')
        except Exception as exc:
            print(f'Skipped {input_path}: {exc}')

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate sliding-window sequences for motion features.'
    )
    parser.add_argument('--input', required=True, help='Input CSV file or directory.')
    parser.add_argument('--output', required=True, help='Output file or directory.')
    parser.add_argument('--history', type=int, default=48, help='History steps.')
    parser.add_argument('--future', type=int, default=12, help='Future steps.')
    parser.add_argument('--stride', type=int, default=6, help='Stride steps.')
    parser.add_argument('--summary', default=None, help='Optional CSV summary path.')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        summary = process_dir(
            args.input,
            args.output,
            args.history,
            args.future,
            args.stride,
        )
        if args.summary:
            os.makedirs(os.path.dirname(args.summary), exist_ok=True)
            summary.to_csv(args.summary, index=False)
    else:
        output_path = args.output
        if os.path.isdir(args.output):
            base = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(args.output, f'{base}.npz')
        count, _ = process_file(
            args.input,
            output_path,
            args.history,
            args.future,
            args.stride,
        )
        print(f'Wrote {output_path} ({count} sequences)')


if __name__ == '__main__':
    main()
