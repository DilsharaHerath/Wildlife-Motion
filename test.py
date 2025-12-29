import numpy as np

def inspect_npz(npz_path, preview_elements=5):
    """
    Inspect the contents of a .npz file.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file
    preview_elements : int
        Number of elements to preview from each array
    """
    print("=" * 60)
    print(f"Inspecting NPZ file: {npz_path}")
    print("=" * 60)

    with np.load(npz_path, allow_pickle=True) as data:
        keys = list(data.keys())

        print(f"Number of arrays: {len(keys)}")
        print(f"Keys: {keys}")

        for key in keys:
            arr = data[key]
            print("\n" + "-" * 50)
            print(f"Key: {key}")
            print(f"Type: {type(arr)}")

            if isinstance(arr, np.ndarray):
                print(f"Shape: {arr.shape}")
                print(f"Dtype: {arr.dtype}")

                # Numerical checks
                if np.issubdtype(arr.dtype, np.number):
                    nan_count = np.isnan(arr).sum()
                    inf_count = np.isinf(arr).sum()

                    print(f"NaNs: {nan_count}")
                    print(f"Infs: {inf_count}")

                    if arr.size > 0:
                        print(f"Min: {np.nanmin(arr)}")
                        print(f"Max: {np.nanmax(arr)}")
                        print(f"Mean: {np.nanmean(arr)}")
                        print(f"Std: {np.nanstd(arr)}")

                # Preview elements
                flat = arr.flatten()
                print(f"Preview ({preview_elements} elements): {flat[:preview_elements]}")

            else:
                print("Non-numpy object stored in this key")

    print("\nInspection completed.")


npz_file = "./data/sequences/LA1.npz"
inspect_npz(npz_file)



# Season Wise prediction - Train test validate