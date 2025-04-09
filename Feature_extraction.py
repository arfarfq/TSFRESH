import h5py
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb


def load_combined_data(filename):

    local_inputs_list, global_inputs_list, labels_list, df_list = [], [], [], []

    with h5py.File(filename, "r") as f:
                local_inputs_list.append(np.array(f["local_inputs"][:]))
                global_inputs_list.append(np.array(f["global_inputs"][:]))
                labels_list.append(np.array(f["labels"][:], dtype=int))
                metadata = f["metadata"][:]
                df = pd.DataFrame({
                    'TIC': metadata['TIC'].astype(int),
                    'sector': metadata['sector'].astype(int),
                    'path_to_fits': metadata['path_to_fits'].astype(str),
                    'TOI Disposition': metadata['TOI Disposition'].astype(str),
                    'label': metadata['label'].astype(int)
                })
                df_list.append(df)

    local_inputs = np.concatenate(local_inputs_list, axis=0)
    global_inputs = np.concatenate(global_inputs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    df_merged = pd.concat(df_list, ignore_index=True)


    local_inputs, global_inputs, labels, df_merged = clean_loaded_data(local_inputs, global_inputs, labels, df_merged)
    return local_inputs, global_inputs, labels, df_merged

def clean_loaded_data(local_inputs, global_inputs, labels, df_merged):
    valid_mask = (~np.isnan(local_inputs).any(axis=(1,2)) & 
                  ~np.isnan(global_inputs).any(axis=(1,2)))
    if not np.all(valid_mask):
        print(f"Removing {len(valid_mask) - np.sum(valid_mask)} invalid samples")
    local_clean = local_inputs[valid_mask]
    global_clean = global_inputs[valid_mask]
    labels_clean = labels[valid_mask]
    df_clean = df_merged.iloc[valid_mask].copy()

    assert len(local_clean) == len(global_clean) == len(labels_clean) == len(df_clean)
    assert not np.isnan(local_clean).any()
    assert not np.isnan(global_clean).any()
    return local_clean, global_clean, labels_clean, df_clean

def save_features_to_hdf5(features, filename):
    features.to_hdf(filename, key="features", mode="w")
    print(f"Features saved to {filename}")

def main():

    # Configuration
    TEST_MODE = False
    TEST_LIMIT = 1000 if TEST_MODE else None

    filename = "/mnt/data/LCs_1024_CNN_Input.h5"
    filename_TCE = "/mnt/data/TCEs_LCs_1024_CNN_Input.h5"

    # Load and prepare data
    _, local_inputs, labels, df_merged = load_combined_data(filename)
    _, local_inputs_b, labels_b, df_merged_b = load_combined_data(filename_TCE)

    # Concatenate the local_inputs (assuming numpy arrays)
    local_inputs = np.concatenate([local_inputs, local_inputs_b], axis=0)
    labels = np.concatenate([labels, labels_b], axis=0)
    df_merged = pd.concat([df_merged, df_merged_b], ignore_index=True)

    df_merged['row_id'] = range(len(df_merged))

    idx = []
    if TEST_MODE:

        pos_idx = df_merged.index[df_merged['label'] == 1].tolist()
        neg_idx = df_merged.index[df_merged['label'] == 0].tolist()
        
        # Balance positive and negative samples
        sample_size = int(TEST_LIMIT/2)
        
        idx.extend(neg_idx[:sample_size])
        idx.extend(pos_idx[:sample_size])
        
        
        local_inputs = [local_inputs[i] for i in idx]
        labels = [labels[i] for i in idx]
        df_merged = df_merged.iloc[idx]

    # Convert to long format for tsfresh
    long_data = []
    for i in range(len(local_inputs)):
        for t in range(len(local_inputs[i])):
            long_data.append({
                'TIC': df_merged['row_id'].iloc[i],
                'time': t,
                'flux': local_inputs[i][t].item()
            })

    df_long = pd.DataFrame(long_data)

    # Feature extraction
    features = extract_features(df_long, column_id="TIC", column_sort="time", column_value="flux", n_jobs=32)
    features = impute(features)

    features = pd.merge(
    features,
    df_merged,
    left_index=True,
    right_index=True,
    how='left'  # Keeps all rows from 'features'
)

    save_features_to_hdf5(features, "/mnt/data/Global_features.h5")

if __name__ == "__main__":
    main()
