import os
import pandas as pd

def create_dataframe(data_dir):
    data = []
    class_to_idx = {}

    for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_to_idx[class_name] = idx
        class_path = os.path.join(data_dir, class_name)

        for file in os.listdir(class_path):
            if file.endswith(".npy"):
                data.append({
                    "path": os.path.join(class_path, file),
                    "label": idx
                })

    df = pd.DataFrame(data)
    return df, class_to_idx