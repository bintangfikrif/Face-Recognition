import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "data_processed"
TARGET_SIZE = (224, 224)

def load_dataset():
    X = []
    y = []
    label_map = {}

    folders = sorted(os.listdir(DATA_DIR))
    for idx, folder in enumerate(folders):
        label_map[folder] = idx
        folder_path = os.path.join(DATA_DIR, folder)

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(folder_path, filename)
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, TARGET_SIZE)

                X.append(img)
                y.append(idx)

    X = np.array(X)
    y = np.array(y)

    print(f"📦 Loaded {len(X)} images from {len(label_map)} classes.")
    return X, y, label_map


def split_dataset(X, y, test_size=0.2, val_size=0.1):
    # Split test terlebih dahulu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Split validation dari train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=42
    )

    print(f"""
    🔹 Train: {len(X_train)}
    🔹 Val:   {len(X_val)}
    🔹 Test:  {len(X_test)}
    """)

    return X_train, X_val, X_test, y_train, y_val, y_test
