import numpy as np
from sklearn.model_selection import StratifiedKFold

# ============================================================
# MEMBACA HASIL PREPROCESSING
# ============================================================

# Karena X dan y dari 11_preprocessing.py hanya ada di memori,
# kita ulangi preprocessing di sini.

import os
import cv2

DATASET_DIR = "/Volumes/OS/all video/Dataset CCTV"

KELAS = [
    "Pelanggan Datang",
    "Pelanggan Menunggu",
    "Pelanggan Dilayani"
]

UKURAN_GAMBAR = (224, 224)

data = []
labels = []

for label, nama_kelas in enumerate(KELAS):

    folder_kelas = os.path.join(DATASET_DIR, nama_kelas)

    for nama_file in os.listdir(folder_kelas):

        # Abaikan file metadata macOS
        if nama_file.startswith("._"):
            continue

        # Hanya membaca file gambar
        if not nama_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path_gambar = os.path.join(folder_kelas, nama_file)

        gambar = cv2.imread(path_gambar)

        if gambar is None:
            continue

        # Resize
        gambar = cv2.resize(gambar, UKURAN_GAMBAR)

        # BGR → RGB
        gambar = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)

        # Normalisasi 0–1
        gambar = gambar.astype("float32") / 255.0

        data.append(gambar)
        labels.append(label)


X = np.array(data, dtype="float32")
y = np.array(labels, dtype="int32")


# ============================================================
# STRATIFIED 5-FOLD
# ============================================================

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# MENAMPILKAN PEMBAGIAN SETIAP FOLD
# ============================================================

print("\n==============================")
print("STRATIFIED 5-FOLD")
print("==============================")

print(f"Total data : {len(X)}")
print(f"Jumlah fold: 5")

for nomor_fold, (train_index, val_index) in enumerate(
    kfold.split(X, y), start=1
):

    y_train = y[train_index]
    y_val = y[val_index]

    print(f"\n---------- FOLD {nomor_fold} ----------")

    print(f"Data training   : {len(train_index)}")
    print(f"Data validasi   : {len(val_index)}")

    print("\nTraining:")
    for i, nama_kelas in enumerate(KELAS):
        jumlah = np.sum(y_train == i)
        print(f"  {nama_kelas}: {jumlah}")

    print("Validasi:")
    for i, nama_kelas in enumerate(KELAS):
        jumlah = np.sum(y_val == i)
        print(f"  {nama_kelas}: {jumlah}")