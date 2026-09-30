import os
import cv2
import numpy as np

# ============================================================
# LOKASI DATASET
# ============================================================

DATASET_DIR = "/Volumes/OS/all video/Dataset CCTV"

KELAS = [
    "Pelanggan Datang",
    "Pelanggan Menunggu",
    "Pelanggan Dilayani"
]

# Ukuran gambar yang digunakan oleh Lightweight CNN
UKURAN_GAMBAR = (224, 224)


# ============================================================
# MEMBACA DAN MELAKUKAN PREPROCESSING
# ============================================================

data = []
labels = []

for label, nama_kelas in enumerate(KELAS):

    folder_kelas = os.path.join(DATASET_DIR, nama_kelas)

    jumlah = 0

    for nama_file in os.listdir(folder_kelas):

        # Abaikan file metadata macOS
        if nama_file.startswith("._"):
            continue

        # Hanya membaca file gambar
        if not nama_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path_gambar = os.path.join(folder_kelas, nama_file)

        # Membaca gambar
        gambar = cv2.imread(path_gambar)

        if gambar is None:
            continue

        # ----------------------------------------------------
        # 1. Ubah ukuran menjadi 224 × 224
        # ----------------------------------------------------
        gambar = cv2.resize(gambar, UKURAN_GAMBAR)

        # ----------------------------------------------------
        # 2. Ubah BGR menjadi RGB
        # ----------------------------------------------------
        gambar = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)

        # ----------------------------------------------------
        # 3. Normalisasi nilai pixel
        #    dari 0–255 menjadi 0–1
        # ----------------------------------------------------
        gambar = gambar.astype("float32") / 255.0

        # Simpan gambar dan label
        data.append(gambar)
        labels.append(label)

        jumlah += 1

    print(f"{nama_kelas}: {jumlah} gambar")


# ============================================================
# UBAH MENJADI NUMPY ARRAY
# ============================================================

X = np.array(data, dtype="float32")
y = np.array(labels, dtype="int32")


# ============================================================
# HASIL PREPROCESSING
# ============================================================

print("\n==============================")
print("HASIL PREPROCESSING")
print("==============================")

print(f"Jumlah data : {len(X)}")
print(f"Shape X     : {X.shape}")
print(f"Shape y     : {y.shape}")

print(f"\nNilai pixel minimum : {X.min():.4f}")
print(f"Nilai pixel maksimum : {X.max():.4f}")

print("\nDistribusi kelas:")

for i, nama_kelas in enumerate(KELAS):
    jumlah = np.sum(y == i)
    print(f"{i} = {nama_kelas}: {jumlah}")