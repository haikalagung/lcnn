import os
import cv2

# ============================================================
# LOKASI DATASET
# ============================================================

DATASET_DIR = "/Volumes/OS/all video/Dataset CCTV"

KELAS = [
    "Pelanggan Datang",
    "Pelanggan Menunggu",
    "Pelanggan Dilayani"
]


# ============================================================
# MEMBACA DATASET
# ============================================================

data = []
labels = []

for label, nama_kelas in enumerate(KELAS):

    folder_kelas = os.path.join(DATASET_DIR, nama_kelas)

    if not os.path.exists(folder_kelas):
        print(f"Folder tidak ditemukan: {folder_kelas}")
        continue

    jumlah = 0

    for nama_file in os.listdir(folder_kelas):

        # ----------------------------------------------------
        # Abaikan file metadata dari macOS
        # Contoh: ._nama_file.jpg
        # ----------------------------------------------------
        if nama_file.startswith("._"):
            continue

        # ----------------------------------------------------
        # Hanya membaca file gambar
        # ----------------------------------------------------
        if not nama_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path_gambar = os.path.join(folder_kelas, nama_file)

        # ----------------------------------------------------
        # Cek apakah gambar bisa dibaca
        # ----------------------------------------------------
        gambar = cv2.imread(path_gambar)

        if gambar is None:
            print(f"Gambar tidak dapat dibaca: {path_gambar}")
            continue

        # Simpan lokasi gambar dan label
        data.append(path_gambar)
        labels.append(label)

        jumlah += 1

    print(f"{nama_kelas}: {jumlah} gambar")


# ============================================================
# HASIL PEMBACAAN DATASET
# ============================================================

print("\n==============================")
print("HASIL PEMBACAAN DATASET")
print("==============================")

print(f"Total gambar : {len(data)}")
print(f"Total label  : {len(labels)}")

print("\nKode kelas:")
for i, nama_kelas in enumerate(KELAS):
    print(f"{i} = {nama_kelas}")