from pathlib import Path
from collections import defaultdict

DATASET_DIR = Path("/Volumes/OS/all video/Dataset CCTV")

KELAS = [
    "Pelanggan Datang",
    "Pelanggan Menunggu",
    "Pelanggan Dilayani"
]

data = defaultdict(lambda: {
    "Pelanggan Datang": 0,
    "Pelanggan Menunggu": 0,
    "Pelanggan Dilayani": 0
})

for kelas in KELAS:
    folder = DATASET_DIR / kelas

    for file in folder.glob("*.jpg"):
        if "._" in file.name:
            continue

        # Contoh:
        # 25 sep_253_frame_000123.jpg
        nama = file.stem

        if "_frame_" not in nama:
            continue

        video = nama.split("_frame_")[0]
        data[video][kelas] += 1

print("=" * 75)
print("DISTRIBUSI KELAS BERDASARKAN VIDEO SUMBER")
print("=" * 75)

for video in sorted(data):
    datang = data[video]["Pelanggan Datang"]
    menunggu = data[video]["Pelanggan Menunggu"]
    dilayani = data[video]["Pelanggan Dilayani"]
    total = datang + menunggu + dilayani

    print(
        f"{video:<35} | "
        f"Datang: {datang:>4} | "
        f"Menunggu: {menunggu:>4} | "
        f"Dilayani: {dilayani:>4} | "
        f"Total: {total:>4}"
    )

print("=" * 75)
print(f"Jumlah video : {len(data)}")
print(f"Total citra  : {sum(sum(x.values()) for x in data.values())}")
print("=" * 75)