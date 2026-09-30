from pathlib import Path
from collections import Counter

DATASET_DIR = Path("/Volumes/OS/all video/Dataset CCTV")

KELAS = [
    "Pelanggan Datang",
    "Pelanggan Menunggu",
    "Pelanggan Dilayani"
]

counter = Counter()

for kelas in KELAS:
    folder = DATASET_DIR / kelas

    for file in folder.glob("*.jpg"):
        if "._" in file.name:
            continue

        # Format nama:
        # 2 sep_21_frame_000001.jpg
        # Bagian sebelum "_frame_" adalah sumber video
        nama = file.stem

        if "_frame_" in nama:
            sumber_video = nama.split("_frame_")[0]
            counter[sumber_video] += 1

print("=" * 60)
print("JUMLAH FRAME BERDASARKAN VIDEO SUMBER")
print("=" * 60)

for sumber, jumlah in sorted(counter.items()):
    print(f"{sumber:<40} : {jumlah}")

print("=" * 60)
print(f"Jumlah video sumber : {len(counter)}")
print(f"Total frame         : {sum(counter.values())}")
print("=" * 60)