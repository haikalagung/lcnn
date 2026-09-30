from pathlib import Path
import cv2

DATASET_DIR = Path("/Volumes/OS/all video/Dataset CCTV")

KELAS = [
    "Pelanggan Datang",
    "Pelanggan Menunggu",
    "Pelanggan Dilayani"
]

total = 0
rusak = 0

print("=" * 60)
print("CEK DATASET CCTV")
print("=" * 60)

for kelas in KELAS:
    folder = DATASET_DIR / kelas

    files = [
        f for f in folder.glob("*.jpg")
        if "._" not in f.name
    ]

    print(f"\n{kelas}")
    print(f"Jumlah file : {len(files)}")

    resolusi = {}
    rusak_kelas = 0

    for file in files:
        img = cv2.imread(str(file))

        if img is None:
            rusak_kelas += 1
            continue

        tinggi, lebar = img.shape[:2]
        ukuran = f"{lebar} x {tinggi}"

        resolusi[ukuran] = resolusi.get(ukuran, 0) + 1

    print(f"File rusak  : {rusak_kelas}")
    print("Resolusi    :")

    for ukuran, jumlah in sorted(resolusi.items()):
        print(f"  {ukuran} : {jumlah}")

    total += len(files)
    rusak += rusak_kelas

print("\n" + "=" * 60)
print("HASIL AKHIR")
print("=" * 60)
print(f"Total citra : {total}")
print(f"Citra rusak : {rusak}")

if total == 4900 and rusak == 0:
    print("\n✓ DATASET SIAP UNTUK TAHAP BERIKUTNYA")
else:
    print("\n! PERLU DIPERIKSA KEMBALI")