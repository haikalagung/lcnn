from pathlib import Path
import shutil
import cv2


# =========================
# PENGATURAN
# =========================

SOURCE_DIR = Path("/Volumes/OS/all video/Frame CCTV Gabungan")
OUTPUT_DIR = Path("/Volumes/OS/all video/Dataset CCTV")

# Proses semua frame
MAX_FRAMES = None


# =========================
# LABEL
# =========================

LABELS = {
    "1": "Pelanggan Datang",
    "2": "Pelanggan Menunggu",
    "3": "Pelanggan Dilayani",
    "4": "Redundan",
    "5": "Tidak Layak",
}


# =========================
# SIAPKAN FOLDER
# =========================

for label in LABELS.values():
    (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)


# =========================
# AMBIL FRAME
# =========================

frames = sorted(
    [
        file
        for file in SOURCE_DIR.glob("*.jpg")
        if "._" not in file.name
    ]
)

if MAX_FRAMES is not None:
    frames = frames[:MAX_FRAMES]


# =========================
# HITUNG YANG SUDAH DILABEL
# =========================

sudah_dilabel = set()

for label in LABELS.values():
    folder = OUTPUT_DIR / label

    for file in folder.glob("*.jpg"):
        if "._" not in file.name:
            sudah_dilabel.add(file.name)


# =========================
# HITUNG JUMLAH LABEL
# =========================

jumlah_label = {
    label: 0
    for label in LABELS.values()
}

for label in LABELS.values():
    folder = OUTPUT_DIR / label

    jumlah_label[label] = sum(
        1
        for file in folder.glob("*.jpg")
        if "._" not in file.name
    )


# =========================
# INFORMASI AWAL
# =========================

print("=" * 60)
print("LABELING DATA CCTV")
print("=" * 60)
print(f"Total frame        : {len(frames):,}")
print(f"Sudah dilabel      : {len(sudah_dilabel):,}")
print(f"Belum dilabel      : {len(frames) - len(sudah_dilabel):,}")
print()
print("Kontrol:")
print("[1] Pelanggan Datang")
print("[2] Pelanggan Menunggu")
print("[3] Pelanggan Dilayani")
print("[4] Redundan")
print("[5] Tidak Layak")
print("[Q] Keluar")
print("=" * 60)


# =========================
# PROSES LABELING
# =========================

for nomor, frame_path in enumerate(frames, start=1):

    # Lewati frame yang sudah pernah dilabel
    if frame_path.name in sudah_dilabel:
        continue

    image = cv2.imread(str(frame_path))

    if image is None:
        print(f"Gagal membaca: {frame_path.name}")
        continue

    while True:

        tampil = image.copy()

        # Informasi progress
        teks = f"{nomor}/{len(frames)}"

        cv2.putText(
            tampil,
            teks,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Labeling CCTV", tampil)

        key = cv2.waitKey(0) & 0xFF

        # =========================
        # KELUAR
        # =========================

        if key == ord("q"):
            cv2.destroyAllWindows()

            print()
            print("=" * 60)
            print("LABELING DIHENTIKAN")
            print("=" * 60)

            for label, jumlah in jumlah_label.items():
                print(f"{label:<25}: {jumlah:,}")

            print()
            print("Hasil yang sudah dilabel tetap tersimpan.")
            print("Jalankan script lagi untuk melanjutkan.")

            exit()


        # =========================
        # LABEL
        # =========================

        if chr(key) in LABELS:

            label = LABELS[chr(key)]

            tujuan = OUTPUT_DIR / label / frame_path.name

            shutil.copy2(frame_path, tujuan)

            jumlah_label[label] += 1
            sudah_dilabel.add(frame_path.name)

            print(
                f"[{nomor}/{len(frames)}] "
                f"{frame_path.name} -> {label}"
            )

            break


cv2.destroyAllWindows()


# =========================
# SELESAI
# =========================

print()
print("=" * 60)
print("LABELING SELESAI")
print("=" * 60)

for label, jumlah in jumlah_label.items():
    print(f"{label:<25}: {jumlah:,}")

print()
print(f"Lokasi dataset: {OUTPUT_DIR}")