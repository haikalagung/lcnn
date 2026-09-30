import shutil
from pathlib import Path


# ============================================================
# FOLDER FRAME ASLI
# ============================================================
INPUT_DIR = Path("/Volumes/OS/all video/Frame CCTV")


# ============================================================
# FOLDER FRAME GABUNGAN
# ============================================================
OUTPUT_DIR = Path("/Volumes/OS/all video/Frame CCTV Gabungan")


# ============================================================
# BUAT FOLDER OUTPUT
# ============================================================
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CARI FRAME ASLI
# File ._ dari macOS tidak ikut
# ============================================================
frame_files = sorted(
    [
        path
        for path in INPUT_DIR.rglob("*.jpg")
        if not path.name.startswith("._")
    ]
)


print("=" * 70)
print("PENGGABUNGAN FRAME CCTV")
print("=" * 70)

print(
    f"Jumlah frame ditemukan : "
    f"{len(frame_files):,}"
)

print(
    f"Folder tujuan          : "
    f"{OUTPUT_DIR}"
)


# ============================================================
# PROSES PENYALINAN
# ============================================================
total = 0


for frame_path in frame_files:

    # --------------------------------------------------------
    # Ambil tanggal dan nama video
    # --------------------------------------------------------
    tanggal = frame_path.parent.parent.name
    nama_video = frame_path.parent.name
    nama_frame = frame_path.name


    # --------------------------------------------------------
    # Nama file baru
    # --------------------------------------------------------
    nama_baru = (
        f"{tanggal}_{nama_video}_{nama_frame}"
    )


    output_path = OUTPUT_DIR / nama_baru


    # --------------------------------------------------------
    # Salin frame
    # --------------------------------------------------------
    shutil.copy2(
        frame_path,
        output_path
    )


    total += 1


    # --------------------------------------------------------
    # Tampilkan progres setiap 500 frame
    # --------------------------------------------------------
    if total % 500 == 0:

        print(
            f"  {total:,} / "
            f"{len(frame_files):,} frame selesai"
        )


# ============================================================
# HASIL AKHIR
# ============================================================
print("\n" + "=" * 70)
print("PENGGABUNGAN SELESAI")
print("=" * 70)

print(
    f"Total frame digabung : "
    f"{total:,}"
)

print(
    f"Lokasi               : "
    f"{OUTPUT_DIR}"
)

print("=" * 70)