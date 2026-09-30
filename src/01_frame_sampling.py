import cv2
from pathlib import Path


# ============================================================
# FOLDER SUMBER VIDEO DI SSD
# ============================================================
INPUT_DIR = Path("/Volumes/OS/all video/Input rekaman cctv")


# ============================================================
# FOLDER HASIL FRAME DI SSD
# ============================================================
OUTPUT_DIR = Path("/Volumes/OS/all video/Frame CCTV")


# ============================================================
# INTERVAL SAMPLING
# 1 FRAME SETIAP 5 DETIK
# ============================================================
INTERVAL = 5


# ============================================================
# FORMAT VIDEO
# ============================================================
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".wmv",
    ".flv",
}


# ============================================================
# CARI SEMUA VIDEO
# ============================================================
video_files = sorted(
    [
        path
        for path in INPUT_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and not path.name.startswith("._")
        )
    ]
)


print("=" * 70)
print("FRAME SAMPLING CCTV")
print("=" * 70)

print(f"Jumlah video ditemukan : {len(video_files)}")
print(f"Interval sampling      : 1 frame / {INTERVAL} detik")
print(f"Folder output          : {OUTPUT_DIR}")


# ============================================================
# BUAT FOLDER OUTPUT
# ============================================================
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TOTAL FRAME
# ============================================================
total_frames = 0


# ============================================================
# PROSES VIDEO
# ============================================================
for nomor, video_path in enumerate(video_files, start=1):

    print(
        f"\n[{nomor}/{len(video_files)}] "
        f"{video_path.parent.name} / {video_path.name}"
    )


    # --------------------------------------------------------
    # BUKA VIDEO
    # --------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))


    if not cap.isOpened():

        print("  Gagal membuka video.")

        continue


    # --------------------------------------------------------
    # INFORMASI VIDEO
    # --------------------------------------------------------
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )


    if fps <= 0:

        print("  FPS tidak terbaca.")

        cap.release()

        continue


    # --------------------------------------------------------
    # HITUNG DURASI
    # --------------------------------------------------------
    duration = frame_count / fps


    # --------------------------------------------------------
    # HITUNG INTERVAL FRAME
    # --------------------------------------------------------
    interval_frame = int(
        round(fps * INTERVAL)
    )


    # --------------------------------------------------------
    # FOLDER OUTPUT
    # --------------------------------------------------------
    tanggal = video_path.parent.name
    nama_video = video_path.stem

    output_folder = (
        OUTPUT_DIR
        / tanggal
        / nama_video
    )


    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # MULAI MEMBACA FRAME
    # --------------------------------------------------------
    frame_index = 0
    saved_count = 0


    while True:

        success, frame = cap.read()


        if not success:

            break


        # ----------------------------------------------------
        # SIMPAN 1 FRAME SETIAP 5 DETIK
        # ----------------------------------------------------
        if frame_index % interval_frame == 0:

            output_file = (
                output_folder
                / f"frame_{saved_count + 1:06d}.jpg"
            )


            cv2.imwrite(
                str(output_file),
                frame
            )


            saved_count += 1
            total_frames += 1


        frame_index += 1


    cap.release()


    # --------------------------------------------------------
    # INFORMASI HASIL VIDEO
    # --------------------------------------------------------
    print(
        f"  Durasi       : "
        f"{duration / 60:.2f} menit"
    )

    print(
        f"  FPS          : "
        f"{fps:.2f}"
    )

    print(
        f"  Frame hasil  : "
        f"{saved_count}"
    )


# ============================================================
# HASIL AKHIR
# ============================================================
print("\n" + "=" * 70)
print("FRAME SAMPLING SELESAI")
print("=" * 70)

print(
    f"Total video        : {len(video_files)}"
)

print(
    f"Total frame        : {total_frames:,}"
)

print(
    f"Lokasi frame       : {OUTPUT_DIR}"
)

print("=" * 70)