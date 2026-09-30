from pathlib import Path


INPUT_DIR = Path("/Volumes/OS/all video/Frame CCTV")


total = 0
jumlah_folder = 0


print("=" * 70)
print("PEMERIKSAAN JUMLAH FRAME")
print("=" * 70)


for folder in sorted(INPUT_DIR.glob("*/*")):

    if not folder.is_dir():
        continue

    jumlah_frame = len(
        list(folder.glob("*.jpg"))
    )

    jumlah_folder += 1
    total += jumlah_frame

    print(
        f"{folder.parent.name} / "
        f"{folder.name} : "
        f"{jumlah_frame} frame"
    )


print("\n" + "=" * 70)
print("HASIL PEMERIKSAAN")
print("=" * 70)

print(f"Jumlah folder video : {jumlah_folder}")
print(f"Total frame         : {total:,}")

print("=" * 70)