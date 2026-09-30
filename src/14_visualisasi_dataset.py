import os
import cv2
import matplotlib.pyplot as plt


# ============================================================
# KONFIGURASI
# ============================================================

DATASET_PATH = "/Volumes/OS/all video/Dataset CCTV"

CLASS_NAMES = [
    "Pelanggan Datang",
    "Pelanggan Menunggu",
    "Pelanggan Dilayani"
]

JUMLAH_CONTOH = 10


# ============================================================
# MEMBACA 10 GAMBAR PER KELAS
# ============================================================

data = {}

for class_name in CLASS_NAMES:

    class_path = os.path.join(DATASET_PATH, class_name)

    files = [
        f for f in os.listdir(class_path)
        if not f.startswith("._")
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files = sorted(files)[:JUMLAH_CONTOH]

    images = []

    for file_name in files:

        image_path = os.path.join(class_path, file_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images.append((file_name, image))

    data[class_name] = images


# ============================================================
# TAMPILKAN
# ============================================================

fig, axes = plt.subplots(
    3,
    JUMLAH_CONTOH,
    figsize=(20, 7)
)

for row, class_name in enumerate(CLASS_NAMES):

    images = data[class_name]

    for col in range(JUMLAH_CONTOH):

        ax = axes[row, col]

        if col < len(images):

            file_name, image = images[col]

            ax.imshow(image)
            ax.set_title(
                f"{class_name}\n{file_name}",
                fontsize=7
            )

        ax.axis("off")


plt.tight_layout()

# Simpan hasil
OUTPUT_PATH = "hasil/visualisasi_dataset.jpg"

os.makedirs("hasil", exist_ok=True)

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)
print(f"Hasil disimpan di:")
print(OUTPUT_PATH)