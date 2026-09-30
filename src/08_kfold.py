from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# ============================================================
# PENGATURAN
# ============================================================

DATASET_DIR = Path("/Volumes/OS/all video/Dataset CCTV")

KELAS = [
    "Pelanggan Datang",
    "Pelanggan Menunggu",
    "Pelanggan Dilayani"
]

N_SPLITS = 5
RANDOM_STATE = 42

# ============================================================
# MEMBACA DATASET
# ============================================================

files = []
labels = []

for kelas in KELAS:
    folder = DATASET_DIR / kelas

    for file in folder.glob("*.jpg"):
        if "._" in file.name:
            continue

        files.append(str(file))
        labels.append(kelas)

print("=" * 65)
print("STRATIFIED K-FOLD CROSS-VALIDATION")
print("=" * 65)

print(f"Total citra : {len(files)}")
print(f"Jumlah fold : {N_SPLITS}")
print(f"Random state: {RANDOM_STATE}")

# ============================================================
# STRATIFIED K-FOLD
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

for fold, (train_idx, val_idx) in enumerate(
    skf.split(files, labels), start=1
):

    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    train_count = Counter(train_labels)
    val_count = Counter(val_labels)

    print("\n" + "-" * 65)
    print(f"FOLD {fold}")
    print("-" * 65)

    print(f"Training : {len(train_idx)} citra")
    print(f"Validasi : {len(val_idx)} citra")

    print("\nDistribusi Training:")
    for kelas in KELAS:
        print(f"  {kelas:<25}: {train_count[kelas]}")

    print("\nDistribusi Validasi:")
    for kelas in KELAS:
        print(f"  {kelas:<25}: {val_count[kelas]}")

print("\n" + "=" * 65)
print("K-FOLD SELESAI")
print("=" * 65)