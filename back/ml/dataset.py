"""Загрузка FER-2013 (5 классов), разбиение train/val/test, аугментация, веса классов.

Датасет берётся из HuggingFace (Aaryan333/fer2013_train_publicTest_privateTest) —
это оригинальный FER-2013 с официальным разбиением Training / PublicTest / PrivateTest.
"""
import glob
import io
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from model import CLASSES

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
HF_REPO = "Aaryan333/fer2013_train_publicTest_privateTest"
CACHE = os.path.join(DATA_DIR, "fer5.npz")

# Исходные метки FER-2013 -> наши 5 классов. Disgust(1) и Fear(2) исключены.
FER_TO_OURS = {3: 0, 4: 1, 0: 2, 5: 3, 6: 4}  # happy, sad, angry, surprise, neutral


def preprocess_gray48(img: Image.Image) -> np.ndarray:
    """Оттенки серого, 48x48, выравнивание гистограммы — так же, как в Node-бэкенде."""
    img = img.convert("L")
    if img.size != (48, 48):
        img = img.resize((48, 48), Image.BILINEAR)
    img = ImageOps.equalize(img)
    return np.asarray(img, dtype=np.uint8)


def _download():
    from huggingface_hub import snapshot_download

    snapshot_download(HF_REPO, repo_type="dataset", local_dir=os.path.join(DATA_DIR, "fer2013"),
                      allow_patterns=["data/*"])


def _load_split(prefix):
    import pandas as pd

    files = glob.glob(os.path.join(DATA_DIR, "fer2013", "data", f"{prefix}-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df[df.label.isin(FER_TO_OURS.keys())]
    x = np.stack([preprocess_gray48(Image.open(io.BytesIO(d["bytes"]))) for d in df.image])
    y = np.array([FER_TO_OURS[l] for l in df.label], dtype=np.int64)
    return x, y


def load_arrays():
    """Возвращает (x_train, y_train, x_test, y_test); кэширует результат в .npz."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    if not glob.glob(os.path.join(DATA_DIR, "fer2013", "data", "*.parquet")):
        print("Скачиваю FER-2013 с HuggingFace...")
        _download()
    xtr, ytr = _load_split("train")
    x1, y1 = _load_split("publicTest")
    x2, y2 = _load_split("privateTest")
    xte, yte = np.concatenate([x1, x2]), np.concatenate([y1, y2])
    np.savez_compressed(CACHE, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte


def load_custom():
    """Собственные снимки с веб-камеры (сохраняются бэкендом в data/custom/<класс>/*.png)."""
    xs, ys = [], []
    for ci, name in enumerate(CLASSES):
        for f in glob.glob(os.path.join(DATA_DIR, "custom", name, "*.png")):
            xs.append(preprocess_gray48(Image.open(f)))
            ys.append(ci)
    if not xs:
        return np.zeros((0, 48, 48), np.uint8), np.zeros((0,), np.int64)
    return np.stack(xs), np.array(ys, dtype=np.int64)


train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.25, contrast=0.25),
])


class FaceDataset(Dataset):
    def __init__(self, x, y, augment=False):
        self.x = torch.from_numpy(x).unsqueeze(1)  # N,1,48,48 uint8
        self.y = torch.from_numpy(y)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i]
        if self.augment:
            img = train_tf(img)
        return img.float() / 255.0, self.y[i]  # нормализация к [0, 1]


def class_weights(y):
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    w = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(w)


def get_loaders(batch_size=64, val_size=0.12, seed=42):
    xtr, ytr, xte, yte = load_arrays()
    xtr, xva, ytr, yva = train_test_split(xtr, ytr, test_size=val_size, stratify=ytr, random_state=seed)
    g = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(FaceDataset(xtr, ytr, augment=True), batch_size, shuffle=True,
                            num_workers=4, persistent_workers=True, generator=g),
        "val": DataLoader(FaceDataset(xva, yva), 256),
        "test": DataLoader(FaceDataset(xte, yte), 256),
    }
    return loaders, class_weights(ytr)
