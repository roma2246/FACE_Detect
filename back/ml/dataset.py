"""Загрузка данных (5 классов), разбиение train/val/test, аугментация, веса классов.

Обучение идёт на двух датасетах с HuggingFace:
  * FER+  (deanngkl/ferplus-7cls)   — картинки FER-2013 (48x48, серые) с исправленной разметкой Microsoft;
  * RAF-DB (elipaluma/RAF-DB_Kaggle) — выровненные цветные фото 100x100 «из жизни».
Для сравнения со старой моделью дополнительно загружается исходный тест FER-2013
(Aaryan333/fer2013_train_publicTest_privateTest) — он в обучении не участвует.
"""
import glob
import io
import os
import re

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from model import CLASSES

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA_DIR, "emo5.npz")

# Исходные метки -> наши 5 классов (happy, sad, angry, surprise, neutral). Disgust и Fear исключены.
FER_TO_OURS = {3: 0, 4: 1, 0: 2, 5: 3, 6: 4}      # FER-2013: 0 angry, 3 happy, 4 sad, 5 surprise, 6 neutral
FERPLUS_TO_OURS = {3: 0, 5: 1, 0: 2, 6: 3, 4: 4}  # FER+: 0 anger, 3 happiness, 4 neutral, 5 sadness, 6 surprise
RAF_TO_OURS = {4: 0, 5: 1, 6: 2, 1: 3, 7: 4}      # RAF-DB: 1 surprise, 4 happy, 5 sad, 6 angry, 7 neutral

REPOS = {
    "ferplus": "deanngkl/ferplus-7cls",
    "rafdb": "elipaluma/RAF-DB_Kaggle",
    "fer2013": "Aaryan333/fer2013_train_publicTest_privateTest",
}
FER_TRAIN_ROWS = 28709  # строки fer2013.csv: [0, 28709) — Training, дальше PublicTest и PrivateTest


def preprocess_gray48(img: Image.Image) -> np.ndarray:
    """Оттенки серого, 48x48, выравнивание гистограммы — так же, как в Node-бэкенде и в браузере."""
    img = img.convert("L")
    if img.size != (48, 48):
        img = img.resize((48, 48), Image.BILINEAR)
    img = ImageOps.equalize(img)
    return np.asarray(img, dtype=np.uint8)


def _parquet(name, prefix):
    import pandas as pd

    pattern = os.path.join(DATA_DIR, name, "data", f"{prefix}-*.parquet")
    if not glob.glob(pattern):
        from huggingface_hub import snapshot_download

        print(f"Скачиваю {REPOS[name]}...")
        snapshot_download(REPOS[name], repo_type="dataset", local_dir=os.path.join(DATA_DIR, name),
                          allow_patterns=["data/*"])
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(pattern))], ignore_index=True)


def _to_arrays(df, mapping):
    df = df[df.label.isin(mapping.keys())]
    x = np.stack([preprocess_gray48(Image.open(io.BytesIO(d["bytes"]))) for d in df.image])
    y = np.array([mapping[l] for l in df.label], dtype=np.int64)
    return x, y


def _build():
    # FER+: официальное разбиение восстанавливается по номеру строки в имени файла (fer0028709.png -> тест)
    fp = _parquet("ferplus", "train")
    row = fp.image.map(lambda d: int(re.search(r"(\d+)", d["path"]).group(1)))
    fp_tr, fp_te = _to_arrays(fp[row < FER_TRAIN_ROWS], FERPLUS_TO_OURS), _to_arrays(fp[row >= FER_TRAIN_ROWS], FERPLUS_TO_OURS)
    raf_tr, raf_te = _to_arrays(_parquet("rafdb", "train"), RAF_TO_OURS), _to_arrays(_parquet("rafdb", "test"), RAF_TO_OURS)
    import pandas as pd
    old = pd.concat([_parquet("fer2013", "publicTest"), _parquet("fer2013", "privateTest")], ignore_index=True)
    old_te = _to_arrays(old, FER_TO_OURS)
    np.savez_compressed(
        CACHE,
        xtr=np.concatenate([fp_tr[0], raf_tr[0]]), ytr=np.concatenate([fp_tr[1], raf_tr[1]]),
        x_ferplus=fp_te[0], y_ferplus=fp_te[1], x_rafdb=raf_te[0], y_rafdb=raf_te[1],
        x_fer2013=old_te[0], y_fer2013=old_te[1],
    )


def _cache():
    if not os.path.exists(CACHE):
        _build()
    return np.load(CACHE)


def load_tests():
    """Тестовые выборки по отдельности: {'ferplus': (x, y), 'rafdb': ..., 'fer2013': ...}."""
    d = _cache()
    return {k: (d[f"x_{k}"], d[f"y_{k}"]) for k in ("ferplus", "rafdb", "fer2013")}


def load_arrays():
    """Возвращает (x_train, y_train, x_test, y_test); тест = FER+ test + RAF-DB test."""
    d = _cache()
    xte = np.concatenate([d["x_ferplus"], d["x_rafdb"]])
    yte = np.concatenate([d["y_ferplus"], d["y_rafdb"]])
    return d["xtr"], d["ytr"], xte, yte


def _custom_files():
    """[(путь, класс, запись)] — имя файла: <человек>__<id записи>_<номер кадра>.png"""
    out = []
    for ci, name in enumerate(CLASSES):
        for f in sorted(glob.glob(os.path.join(DATA_DIR, "custom", name, "*.png"))):
            session = os.path.basename(f).rsplit("_", 1)[0]
            out.append((f, ci, f"{name}/{session}"))
    return out


def _stack(files):
    if not files:
        return np.zeros((0, 48, 48), np.uint8), np.zeros((0,), np.int64)
    x = np.stack([preprocess_gray48(Image.open(f)) for f, _, _ in files])
    return x, np.array([c for _, c, _ in files], dtype=np.int64)


def load_custom():
    """Все собственные снимки с веб-камеры (бэкенд сохраняет их в data/custom/<класс>/*.png)."""
    return _stack(_custom_files())


def load_custom_split(test_share=0.2):
    """Свои снимки -> (x_train, y_train, x_test, y_test).

    Соседние кадры одной записи почти одинаковы, поэтому случайное разбиение завышало бы точность.
    В тест уходит последняя часть (20%) каждой записи целиком, одним непрерывным куском.
    """
    by_session = {}
    for item in _custom_files():
        by_session.setdefault(item[2], []).append(item)
    train, test = [], []
    for items in by_session.values():
        k = int(round(len(items) * test_share)) if len(items) >= 5 else 0
        train += items[: len(items) - k]
        test += items[len(items) - k:]
    return (*_stack(train), *_stack(test))


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
