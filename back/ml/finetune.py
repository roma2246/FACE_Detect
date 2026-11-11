"""Дообучение базовой модели на собственных снимках с веб-камеры.

Снимки сохраняет Node-бэкенд в data/custom/<класс>/*.png (48x48, серые).
Чтобы модель не «забыла» FER-2013, свои снимки смешиваются со случайной частью FER.
Результат: models/emotion_cnn_ft.pth и models/emotion_cnn_ft.onnx (бэкенд подхватывает автоматически).

Запуск:  .venv/bin/python finetune.py [--epochs 8]
Каждая строка вывода — JSON, чтобы бэкенд мог показывать прогресс в интерфейсе.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from dataset import FaceDataset, class_weights, load_arrays, load_custom
from model import CLASSES, EmotionCNN
from train import MODELS, evaluate, export_onnx, pick_device, set_seed


def log(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--fer-samples", type=int, default=6000)
    args = p.parse_args()
    set_seed(42)

    xc, yc = load_custom()
    if len(yc) < 10:
        log(type="error", message="Слишком мало своих снимков (нужно хотя бы 10).")
        raise SystemExit(1)
    log(type="info", message=f"Своих снимков: {len(yc)} " +
        str({c: int((yc == i).sum()) for i, c in enumerate(CLASSES)}))

    xtr, ytr, xte, yte = load_arrays()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ytr), size=min(args.fer_samples, len(ytr)), replace=False)

    # Свои снимки повторяем, чтобы они составляли заметную долю батчей (~30%).
    reps = max(1, int(0.3 * len(idx) / max(1, len(yc))))
    ds = ConcatDataset([FaceDataset(xtr[idx], ytr[idx], augment=True)] +
                       [FaceDataset(xc, yc, augment=True)] * reps)
    all_y = np.concatenate([ytr[idx]] + [yc] * reps)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(FaceDataset(xte, yte), 256)
    custom_loader = DataLoader(FaceDataset(xc, yc), 256)

    device = pick_device()
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS, "emotion_cnn.pth"), map_location=device))

    base_test, _, _ = evaluate(model, test_loader, device)
    base_custom, _, _ = evaluate(model, custom_loader, device)
    log(type="info", message=f"До дообучения: FER test {base_test:.3f}, свои снимки {base_custom:.3f}")

    # Замораживаем первый свёрточный блок: низкоуровневые признаки уже хорошие.
    for prm in model.features[0].parameters():
        prm.requires_grad = False
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=args.lr)
    crit = nn.CrossEntropyLoss(weight=class_weights(all_y).to(device), label_smoothing=0.05)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, loss_sum = 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            tot += len(y)
        acc_c, _, _ = evaluate(model, custom_loader, device)
        log(type="epoch", epoch=ep, epochs=args.epochs, loss=round(loss_sum / tot, 4), custom_acc=round(float(acc_c), 3))

    test_acc, _, _ = evaluate(model, test_loader, device)
    torch.save(model.state_dict(), os.path.join(MODELS, "emotion_cnn_ft.pth"))
    export_onnx(model, os.path.join(MODELS, "emotion_cnn_ft.onnx"))
    log(type="done", test_acc=round(float(test_acc), 3), base_test_acc=round(float(base_test), 3),
        custom_acc=round(float(acc_c), 3), base_custom_acc=round(float(base_custom), 3))


if __name__ == "__main__":
    main()
