"""Обучение EmotionCNN на FER+ и RAF-DB (5 классов) + оценка, графики, матрица ошибок, экспорт в ONNX.

Запуск:  .venv/bin/python train.py [--epochs 40]
"""
import argparse
import json
import os
import random

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from dataset import FaceDataset, get_loaders, load_tests
from model import CLASSES, EmotionCNN

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "..", "models")
REPORTS = os.path.join(ROOT, "reports")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return (preds == labels).mean(), preds, labels


def train_epochs(model, loaders, weights, device, epochs, lr, patience, ckpt_path):
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    hist = {"loss": [], "train_acc": [], "val_acc": []}
    best, bad = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(y)
        val_acc, _, _ = evaluate(model, loaders["val"], device)
        sched.step(val_acc)
        hist["loss"].append(loss_sum / tot)
        hist["train_acc"].append(correct / tot)
        hist["val_acc"].append(float(val_acc))
        mark = ""
        if val_acc > best:
            best, bad = val_acc, 0
            torch.save(model.state_dict(), ckpt_path)
            mark = "  * сохранено"
        else:
            bad += 1
        print(f"эпоха {ep:2d}  loss {hist['loss'][-1]:.3f}  train {hist['train_acc'][-1]:.3f}  "
              f"val {val_acc:.3f}  lr {opt.param_groups[0]['lr']:.1e}{mark}", flush=True)
        if bad >= patience:
            print("Ранняя остановка.")
            break
    return hist, best


def export_onnx(model, path):
    model = model.cpu().eval()
    dummy = torch.zeros(1, 1, 48, 48)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    print("ONNX:", os.path.abspath(path))


def save_reports(hist, preds, labels, prefix=""):
    os.makedirs(REPORTS, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["loss"]); ax[0].set_title("Train loss"); ax[0].set_xlabel("эпоха")
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].set_xlabel("эпоха")
    fig.tight_layout(); fig.savefig(os.path.join(REPORTS, f"{prefix}curves.png"), dpi=120)
    cm = confusion_matrix(labels, preds, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout(); fig.savefig(os.path.join(REPORTS, f"{prefix}confusion_matrix.png"), dpi=120)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()

    set_seed(42)
    os.makedirs(MODELS, exist_ok=True)
    device = pick_device()
    print("Устройство:", device)
    loaders, weights = get_loaders()
    model = EmotionCNN().to(device)
    ckpt = os.path.join(MODELS, "emotion_cnn.pth")
    hist, best = train_epochs(model, loaders, weights, device, args.epochs, args.lr, args.patience, ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    acc, preds, labels = evaluate(model, loaders["test"], device)
    report = classification_report(labels, preds, target_names=CLASSES, digits=3)
    print(f"\nЛучшая val acc: {best:.3f}\nTEST accuracy: {acc:.3f}\n{report}")
    save_reports(hist, preds, labels)
    # Точность по каждой тестовой выборке отдельно (fer2013 — старая разметка, для сравнения с прежней моделью)
    per_set = {}
    for name, (x, y) in load_tests().items():
        a, _, _ = evaluate(model, torch.utils.data.DataLoader(FaceDataset(x, y), 256), device)
        per_set[name] = float(a)
        print(f"  {name:8s} test acc: {a:.3f}  (n={len(y)})")
    with open(os.path.join(REPORTS, "metrics.json"), "w") as f:
        json.dump({"val_acc": float(best), "test_acc": float(acc), "test_sets": per_set,
                   "report": classification_report(labels, preds, target_names=CLASSES, output_dict=True)},
                  f, indent=2)
    export_onnx(model, os.path.join(MODELS, "emotion_cnn.onnx"))


if __name__ == "__main__":
    main()
