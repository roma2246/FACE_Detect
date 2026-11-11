"""Лёгкая CNN для классификации эмоций: вход 1x48x48, выход 5 классов (логиты)."""
import torch.nn as nn

# Порядок классов фиксирован — он же используется в Node-бэкенде и во фронте.
CLASSES = ["happy", "sad", "angry", "surprise", "neutral"]


def conv_block(c_in, c_out):
    # Два свёрточных слоя в блоке дают заметный прирост точности на FER-2013
    # при сохранении скорости (модель ~1.2M параметров, <1 мс на CPU).
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
    )


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        # 3 блока: 32 -> 64 -> 128 фильтров; 48 -> 24 -> 12 -> 6
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
