"""Экспорт модели для браузера: ONNX с двумя выходами — логиты и 128 признаков
предпоследнего слоя. По признакам браузер может сам дообучить последний слой
(режим без бэкенда, например на Vercel).

    .venv/bin/python export_browser.py
"""
import json
import os

import torch
import torch.nn as nn

from model import EmotionCNN

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


class WithFeatures(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        c = self.model.classifier
        feats = c[2](c[1](c[0](self.model.features(x))))  # Flatten -> Linear -> ReLU
        return c[4](feats), feats  # Dropout в режиме eval ничего не делает


def export_browser(model, path):
    wrapped = WithFeatures(model).eval()
    dummy = torch.zeros(1, 1, 48, 48)
    torch.onnx.export(wrapped, dummy, path, input_names=["input"], output_names=["logits", "features"],
                      dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}, "features": {0: "n"}},
                      opset_version=17, dynamo=False)


if __name__ == "__main__":
    state = torch.load(os.path.join(MODELS, "emotion_cnn.pth"), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    net = EmotionCNN()
    net.load_state_dict(state)
    out = os.path.join(MODELS, "emotion_cnn_web.onnx")
    export_browser(net, out)

    # Веса последнего слоя (128 -> 5): с них браузер начинает дообучение
    head = net.classifier[4]
    with open(os.path.join(MODELS, "emotion_head.json"), "w") as f:
        json.dump({"w": [round(v, 6) for v in head.weight.detach().flatten().tolist()],
                   "b": [round(v, 6) for v in head.bias.detach().tolist()]}, f)
    print("saved", out, "and emotion_head.json")
