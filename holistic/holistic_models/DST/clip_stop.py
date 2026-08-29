import base64
import os
import sys

import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CLIP_MODULES_PATH = os.path.join(
    CURRENT_DIR, "../../../modules/qa/LANA/finetune_src"
)
if CLIP_MODULES_PATH not in sys.path:
    sys.path.insert(0, CLIP_MODULES_PATH)

from lana_models.clip_model import load_clip
from lana_models.tokenization_clip import SimpleTokenizer


class ClipStopMatcher:
    def __init__(self):
        self.enabled = os.environ.get("CLIP_STOP_ENABLED", "0") == "1"
        self.model = None
        self.tokenizer = None
        self.device = None
        self.feature_offsets = {}
        self.feature_file = None
        self.text_cache = {}
        if not self.enabled:
            return

        weights_path = os.environ.get("CLIP_STOP_WEIGHTS", "")
        tokenizer_path = os.environ.get("CLIP_STOP_TOKENIZER", "")
        features_path = os.environ.get("CLIP_STOP_FEATURES", "")
        if not weights_path or not tokenizer_path or not features_path:
            print("[CLIPStop] disabled: model, tokenizer, and features are required")
            self.enabled = False
            return

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = load_clip(weights_path, device=self.device, jit=False)
        self.model.eval()
        self.tokenizer = SimpleTokenizer(tokenizer_path)
        self._index_features(features_path)
        print(
            f"[CLIPStop] enabled on {self.device} with "
            f"{len(self.feature_offsets)} panorama features"
        )

    def _index_features(self, features_path):
        self.feature_file = open(features_path, "rb")
        while True:
            offset = self.feature_file.tell()
            line = self.feature_file.readline()
            if not line:
                break
            columns = line.split(b"\t", 5)
            if len(columns) >= 6:
                key = (
                    columns[0].decode("utf-8"),
                    columns[1].decode("utf-8"),
                )
                self.feature_offsets[key] = offset

    def _load_image_features(self, scan, viewpoint):
        offset = self.feature_offsets.get((scan, viewpoint))
        if offset is None:
            return None
        self.feature_file.seek(offset)
        columns = self.feature_file.readline().split(b"\t", 5)
        if len(columns) < 6:
            return None
        features = np.frombuffer(
            base64.b64decode(columns[5].strip()),
            dtype=np.float32,
        ).reshape(36, -1)
        return features / np.linalg.norm(
            features,
            axis=-1,
            keepdims=True,
        ).clip(min=1e-8)

    def _encode_text(self, text):
        if text in self.text_cache:
            return self.text_cache[text]
        encoded = self.tokenizer.encode(text)
        if len(encoded) > self.model.context_length:
            encoded = encoded[: self.model.context_length - 1]
            encoded.append(self.tokenizer.encoder["<EOS>"])
        tokens = torch.zeros(
            (1, self.model.context_length),
            dtype=torch.long,
            device=self.device,
        )
        tokens[0, : len(encoded)] = torch.tensor(encoded, device=self.device)
        with torch.inference_mode():
            text_feature = self.model.encode_text(tokens).float()
            text_feature = text_feature / text_feature.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        self.text_cache[text] = text_feature[0]
        return self.text_cache[text]

    def score(self, scan, viewpoint, text):
        if not self.enabled or not text:
            return None
        image_features = self._load_image_features(scan, viewpoint)
        if image_features is None:
            return None
        text_feature = self._encode_text(text)
        image_tensor = torch.from_numpy(image_features).to(self.device)
        with torch.inference_mode():
            score = image_tensor @ text_feature
        return float(score.max().item())

    def score_topk(self, scan, viewpoint, text, top_k=12):
        if not self.enabled or not text:
            return None
        image_features = self._load_image_features(scan, viewpoint)
        if image_features is None:
            return None
        text_feature = self._encode_text(text)
        image_tensor = torch.from_numpy(image_features).to(self.device)
        with torch.inference_mode():
            scores = image_tensor @ text_feature
            count = min(max(1, int(top_k)), scores.numel())
            score = scores.topk(count).values.mean()
        return float(score.item())
