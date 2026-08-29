"""Guide-side search for the wording that best distinguishes the target.

The guide knows the target location and the environment, so rather than reciting
a detector's output it can choose which words to say: ones that fit the target
panorama and, just as importantly, do not fit anywhere else in the building.
That is what a guide does when it says "the bathroom with the skylight, not the
other one".

The search runs inside the episode the first time the navigator asks, and only
the resulting sentence is spoken.
"""

import os

import numpy as np
import torch

from holistic_models.DST.clip_stop import ClipStopMatcher


ROOM_WORDS = (
    "living room", "dining room", "family room", "utility room", "meeting room",
    "rec room", "tv room", "bedroom", "bathroom", "kitchen", "hallway",
    "entryway", "office", "stairway", "staircase", "garage", "patio",
    "washroom", "loft", "attic", "basement", "closet", "porch", "balcony",
    "lounge", "library", "spa", "gym", "laundry", "toilet", "foyer",
    "corridor", "terrace", "deck", "pool",
)

OBJECT_WORDS = (
    "arch", "archway", "armchair", "bathtub", "bed", "bench", "blinds",
    "books", "bookshelf", "cabinet", "chair", "clock", "closet", "column",
    "couch", "counter", "countertop", "curtain", "desk", "dishwasher",
    "dresser", "faucet", "fence", "fireplace", "footstool", "fountain",
    "handrail", "headboard", "lamp", "mat", "microwave", "mirror",
    "nightstand", "painting", "phone", "pillow", "plant", "planter",
    "plants", "plug", "rack", "railing", "seat", "shelf", "shelves",
    "sink", "skylight", "stair", "stairs", "stool", "stove", "table",
    "thermostat", "toilet", "towel", "towels", "trashcan", "tv", "vanity",
    "vent", "wardrobe", "washbasin", "windowframe",
)


class H14Witness:
    """An independent opinion on what is visible at the goal.

    Self-certification is worthless: if the same ViT-B/16 that picks the words
    also vouches for them, and the navigator decodes with that same encoder,
    the check shares every blind spot it is meant to catch.  This reads
    precomputed ViT-H/14 (LAION-2B) panorama features, whose weights and
    training data are unrelated to the decoder, and only vetoes words.
    """

    def __init__(self):
        self.enabled = False
        feature_path = os.environ.get("GUIDE_H14_FEATURES", "")
        text_path = os.environ.get("GUIDE_H14_TEXT", "")
        if not feature_path or not text_path:
            return
        try:
            import h5py

            self.features = h5py.File(feature_path, "r")
            payload = np.load(text_path, allow_pickle=True)
        except Exception as error:  # missing file or h5py; fall back silently
            print(f"[H14Witness] disabled: {error}", flush=True)
            return
        self.index = {
            str(word): position
            for position, word in enumerate(payload["words"].tolist())
        }
        self.vectors = payload["vectors"].astype(np.float32)
        self.enabled = True

    def matrix(self, scan, viewpoints):
        rows = []
        for viewpoint in viewpoints:
            key = f"{scan}_{viewpoint}"
            if key not in self.features:
                continue
            views = np.asarray(self.features[key], dtype=np.float32)
            norms = np.linalg.norm(views, axis=-1, keepdims=True)
            rows.append(views / np.maximum(norms, 1e-8))
        if not rows:
            return None
        return np.stack(rows)

    def confirm(self, goal_matrix, other_matrix, words, top_k, percentile):
        """Words whose bare mention beats the p-th percentile of the building."""
        known = [word for word in words if word in self.index]
        if not known or goal_matrix is None or other_matrix is None:
            return list(words)
        vectors = self.vectors[[self.index[word] for word in known]]

        def per_pano(matrix):
            panos, views, dims = matrix.shape
            flat = matrix.reshape(panos * views, dims) @ vectors.T
            scores = flat.reshape(panos, views, len(known))
            count = min(max(1, top_k), views)
            ordered = np.sort(scores, axis=1)[:, ::-1][:, :count, :]
            return ordered.mean(axis=1).T

        goal_score = per_pano(goal_matrix).max(axis=-1)
        threshold = np.quantile(
            per_pano(other_matrix), percentile / 100.0, axis=-1
        )
        return [
            word
            for word, keep in zip(known, (goal_score > threshold).tolist())
            if keep
        ]


class GuideCaptionSearch:
    def __init__(self, shortest_distances=None):
        self.enabled = os.environ.get("GUIDE_CAPTION_SEARCH", "0") == "1"
        self.contrast = float(os.environ.get("GUIDE_CAPTION_CONTRAST", "0.75"))
        self.object_slots = int(os.environ.get("GUIDE_CAPTION_OBJECTS", "4"))
        self.top_k = int(os.environ.get("GUIDE_CAPTION_TOPK", "16"))
        # "sentence" keeps the utterance readable.  The template has to sit
        # inside the search, not wrap its output: the search optimises the exact
        # string it will send, so phrasing it afterwards would ship a string
        # nobody scored.
        self.template = os.environ.get("GUIDE_CAPTION_TEMPLATE", "bag")
        # Above zero, only words CLIP can confirm on the goal may be spoken.
        self.truth_percentile = float(
            os.environ.get("GUIDE_CAPTION_TRUTH_PCT", "0")
        )
        # Sampling distractors keeps the per-episode cost bounded on the largest
        # scans; the worst-case competitor is what matters, not every node.
        self.max_distractors = int(
            os.environ.get("GUIDE_CAPTION_MAX_DISTRACTORS", "400")
        )
        self.shortest_distances = shortest_distances
        self.verify_mode = os.environ.get("GUIDE_CAPTION_VERIFY", "")
        self.witness = (
            H14Witness()
            if self.enabled
            and self.truth_percentile > 0
            and self.verify_mode in {"h14", "both"}
            else None
        )
        self.matcher = ClipStopMatcher() if self.enabled else None
        if self.matcher is not None and not self.matcher.enabled:
            self.enabled = False
        self.cache = {}
        self.searches = 0

    def _pano_matrix(self, scan, viewpoints):
        rows = []
        for viewpoint in viewpoints:
            features = self.matcher._load_image_features(scan, viewpoint)
            if features is not None:
                rows.append(features)
        if not rows:
            return None
        return torch.from_numpy(np.stack(rows)).to(self.matcher.device)

    def _encode(self, texts):
        return torch.stack(
            [self.matcher._encode_text(text) for text in texts]
        )

    def _per_pano(self, matrix, vectors):
        """Top-k mean similarity of each caption against each panorama."""
        similarity = torch.einsum("nvd,md->mnv", matrix, vectors)
        count = min(max(1, self.top_k), similarity.shape[-1])
        return similarity.topk(count, dim=-1).values.mean(dim=-1)

    def _best_topk(self, matrix, vectors):
        """Highest top-k mean similarity over the panoramas, per caption."""
        return self._per_pano(matrix, vectors).max(dim=-1).values

    def _scan_nodes(self, scan, goals):
        if not self.shortest_distances:
            return []
        try:
            nodes = list(self.shortest_distances[scan].keys())
        except (KeyError, TypeError):
            return []
        excluded = set(goals)
        others = [node for node in nodes if node not in excluded]
        if len(others) > self.max_distractors:
            step = len(others) / float(self.max_distractors)
            others = [
                others[int(index * step)] for index in range(self.max_distractors)
            ]
        return others

    @staticmethod
    def _article(word):
        """Readable English: plurals take no article, vowels take "an"."""
        if not word:
            return word
        if word.endswith("s") and not word.endswith("ss"):
            return word
        return f"an {word}" if word[0] in "aeiou" else f"a {word}"

    def _phrase(self, words):
        """The utterance for a chosen word list."""
        if self.template != "sentence" or not words:
            return ", ".join(words)
        target = self._article(words[0])
        room = words[1] if len(words) > 1 else ""
        objects = [word for word in words[2:] if word]
        head = f"{target} in {self._article(room)}" if room else target
        if not objects:
            return head
        spoken = [self._article(word) for word in objects]
        if len(spoken) == 1:
            tail = spoken[0]
        else:
            tail = f"{', '.join(spoken[:-1])} and {spoken[-1]}"
        return f"{head} with {tail}"

    def describe(self, instr_id, scan, goals, target):
        """Words for this target, computed once per episode."""
        if not self.enabled or not goals:
            return ""
        key = str(instr_id)
        if key in self.cache:
            return self.cache[key]

        goal_nodes = list(goals)
        other_nodes = self._scan_nodes(scan, goals)
        goal_matrix = self._pano_matrix(scan, goal_nodes)
        other_matrix = self._pano_matrix(scan, other_nodes)
        if goal_matrix is None or other_matrix is None:
            self.cache[key] = ""
            return ""

        target = str(target or "").strip() or "target"

        def visible(words):
            """Words a bare mention of which scores unusually high on the goal.

            A word only earns the right to be spoken if the goal panorama beats
            the p-th percentile of the rest of the building on that word alone,
            so the guide cannot name something it cannot see.
            """
            texts = [f"a photo of a {word}" for word in words]
            with torch.inference_mode():
                vectors = self._encode(texts)
                goal_score = self._per_pano(goal_matrix, vectors).max(dim=-1).values
                other_scores = self._per_pano(other_matrix, vectors)
                threshold = torch.quantile(
                    other_scores, self.truth_percentile / 100.0, dim=-1
                )
            return [
                word
                for word, keep in zip(words, (goal_score > threshold).tolist())
                if keep
            ]

        if self.witness is not None and self.witness.enabled:
            witness_goal = self.witness.matrix(scan, goal_nodes)
            witness_other = self.witness.matrix(scan, other_nodes)
            self_certified = visible

            def visible(words):
                # A percentile test on one encoder still passes roughly
                # (100 - p)% of absent words by chance.  Requiring both
                # encoders to agree multiplies those two error rates.
                confirmed = self.witness.confirm(
                    witness_goal,
                    witness_other,
                    words,
                    self.top_k,
                    self.truth_percentile,
                )
                if self.verify_mode != "both":
                    return confirmed
                allowed = set(self_certified(words))
                return [word for word in confirmed if word in allowed]

        rooms = list(ROOM_WORDS)
        objects = list(OBJECT_WORDS)
        if self.truth_percentile > 0:
            rooms = visible(rooms) or list(ROOM_WORDS)
            objects = visible(objects)

        def pick(prefix, options):
            texts = [self._phrase(prefix + [option]) for option in options]
            with torch.inference_mode():
                vectors = self._encode(texts)
                goal_score = self._best_topk(goal_matrix, vectors)
                other_score = self._best_topk(other_matrix, vectors)
            value = goal_score - self.contrast * other_score
            return options[int(value.argmax())]

        chosen = [target, pick([target], rooms)]
        for _ in range(self.object_slots):
            pool = [word for word in objects if word not in chosen]
            if not pool:
                break
            chosen.append(pick(chosen, pool))

        caption = self._phrase(chosen)
        self.cache[key] = caption
        self.searches += 1
        if os.environ.get("GUIDE_CAPTION_DEBUG") == "1":
            print(f"[GuideCaption] {key}: {caption}", flush=True)
        return caption
