# FAN — DialNav Challenge, ECCV 2026 EAD

Team FAN (Five-Ages-Navigator) with **DRG**, Distinctive Referring Guidance.

The navigator in the released baseline reaches the goal viewpoint in over 95% of
episodes and stops on it in about 21%: it arrives and fails to recognise where it
is. This entry reallocates the guide's language budget from routing to
*reference*. The guide, which legitimately observes the target, says what
distinguishes that place from everywhere else in the building, and the navigator
turns stopping into retrieval over the panoramas it has itself visited.

No weights are trained or fine-tuned. The released RAINbow checkpoints (DST
navigator, LANA questioner and answerer, GTL localizer) are used unchanged.

## Method

- **Retrospective grounding stop.** The navigator explores under a step budget
  with a reserve, scores every panorama it has personally visited against the
  guide's utterance with CLIP ViT-B/16 (top-16 view pooling), walks back to the
  best match along its *observed* graph, and stops there. It never receives the
  house graph, a waypoint, or a guide-selected action.
- **Distinctive referring guidance.** Over a fixed vocabulary of 35 room and 63
  object nouns, the guide greedily selects the expression that maximises
  `max_goal sim − 0.75 · max_distractor sim`, the classical referring-expression
  objective, instantiated with a vision–language model.
- **Independent visual grounding gate.** A noun is admissible only if CLIP
  ViT-H/14 (LAION-2B) confirms it at the goal above the 80th percentile of the
  building. That encoder is unrelated to the ViT-B/16 the navigator decodes
  with; certification by the navigator's own encoder would be circular.
- **One natural-language channel.** The nouns are spoken as a sentence inside
  the answer ("… You should see a sink in a bedroom with a washbasin, an
  archway, a toilet and a vanity.") and parsed back out of it. No identifier,
  image, or feature vector crosses the channel.

## Installation

```bash
conda create -n fan python=3.10 && conda activate fan
pip install -r requirements.txt
```

Also required: the Matterport3D Simulator built for Python 3.10, and PyTorch
2.6.0 with CUDA 12.4.

## Data

Download the official DialNav release and point `DATA_ROOT` at it. Expected
layout:

```
$DATA_ROOT/dataset/checkpoints/{nav_rainbow,q_rainbow,a_rainbow}   # official
$DATA_ROOT/dataset/checkpoints/loc_rainbow.pth                     # official
$DATA_ROOT/dataset/features/CLIP-ViT-B-16-views.tsv                # official
$DATA_ROOT/dataset/features/clip_vit-h14_mp3d_original.hdf5        # official
$DATA_ROOT/dataset/connectivity/                                   # official
$DATA_ROOT/dataset/RAIN_holistic/{val_seen,val_unseen,test}.json    # official
$DATA_ROOT/cache/ViT-B-16.pt                                       # public CLIP
```

No additional dataset is used and no preprocessing is applied to the official
files. `$DATA_ROOT/cache/ViT-B-16.pt` is the public OpenAI CLIP ViT-B/16
release, used only to encode text.

`vocab/h14_vocab_text.npz` ships with this repository: the 98 vocabulary words
embedded with CLIP ViT-H/14. It is a model constant carrying no episode, scene,
or goal information, and `scripts/make_h14_text.py` regenerates it (that script
alone needs `open_clip_torch`).

## Training

None. Every checkpoint is used as released.

## Evaluation

```bash
DATA_ROOT=/path/to/data bash scripts/run_delivered.sh val_seen
DATA_ROOT=/path/to/data bash scripts/run_delivered.sh val_unseen
DATA_ROOT=/path/to/data bash scripts/run_delivered.sh test
```

Eight shards on eight GPUs, batch size 8, seed 0, greedy decoding. The run is
deterministic: repeating it reproduces the metrics below exactly, to every digit.

| Split | Episodes | SR (%) | DTC | Oracle SR (%) |
|---|---|---|---|---|
| val_seen | 91 | 81.32 | 1.53 | 96.70 |
| val_unseen | 241 | 81.33 | 1.80 | 94.19 |
| test | 285 | 82.81 | 1.88 | 96.49 |

The challenge score is `E(D) × Success`; on the validation splits it is 0.8128
and 0.8075. The test score is computed by the organizers.

## Configuration

`configs/delivered.env` holds the submitted configuration in full — the only
configuration in this repository. The two parameters of DRG are the length of
the referring expression (4 object nouns) and the strictness of the grounding
gate (p = 80); both are selected on `val_unseen`, where p = 80 scores 0.779
against 0.729 for p = 90, and the sentence form scores 0.807 against 0.779 for a
bare noun list. The selected setting is applied unchanged to every split.

## Compliance

Communication is natural language only. The navigator plans exclusively on the
graph it has itself observed and physically returns to its chosen viewpoint
within the episode, so no answer is revised after the episode ends. The guide
uses only what it is entitled to observe: the target location and its own
building. No commercial API is called. No component is trained or fine-tuned,
and no annotation, dialog, or trajectory of the test split informs any part of
the system.

## License

See `LICENSE` and `NOTICE`. Built on the official RAINbow release.
