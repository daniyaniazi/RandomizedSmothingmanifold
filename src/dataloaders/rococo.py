"""RoCOCO dataset loader for CLIP image-text retrieval experiments.

Annotation structure (RoCOCO protocol):
  coco_karpathy_test.json: 5 captions per image — all GT
  danger / same_concept / diff_concept / rand_voca:
      10 captions per image — first 5 = GT, last 5 = adversarial

Retrieval pool for adversarial annotations:
  ALL 10 captions mixed → text pool
  img2txt[i]  = indices of GT captions (0..4 per image) → ground truth
  wrongtext   = indices of adversarial captions (5..9 per image) → RSMS numerator
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class RoCoCoImageDataset(Dataset):
    """Minimal dataset for encoding images with CLIP preprocess."""

    def __init__(self, image_paths: List[str], preprocess):
        self.paths      = image_paths
        self.preprocess = preprocess

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.preprocess(img), i
        except Exception:
            return self.preprocess(Image.new("RGB", (224, 224))), i


ANN_STEMS = ["coco_karpathy_test", "danger", "same_concept", "diff_concept", "rand_voca"]


@dataclass
class RoCoCoSample:
    image_id: str
    image_path: str
    gt_captions: List[str]                    # from coco_karpathy_test
    adv_captions: Dict[str, List[str]] = field(default_factory=dict)
    # Full 10-caption list per adversarial annotation (first 5 GT + last 5 adv)
    all_captions: Dict[str, List[str]] = field(default_factory=dict)


def _load_annotation(ann_path: Path) -> Dict[str, List[str]]:
    """Load annotation JSON → {image_id: [caption, ...]}."""
    data = json.loads(ann_path.read_text())
    result: Dict[str, List[str]] = {}
    for item in data:
        img_id = item["image"]
        caps   = item.get("caption", [])
        if isinstance(caps, str):
            caps = [caps]
        result[img_id] = caps
    return result


class RoCoCoDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        annotation_files: Optional[List[str]] = None,
    ):
        self.image_dir = Path(image_dir)
        self.ann_dir   = Path(annotation_dir)

        if annotation_files is None:
            annotation_files = [f"{s}.json" for s in ANN_STEMS]

        ann_maps: Dict[str, Dict[str, List[str]]] = {}
        for fname in annotation_files:
            path = self.ann_dir / fname
            if not path.exists():
                print(f"  WARNING: annotation not found: {path}")
                continue
            stem = Path(fname).stem
            ann_maps[stem] = _load_annotation(path)

        gt_map    = ann_maps.get("coco_karpathy_test", {})
        image_ids = sorted(gt_map.keys())

        self.samples: List[RoCoCoSample] = []
        self.ann_stems = [Path(f).stem for f in annotation_files]

        for img_id in image_ids:
            abs_path = str(self.image_dir / img_id)
            if not Path(abs_path).exists():
                continue
            adv: Dict[str, List[str]] = {}
            all_caps: Dict[str, List[str]] = {}
            for stem, amap in ann_maps.items():
                if stem != "coco_karpathy_test" and img_id in amap:
                    full = amap[img_id]         # 10 captions: first 5 GT + last 5 adv
                    n_gt = min(5, len(full))
                    adv[stem]      = full[n_gt:]  # adversarial only
                    all_caps[stem] = full          # all 10
            self.samples.append(RoCoCoSample(
                image_id=img_id,
                image_path=abs_path,
                gt_captions=gt_map.get(img_id, []),
                adv_captions=adv,
                all_captions=all_caps,
            ))

        print(f"RoCoCoDataset: {len(self.samples):,} images  anns={list(ann_maps.keys())}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

    @property
    def image_paths(self): return [s.image_path for s in self.samples]

    @property
    def image_ids(self): return [s.image_id for s in self.samples]

    def get_retrieval_pool(
        self, ann_stem: str
    ) -> Tuple[List[str], Dict[int, List[int]], List[int]]:
        """Build retrieval text pool following the RoCOCO protocol.

        For coco_karpathy_test:
            pool = 5 GT captions per image
            img2txt[i] = GT indices, wrongtext = []

        For danger / same / diff / rand:
            pool = 10 captions per image (first 5 GT + last 5 adversarial)
            img2txt[i] = GT indices (0..4 per image)
            wrongtext  = adversarial indices (5..9 per image)  → RSMS numerator

        Returns:
            captions:  flat list of all captions
            img2txt:   image_idx → list of GT caption indices
            wrongtext: list of adversarial caption indices
        """
        captions:  List[str]            = []
        img2txt:   Dict[int, List[int]] = {}
        wrongtext: List[int]            = []

        for i, s in enumerate(self.samples):
            if ann_stem == "coco_karpathy_test":
                for cap in s.gt_captions:
                    idx = len(captions)
                    img2txt.setdefault(i, []).append(idx)
                    captions.append(cap)
            else:
                full  = s.all_captions.get(ann_stem, [])
                n_gt  = min(5, len(full))
                for j, cap in enumerate(full):
                    idx = len(captions)
                    captions.append(cap)
                    if j < n_gt:
                        img2txt.setdefault(i, []).append(idx)   # GT
                    else:
                        wrongtext.append(idx)                     # adversarial

        return captions, img2txt, wrongtext
