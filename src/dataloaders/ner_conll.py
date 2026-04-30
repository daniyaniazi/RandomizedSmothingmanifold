"""NER dataloaders for CoNLL-style token classification tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from src.configs import DataloaderConfig, DatasetConfig


@dataclass
class NERDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    tokenizer_name: str


def _tokenize_and_align_labels(batch, tokenizer, text_field: str, label_field: str, max_length: int):
    tokenized = tokenizer(
        batch[text_field],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )

    all_labels = []
    for i, labels in enumerate(batch[label_field]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def build_conll_dataloaders(
    dataset_cfg: DatasetConfig,
    loader_cfg: DataloaderConfig,
    encoder_name: str,
) -> NERDataBundle:
    dataset: DatasetDict = load_dataset(dataset_cfg.name)

    label_feature = dataset[dataset_cfg.split_train].features[dataset_cfg.label_field]
    label_names: List[str] = label_feature.feature.names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    processed = dataset.map(
        lambda b: _tokenize_and_align_labels(
            b,
            tokenizer,
            dataset_cfg.text_field,
            dataset_cfg.label_field,
            dataset_cfg.max_length,
        ),
        batched=True,
        remove_columns=dataset[dataset_cfg.split_train].column_names,
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def _loader(split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            processed[split],
            batch_size=loader_cfg.batch_size,
            shuffle=shuffle,
            num_workers=dataset_cfg.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

    return NERDataBundle(
        train_loader=_loader(dataset_cfg.split_train, loader_cfg.shuffle_train),
        val_loader=_loader(dataset_cfg.split_val, False),
        test_loader=_loader(dataset_cfg.split_test, False),
        id2label=id2label,
        label2id=label2id,
        tokenizer_name=encoder_name,
    )
