from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class ImageToTextDataset(Dataset):
    dataset: pd.DataFrame
    vocab: list[str]
    max_visual_length: int
    max_text_length: int | None = None

    def __len__(self) -> int:
        return len(self.dataset)

    def create_visual_tokens(self, visual_tokens: str) -> list[int]:
        visual_token_ids = []
        for tokens in visual_tokens.split("\t"):
            token_ids = [int(i) + len(self.vocab) + 2 for i in tokens.split()]
            visual_token_ids.extend([len(self.vocab)] + token_ids)

        num_paddings = self.max_visual_length - len(visual_token_ids)
        return visual_token_ids + [len(self.vocab) + 1] * num_paddings

    def create_text_tokens(self, text: str | None) -> list[int]:
        if text is None:
            return [self.vocab.index("<s>")]

        tokens = [self.vocab.index(c) for c in text[: self.max_text_length - 2]]
        tokens = [self.vocab.index("<s>")] + tokens + [self.vocab.index("</s>")]

        paddings = self.max_text_length - len(tokens)
        return tokens + [self.vocab.index("<pad>")] * paddings

    def create_attention_mask(self, input_ids: list[int]) -> list[int]:
        return [1 if i != self.vocab.index("<pad>") else 0 for i in input_ids]

    def create_labels(self, input_ids: list[int]) -> list[int]:
        labels = input_ids[1:] + [-100]
        return [i if i != self.vocab.index("<pad>") else -100 for i in labels]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.dataset.iloc[index]

        visual_token_ids = self.create_visual_tokens(example.visual_tokens)
        text_token_ids = self.create_text_tokens(getattr(example, "text", None))

        input_ids = visual_token_ids + text_token_ids
        attention_mask = self.create_attention_mask(input_ids)
        labels = self.create_labels(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int64),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }


class TextToImageDataset(ImageToTextDataset):
    dataset: pd.DataFrame
    vocab: list[str]
    max_visual_length: int
    max_text_length: int | None = None

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.dataset.iloc[index]

        visual_token_ids = self.create_visual_tokens(example.visual_tokens)
        text_token_ids = self.create_text_tokens(getattr(example, "text", None))

        input_ids = text_token_ids + visual_token_ids
        attention_mask = self.create_attention_mask(input_ids)
        labels = self.create_labels(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int64),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
