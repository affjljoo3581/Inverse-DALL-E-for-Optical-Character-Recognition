from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, get_scheduler

from data import ImageToTextDataset

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class InverseDALLELightningModule(LightningModule):
    def __init__(self, config: DictConfig, vocab: list[str]):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.max_text_length = config.data.max_text_length

        vocab_size = len(self.vocab) + config.data.num_visual_tokens + 2
        vocab_size = (vocab_size + 7) // 8 * 8
        self.model = GPT2LMHeadModel(GPT2Config(**config.model, vocab_size=vocab_size))

        if config.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(
        self, labels: torch.Tensor, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits = self.model(**kwargs, use_cache=False).logits
        logits_image = logits[:, : -self.max_text_length].flatten(0, 1)
        logits_text = logits[:, -self.max_text_length :].flatten(0, 1)

        labels_images = labels[:, : -self.max_text_length].flatten()
        labels_text = labels[:, -self.max_text_length :].flatten()

        loss_image_lm = F.cross_entropy(logits_image, labels_images)
        loss_text_lm = F.cross_entropy(logits_text, labels_text)
        loss_total = loss_image_lm + loss_text_lm
        return {
            "loss_image_lm": loss_image_lm,
            "loss_text_lm": loss_text_lm,
            "loss_total": loss_total,
        }

    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        metrics = self(**batch)
        self.log("step", self.global_step)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()})
        return metrics["loss_total"]

    def validation_step(
        self, batch: dict[str, torch.Tensor], idx: int
    ) -> list[tuple[str, str]]:
        metrics = self(**batch)
        self.log("step", self.global_step)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})

        input_ids = batch["input_ids"][:, : -self.max_text_length + 1]
        attention_mask = batch["attention_mask"][:, : -self.max_text_length + 1]

        generated = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_text_length - 1,
            eos_token_id=self.vocab.index("</s>"),
            pad_token_id=self.vocab.index("<pad>"),
        )
        generated = generated[:, input_ids.size(1) :]
        labels = batch["input_ids"][:, -self.max_text_length :]

        generated = [self.decode_to_text(seq) for seq in generated.tolist()]
        labels = [self.decode_to_text(seq) for seq in labels.tolist()]
        return list(zip(generated, labels))

    def decode_to_text(self, sequence: list[int]) -> str:
        text = ""
        for token_id in sequence:
            if token_id >= len(self.vocab) or self.vocab[token_id] in ["<s>", "<pad>"]:
                continue
            elif self.vocab[token_id] == "</s>":
                break
            text += self.vocab[token_id]
        return text

    def validation_epoch_end(self, outputs: list[list[tuple[str, str]]]):
        outputs = sum(outputs, [])
        if not outputs:
            return
        self.log("val/accuracy", sum(x == y for x, y in outputs) / len(outputs))
        self.logger.log_text("val/text", columns=["pred", "label"], data=outputs[:100])

    def parameter_groups(self) -> list[dict[str, Any]]:
        do_decay = [p for p in self.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.ndim < 2]
        return [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = AdamW(self.parameter_groups(), **self.config.optim.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.config.optim.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "ApexMixedPrecisionPlugin" in checkpoint:
            checkpoint.pop("ApexMixedPrecisionPlugin")


class InverseDALLEDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.train_df = pd.read_csv(self.config.data.train_dataset, engine="pyarrow")
        self.test_df = pd.read_csv(self.config.data.val_dataset, engine="pyarrow")

        self.vocab = ["<s>", "</s>", "<pad>"] + sorted(
            set("".join(self.train_df.text) + "".join(self.test_df.text))
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ImageToTextDataset(
            dataset=self.train_df,
            vocab=self.vocab,
            max_visual_length=self.config.data.max_visual_length,
            max_text_length=self.config.data.max_text_length,
        )
        self.val_dataset = ImageToTextDataset(
            dataset=self.test_df,
            vocab=self.vocab,
            max_visual_length=self.config.data.max_visual_length,
            max_text_length=self.config.data.max_text_length,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count(), 4),
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=min(os.cpu_count(), 4),
            persistent_workers=True,
        )
