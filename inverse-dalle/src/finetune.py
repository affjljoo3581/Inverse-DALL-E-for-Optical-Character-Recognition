from __future__ import annotations

import argparse
import os
import warnings
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, get_scheduler

from data import TextToImageDataset

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DALLELightningModule(LightningModule):
    def __init__(self, config: DictConfig, vocab: list[str]):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.max_text_length = config.data.max_text_length
        self.model = GPT2LMHeadModel.from_pretrained(config.model)

    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        labels = batch.pop("labels")
        logits = self.model(**batch, use_cache=False).logits

        logits_text = logits[:, : self.max_text_length].flatten(0, 1)
        logits_image = logits[:, self.max_text_length :].flatten(0, 1)

        labels_text = labels[:, : self.max_text_length].flatten()
        labels_images = labels[:, self.max_text_length :].flatten()

        loss_text_lm = F.cross_entropy(logits_text, labels_text)
        loss_image_lm = F.cross_entropy(logits_image, labels_images)
        loss_total = loss_text_lm + loss_image_lm

        self.log("step", self.global_step)
        self.log("train/loss_text_lm", loss_text_lm)
        self.log("train/loss_image_lm", loss_image_lm)
        self.log("train/loss_total", loss_total)
        return loss_total

    def parameter_groups(self) -> list[dict[str, Any]]:
        do_decay = [p for p in self.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.ndim < 2]
        return [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = AdamW(self.parameter_groups(), **self.config.optim.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.config.optim.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class DALLEDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        with open(os.path.join(config.model, "vocab.txt")) as fp:
            self.vocab = [line.strip("\n") for line in fp]

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = TextToImageDataset(
            dataset=pd.read_csv(self.config.data.train_dataset, engine="pyarrow"),
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


def main(
    name: str,
    config: DictConfig,
    resume_from: str | None = None,
    resume_id: str | None = None,
):
    datamodule = DALLEDataModule(config)
    module = DALLELightningModule(config, vocab=datamodule.vocab)

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        precision=16,
        amp_backend="apex",
        max_steps=config.optim.scheduler.num_training_steps,
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        callbacks=[LearningRateMonitor("step")],
        logger=WandbLogger(project="inverse-dalle-ocr-gpt3", name=name, id=resume_id),
    )
    trainer.fit(module, datamodule, ckpt_path=resume_from)

    module.model.save_pretrained(name)
    with open(os.path.join(name, "vocab.txt"), "w") as fp:
        fp.write("\n".join(datamodule.vocab))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume-from")
    parser.add_argument("--resume-id")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    name = os.path.splitext(os.path.basename(args.config))[0]
    main(name, config, args.resume_from, args.resume_id)
