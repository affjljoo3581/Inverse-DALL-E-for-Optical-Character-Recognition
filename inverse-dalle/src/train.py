from __future__ import annotations

import argparse
import os
import warnings

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightning import InverseDALLEDataModule, InverseDALLELightningModule

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    name: str,
    config: DictConfig,
    resume_from: str | None = None,
    resume_id: str | None = None,
):
    datamodule = InverseDALLEDataModule(config)
    module = InverseDALLELightningModule(config, vocab=datamodule.vocab)

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        precision=16,
        amp_backend="apex",
        log_every_n_steps=config.train.log_every_n_steps,
        max_steps=config.optim.scheduler.num_training_steps,
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        val_check_interval=min(config.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(config.train.validation_interval), 1),
        callbacks=[ModelCheckpoint(save_last=True), LearningRateMonitor("step")],
        logger=WandbLogger(project="inverse-dalle-ocr-gpt3", name=name, id=resume_id),
    )
    trainer.fit(module, datamodule, ckpt_path=resume_from)


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
