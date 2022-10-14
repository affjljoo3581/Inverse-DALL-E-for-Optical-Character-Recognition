from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torchvision.utils import make_grid

from modeling import VQVAEDecoder, VQVAEEncoder, VQVAEQuantizer

try:
    from apex.optimizers import FusedAdam as Adam
except ModuleNotFoundError:
    from torch.optim import Adam


class VQVAETrainingModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.num_log_batches = math.ceil(64 / self.config.train.batch_size)

        self.encoder = VQVAEEncoder(**config.model.encoder)
        self.decoder = VQVAEDecoder(**config.model.decoder)
        self.quantizer = VQVAEQuantizer(**config.model.quantizer)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        encodings = self.encoder(images)
        latents, loss_quantization, perplexity = self.quantizer(encodings)
        decodings = self.decoder(latents)

        loss_reconstruction = F.l1_loss(images, decodings)
        loss_total = loss_reconstruction + loss_quantization

        metrics = {
            "loss_total": loss_total,
            "loss_reconstruction": loss_reconstruction,
            "loss_quantization": loss_quantization,
            "perplexity": perplexity,
            "encoding_norm": encodings.norm(dim=1).mean(),
        }
        return metrics, decodings

    def training_step(self, images: torch.Tensor, batch_idx: int) -> torch.Tensor:
        metrics, _ = self(images)
        self.log("step", self.global_step)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()})
        return metrics["loss_total"]

    def validation_step(
        self, images: torch.Tensor, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        metrics, decodings = self(images)
        self.log("step", self.global_step)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})

        # Prevent from storing unnecessary image tensors which consume large portion of
        # GPU memory and occur OOM at validation.
        if batch_idx < self.num_log_batches:
            return images, decodings
        return None

    def validation_epoch_end(self, outputs: tuple[torch.Tensor, torch.Tensor] | None):
        images = torch.cat([output[0] for output in outputs if output is not None])
        decodings = torch.cat([output[1] for output in outputs if output is not None])

        grid = torch.stack((images[:64], decodings[:64]), dim=1).flatten(0, 1)
        grid = make_grid(grid, nrow=8, value_range=(-1, 1))
        self.logger.log_image("val/reconstructed", [grid])

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), **self.config.optim)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "ApexMixedPrecisionPlugin" in checkpoint:
            checkpoint.pop("ApexMixedPrecisionPlugin")
