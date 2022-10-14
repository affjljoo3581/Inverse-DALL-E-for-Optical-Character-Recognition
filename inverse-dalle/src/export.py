from __future__ import annotations

import argparse
import os
import warnings

from omegaconf import DictConfig, OmegaConf

from lightning import InverseDALLEDataModule, InverseDALLELightningModule

warnings.filterwarnings("ignore")


def main(name: str, config: DictConfig, checkpoint: str):
    datamodule = InverseDALLEDataModule(config)
    module = InverseDALLELightningModule.load_from_checkpoint(
        checkpoint, config=config, vocab=datamodule.vocab, map_location="cpu"
    )
    module.model.half().save_pretrained(name)
    with open(os.path.join(name, "vocab.txt"), "w") as fp:
        fp.write("\n".join(datamodule.vocab))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    name = os.path.splitext(os.path.basename(args.config))[0]
    main(name, config, args.checkpoint)
