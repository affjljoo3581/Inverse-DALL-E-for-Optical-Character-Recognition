from __future__ import annotations

import argparse
import os
import warnings

import albumentations as A
import cv2
import pandas as pd
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data import BatchSamplerForTokenization, ImageDatasetForTokenization
from lightning import VQVAETrainingModule

warnings.filterwarnings("ignore")


@torch.inference_mode()
def main(args: argparse.Namespace, config: DictConfig):
    model = VQVAETrainingModule.load_from_checkpoint(args.checkpoint, config=config)
    model.cuda().eval()

    dataframe = pd.read_csv(args.dataset)
    image_files = dataframe.img_path.map(
        lambda x: os.path.join(os.path.dirname(args.dataset), x)
    )

    transform = None
    if args.use_tta:
        transform = [
            A.Rotate(limit=8, p=1.0, border_mode=cv2.BORDER_CONSTANT),
            A.HueSaturationValue(180, 50, 50, p=1.0),
        ]
        transform = A.Compose(transform)

    dataset = ImageDatasetForTokenization(
        image_files=image_files,
        max_length=args.max_length,
        pooling=2 ** (len(config.model.encoder.num_layers) - 1),
        transform=transform,
    )
    batch_sampler = BatchSamplerForTokenization(dataset, args.batch_size)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
    )

    outputs = []
    for _ in range(args.num_repeats if args.use_tta else 1):
        predictions = []
        for indices, images in zip(batch_sampler, tqdm.tqdm(dataloader)):
            output = model.quantizer(model.encoder(images.cuda()), quantize_only=True)
            for index, token_ids in zip(indices, output.cpu().numpy()):
                tokens = "\t".join(" ".join(map(str, ids)) for ids in token_ids)
                predictions.append((index, tokens))
        predictions = [tokens for _, tokens in sorted(predictions, key=lambda x: x[0])]

        output = dataframe.copy()
        output["visual_tokens"] = predictions
        outputs.append(output)
    pd.concat(outputs).to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset", default="resources/external.csv")
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--use-tta", action="store_true", default=False)
    parser.add_argument("--num-repeats", type=int, default=10)
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(args, config)
