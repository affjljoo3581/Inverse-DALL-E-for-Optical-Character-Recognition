from __future__ import annotations

import glob
import os
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable

import cv2
import imagesize
import torch
import tqdm
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


@dataclass
class ImageDatasetWithPadding(Dataset):
    image_files: list[str]
    image_size: int = 128

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(self.image_files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fxy = self.image_size / max(image.shape)
        image = cv2.resize(image, None, fx=fxy, fy=fxy, interpolation=cv2.INTER_AREA)

        padding_width = self.image_size - image.shape[1]
        padding_height = self.image_size - image.shape[0]

        image = cv2.copyMakeBorder(
            image,
            top=padding_height // 2,
            bottom=padding_height - padding_height // 2,
            left=padding_width // 2,
            right=padding_width - padding_width // 2,
            borderType=cv2.BORDER_CONSTANT,
        )
        image = 2 * torch.from_numpy(image).float().permute(2, 0, 1) / 0xFF - 1
        return image


class ImageDatasetForTokenization(Dataset):
    def __init__(
        self,
        image_files: list[str],
        max_length: int = 64,
        pooling: int = 16,
        transform: Callable | None = None,
    ):
        self.image_files = image_files
        self.max_length = max_length
        self.pooling = pooling
        self.transform = transform

        self.shape_templates = []
        for i in range(1, max_length + 1):
            j = (max_length - i) // i
            padding = max_length - (j + 1) * i
            if j > 0 and padding < max_length**0.5:
                self.shape_templates.append((j * pooling, i * pooling, j / i))

    def __len__(self) -> int:
        return len(self.image_files)

    def resize_shape(self, width: int, height: int) -> tuple[int, int]:
        aspect_ratio = width / height
        return min(self.shape_templates, key=lambda x: abs(aspect_ratio - x[-1]))[:-1]

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.cvtColor(cv2.imread(self.image_files[index]), cv2.COLOR_BGR2RGB)
        width, height = self.resize_shape(image.shape[1], image.shape[0])

        fxy = min(width / image.shape[1], height / image.shape[0])
        image = cv2.resize(image, None, fx=fxy, fy=fxy, interpolation=cv2.INTER_AREA)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        right, bottom = width - image.shape[1], height - image.shape[0]
        image = cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_CONSTANT)

        image = 2 * torch.from_numpy(image).float().permute(2, 0, 1) / 0xFF - 1
        return image


class BatchSamplerForTokenization(Sampler[list[int]]):
    def __init__(self, dataset: ImageDatasetForTokenization, batch_size: int):
        shape_groups = defaultdict(list)
        for i, image_file in enumerate(tqdm.tqdm(dataset.image_files)):
            width, height = dataset.resize_shape(*imagesize.get(image_file))
            shape_groups[(width, height)].append(i)

        self.batch_indices = []
        for indices in shape_groups.values():
            for i in range(0, len(indices), batch_size):
                self.batch_indices.append(indices[i : i + batch_size])

    def __len__(self) -> int:
        return len(self.batch_indices)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batch_indices


def create_train_val_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    dataset = ImageDatasetWithPadding(
        image_files=glob.glob(config.data.image_files),
        image_size=config.data.image_size,
    )
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=config.data.validation_ratio, random_state=42
    )

    train_dataloader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config.train.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader
