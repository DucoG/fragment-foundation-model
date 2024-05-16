from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple
import logging

import numpy as np

from omegaconf import DictConfig
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from fragformer.datamodule.components.pretrain_dataset import PolarsBatchedParquetDataset
from fragformer.datamodule.components.tokenizers import Tokenizer


# A minimialistic lightning datamodule for the pretraining
class PretrainDataModule(LightningDataModule):
    def __init__(
        self,
        parquet_path: Union[str, Path],
        columns: List[str],
        context_window: int,
        tokenizer: Tokenizer,
        val_split: float = 0,
        transform: Optional[Any] = None,
        padder: Optional[Any] = None,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> None:

        super().__init__()

        # Save the parameters
        self.save_hyperparameters()

        self.parquet_files = self.parse_paths(parquet_path)
        self.train_parquet_files, self.val_parquet_files = self.split_data(self.parquet_files, val_split)
        self.columns = columns
        self.context_window = context_window
        self.transform = transform
        self.tokenizer = tokenizer
        self.padder = padder

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = PolarsBatchedParquetDataset(
            parquet_files=self.train_parquet_files,
            columns=self.columns,
            context_window=self.context_window,
            batch_size=self.hparams.batch_size,
            tokenizer=self.tokenizer,
            padder=self.padder,
            transform=self.transform)
        
        if self.val_parquet_files:
            self.val_dataset = PolarsBatchedParquetDataset(
                parquet_files=self.val_parquet_files,
                columns=self.columns,
                context_window=self.context_window,
                batch_size=self.hparams.batch_size,
                tokenizer=self.tokenizer,
                padder=self.padder,
                transform=self.transform)
        else:
            self.val_dataset = []

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=self.hparams.pin_memory,
        )

    def parse_paths(self, path: Union[str, Path]) -> List[Path]:
        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

        if path.is_dir():
            return list(path.glob('*.parquet'))

        if path.is_file():
            return [path]

    def split_parquet_files(self, parquet_files: List[Path], val_split: float) -> Tuple[List[Path], List[Path]]:
        if val_split == 0 or len(parquet_files) == 1:
            logging.info("No validation split or only one file found. Skipping validation split.")
            return parquet_files, []
        
        # shuffle the files
        np.random.shuffle(parquet_files)

        split_idx = max(len(parquet_files) * val_split, 1)
        return parquet_files[split_idx:], parquet_files[:split_idx]