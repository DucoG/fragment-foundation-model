import random
from typing import Any, Optional, Tuple, Union, Dict, List
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import polars as pl
from pathlib import Path

from fragformer.datamodule.components.tokenizers import Tokenizer

class PolarsParquetBatchIterator:
    def __init__(self, 
                 parquet_files: Union[List[str], List[Path]],
                 columns: List[str], 
                 context_window: int,
                 batch_size: int,
                 tokenizer: Tokenizer,
                 transform: Optional[Any] = None,
                 padder: Optional[Any] = None
        ):
        self.parquet_files = parquet_files
        self.columns = columns
        self.context_window = context_window
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.transform = transform
        self.padder = padder(self.tokenizer.pad_token_id, self.max_seq_len) if padder is not None else None

        self.file_index: int = 0
        self.data_index: int = 0
        self.leftover: Union[pl.DataFrame, None] = None

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.data_index == 0:  # Load new file
                if self.file_index >= len(self.parquet_files):
                    if self.leftover is not None:
                        result = self.leftover
                        self.leftover = None
                        return result
                    raise StopIteration
                self.load_next_file()

            if self.data_index + self.batch_size <= self.current_data.shape[0]:
                batch = self.preprocess_batch(self.current_data[self.data_index:self.data_index + self.batch_size])
                self.data_index += self.batch_size
                return batch

            self.leftover = self.current_data[self.data_index:] if self.data_index < self.current_data.shape[0] else None
            self.data_index = 0
    
    def load_next_file(self):
        self.current_data = (pl.scan_parquet(self.parquet_files[self.file_index])
                      .select(['read1_seq', 'read2_seq'])
                      .with_columns([
                          pl.col('read1_seq').str.slice(
                              (pl.col('read1_seq').str.len_chars() / 2 - self.context_window).cast(int),
                              2 * self.context_window
                            ),
                            pl.col('read2_seq').str.slice(
                                (pl.col('read2_seq').str.len_chars() / 2 - self.context_window).cast(int),
                                2 * self.context_window
                            )
                      ])
                      ).collect().map_rows(lambda x: '[CLS]' + x[0].replace('', ' ') + '[SEP]' + x[1].replace('', ' ') + '[SEP]').sample(fraction=1)
        
        if self.transform:
            self.current_data = self.transform(self.current_data)
                
        if self.leftover is not None:
            self.current_data = pl.concat([self.leftover, self.current_data], how='vertical')
            self.leftover = None
        self.file_index += 1

    def preprocess_batch(self, batch) -> torch.Tensor:
        batch = self.current_data[self.data_index:self.data_index + self.batch_size]
        batch = batch.map_rows(lambda x: (self.tokenizer.encode(x[0]),)).to_numpy()
        batch = torch.from_numpy(np.stack(batch[:, 0], dtype=np.float32))
        return batch
        

class PolarsBatchedParquetDataset(IterableDataset):
    def __init__(self, 
                 parquet_files: Union[List[str], List[Path]],
                 columns: List[str],
                 context_window: int,
                 batch_size: int,
                 tokenizer: Tokenizer,
                 transform: Optional[Any] = None,
                    padder: Optional[Any] = None
                 ):
        super().__init__()
        parquet_files = parquet_files if isinstance(parquet_files, list) else [parquet_files]
        random.shuffle(parquet_files)
        self.parquet_files = parquet_files

        self.columns = columns
        self.context_window = context_window
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.transform = transform
        self.padder = padder

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return PolarsParquetBatchIterator(self.parquet_files, 
                                              self.columns, 
                                              self.context_window,
                                              self.batch_size, 
                                              self.tokenizer, 
                                              self.transform,
                                              self.padder)
        else:
            return PolarsParquetBatchIterator(
                [elem for ind, elem in enumerate(self.parquet_files) if (ind % worker_info.num_workers) == worker_info.id], 
                self.columns, 
                self.context_window,
                self.batch_size,
                self.tokenizer,
                self.transform,
                self.padder)
