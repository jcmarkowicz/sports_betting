"""Compact, selectively readable paths. Only open trusted source tables."""
import json
import hashlib
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd


class CompactPaths(Sequence):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'metadata.json').read_text())
        if self.metadata['version'] != 1:
            raise ValueError('Unsupported compact path version')
        if hashlib.sha256((self.directory / 'source.pkl').read_bytes()).hexdigest() != self.metadata['source_sha256']:
            raise ValueError('Compact source table checksum differs')
        self.source = pd.read_pickle(self.directory / 'source.pkl')
        self.ids = np.load(self.directory / 'donors.npy', mmap_mode='r')
        self.overrides = {name: np.load(self.directory / file, mmap_mode='r')
                          for name, file in self.metadata['overrides'].items()}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        frame = self.source.iloc[self.ids[index]].reset_index(drop=True).copy()
        for name, array in self.overrides.items():
            values = array[index].copy()
            if name in self.metadata.get('string_values', {}):
                values = np.asarray(self.metadata['string_values'][name])[values]
            frame[name] = values
        frame['source_row_position'] = np.asarray(self.ids[index]).copy()
        for name, source_name in self.metadata.get('source_aliases', {}).items():
            frame[name] = self.source[source_name].iloc[self.ids[index]].to_numpy(copy=True)
        frame = frame[self.metadata['columns']]
        for name, dtype in self.metadata['dtypes'].items():
            frame[name] = frame[name].astype(dtype)
        frame.attrs['generation'] = dict(self.metadata['generation'], path_id=int(index))
        return frame

    def close(self):
        for array in [self.ids, *self.overrides.values()]:
            array._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CompactPathWriter:
    def __init__(self, directory, source, count):
        self.directory = Path(directory)
        # Never replace an existing dataset or publish an incomplete manifest.
        self.directory.mkdir(parents=True, exist_ok=False)
        self.source = source.reset_index(drop=True).copy()
        self.source["date"] = pd.to_datetime(self.source.date)
        self.source.to_pickle(self.directory / 'source.pkl')
        self.ids = np.lib.format.open_memmap(self.directory / 'donors.npy', mode='w+',
                                            dtype='int32', shape=(count, len(source)))
        self.arrays, self.files, self.strings, self.aliases = {}, {}, {}, {}
        self.number = 0

    def append(self, frame):
        ids = frame.source_row_position.to_numpy(dtype='int32')
        if len(ids) != self.ids.shape[1]:
            raise ValueError('Compact paths must preserve the number of fight slots')
        self.ids[self.number] = ids
        for name in frame:
            if name == 'source_row_position':
                continue
            source_name = {'source_date': 'date', 'source_event_name': 'event_name'}.get(name)
            if source_name in self.source:
                if not frame[name].reset_index(drop=True).equals(self.source[source_name].iloc[ids].reset_index(drop=True)):
                    raise ValueError(f'Source metadata differs for {name}')
                self.aliases[name] = source_name
                continue
            values = frame[name].to_numpy()
            if name not in self.arrays:
                if name in self.source and frame[name].reset_index(drop=True).equals(
                        self.source[name].iloc[ids].reset_index(drop=True)):
                    continue
                if values.dtype.kind == 'O':
                    if not all(isinstance(v, str) for v in values):
                        raise ValueError(f'Unsupported compact override column: {name}')
                    vocabulary = list(dict.fromkeys(
                        (self.source[name].tolist() if name in self.source else []) + values.tolist()))
                    self.strings[name] = vocabulary
                    lookup = {v: i for i, v in enumerate(vocabulary)}
                    values = np.array([lookup[v] for v in values], dtype='int32')
                elif name in ('template_row_position', 'sampled_event_id') and values.dtype.kind in 'iu' and values.min() >= np.iinfo(np.int32).min and values.max() <= np.iinfo(np.int32).max:
                    values = values.astype('int32')
                file = f'column_{len(self.arrays)}.npy'
                array = np.lib.format.open_memmap(self.directory / file, mode='w+',
                    dtype=values.dtype, shape=self.ids.shape)
                for previous in range(self.number):
                    previous_values = self.source[name].iloc[self.ids[previous]].to_numpy()
                    if name in self.strings:
                        previous_values = [lookup[v] for v in previous_values]
                    array[previous] = previous_values
                self.arrays[name], self.files[name] = array, file
            array = self.arrays[name]
            if name in self.strings:
                vocabulary = self.strings[name]
                lookup = {v: i for i, v in enumerate(vocabulary)}
                encoded = []
                for value in frame[name]:
                    if value not in lookup:
                        lookup[value] = len(vocabulary)
                        vocabulary.append(value)
                    encoded.append(lookup[value])
                values = encoded
            array[self.number] = values
        self.columns = list(frame.columns)
        self.dtypes = {name: str(dtype) for name, dtype in frame.dtypes.items()}
        self.generation = dict(frame.attrs['generation'])
        self.generation.pop('path_id', None)
        self.number += 1

    def close(self):
        for array in [self.ids, *self.arrays.values()]:
            array._mmap.close()

    def finish(self):
        if self.number != len(self.ids):
            raise ValueError('Incomplete path dataset')
        for array in [self.ids, *self.arrays.values()]:
            array.flush()
        metadata = dict(version=1, columns=self.columns, dtypes=self.dtypes,
                        generation=self.generation, overrides=self.files,
                        string_values=self.strings,
                        source_aliases=self.aliases,
                        source_sha256=hashlib.sha256((self.directory / 'source.pkl').read_bytes()).hexdigest())
        (self.directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        for array in [self.ids, *self.arrays.values()]:
            array._mmap.close()
        return CompactPaths(self.directory)
