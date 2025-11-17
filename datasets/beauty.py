from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import gzip
import json
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class BeautyDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'beauty'

    @classmethod
    def url(cls):
        return ['http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz',
                'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz']

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['beauty_reviews.json.gz', 'beauty_meta.json.gz']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        
        print("Raw file doesn't exist. Downloading...")
        for idx, url in enumerate(self.url()):
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(url, tmpfile)
            os.makedirs(folder_path, exist_ok=True)
            shutil.move(tmpfile, folder_path.joinpath(self.all_raw_file_names()[idx]))
            print()

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        meta_raw = self.load_meta_dict()
        # Keep all items (no metadata filter)
        df = self.filter_triplets(df)
        
        # Sort by original_order for mapping (preserve file order)
        print('Sorting by original file order for ID mapping...')
        df = df.sort_values(by='original_order', kind='mergesort').reset_index(drop=True)
        
        # Densify index (mapping based on original file order)
        df, umap, smap = self.densify_index(df)
        
        # Now sort by timestamp for proper train/val/test split
        print('Sorting by timestamp for train/val/test split...')
        df = df.sort_values(by=['timestamp', 'original_order'], ascending=[True, True], kind='mergesort')
        df = df.drop(columns=['original_order']).reset_index(drop=True)
        
        # Split dataset
        train, val, test = self.split_df(df, len(umap))
        
        # Create metadata with fallback for items without metadata
        meta = {}
        for item_id in smap.values():
            orig_id = [k for k, v in smap.items() if v == item_id][0]
            if orig_id in meta_raw:
                meta[item_id] = meta_raw[orig_id]
            else:
                meta[item_id] = f"Product_{orig_id[:8]}"
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'meta': meta,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[0])
        
        data = []
        original_order = 0
        with gzip.open(file_path, 'rb') as f:
            for line in f:
                original_order += 1
                try:
                    review = json.loads(line.decode('utf-8'))
                except json.JSONDecodeError:
                    review = ast.literal_eval(line.decode('utf-8'))
                data.append({
                    'original_order': original_order,
                    'uid': review['reviewerID'],
                    'sid': review['asin'],
                    'rating': review['overall'],
                    'timestamp': review['unixReviewTime']
                })
        
        df = pd.DataFrame(data)
        # Keep original file order (do NOT sort here)
        return df
    
    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[1])

        meta_dict = {}
        with gzip.open(file_path, 'rb') as f:
            for line in f:
                try:
                    item = json.loads(line.decode('utf-8'))
                except json.JSONDecodeError:
                    item = ast.literal_eval(line.decode('utf-8'))
                if 'title' in item and len(item['title']) > 0:
                    meta_dict[item['asin'].strip()] = item['title'].strip()
        
        return meta_dict
