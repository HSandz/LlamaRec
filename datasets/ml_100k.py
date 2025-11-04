from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class ML100KDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-100k'

    @classmethod
    def url(cls):  # as of Sep 2023
        return 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'u.data',
                'u.item',
                'u.user']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return

        print("Raw file doesn't exist. Downloading...")
        tmproot = Path(tempfile.mkdtemp())
        tmpzip = tmproot.joinpath('file.zip')
        tmpfolder = tmproot.joinpath('folder')
        download(self.url(), tmpzip)
        unzip(tmpzip, tmpfolder)
        if self.zip_file_content_is_folder():
            tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
        shutil.move(tmpfolder, folder_path)
        shutil.rmtree(tmproot)
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
        # Filter by minimum rating threshold
        if self.min_rating > 0:
            df = df[df['rating'] >= self.min_rating]
        # Filter items without meta info
        df = df[df['sid'].isin(meta_raw.keys())]
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        meta = {smap[k]: v for k, v in meta_raw.items() if k in smap}
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
        file_path = folder_path.joinpath('u.data')
        # MovieLens 100K uses tab-separated format: user_id, item_id, rating, timestamp
        df = pd.read_csv(file_path, sep='\t', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('u.item')
        # MovieLens 100K u.item format: movie_id | movie_title | release_date | ... (pipe-separated)
        df = pd.read_csv(file_path, sep='|', header=None, encoding="ISO-8859-1", 
                         usecols=[0, 1])  # Only read movie_id and title columns
        df.columns = ['movie_id', 'title']
        
        meta_dict = {}
        for row in df.itertuples():
            movie_id = row.movie_id
            title_with_year = row.title
            
            # Safely extract year and title
            if len(title_with_year) >= 7 and title_with_year[-6:-1].isdigit():
                title = title_with_year[:-7].strip()  # Remove year (e.g., " (1995)")
                year = title_with_year[-7:]
            else:
                title = title_with_year.strip()
                year = ''
            
            # Remove other parentheses content
            title = re.sub('\(.*?\)', '', title).strip()
            
            # Move articles (a, an, the) from end to beginning
            # Example: "Shawshank Redemption, The" -> "The Shawshank Redemption"
            if len(title) > 5 and any(', '+x in title.lower()[-5:] for x in ['a', 'an', 'the']):
                title_parts = title.split(', ')
                if len(title_parts) >= 2:
                    title_pre = ', '.join(title_parts[:-1])
                    title_post = title_parts[-1]
                    title = title_post + ' ' + title_pre
            
            meta_dict[movie_id] = title + year
        
        return meta_dict
