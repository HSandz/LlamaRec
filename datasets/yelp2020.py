from .base import AbstractDataset
from .utils import *

from datetime import datetime
from pathlib import Path
import pickle
import shutil
import tempfile
import os
import json
import calendar
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class Yelp2020Dataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'yelp2020'

    @classmethod
    def url(cls):
        return 'https://drive.google.com/uc?id=1ugbgehShD2xTqdFWcNoba6xN5IQnT93R'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['yelp_academic_dataset_review.json', 'yelp_academic_dataset_business.json']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and all(folder_path.joinpath(f).is_file() for f in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        
        print("Downloading Yelp2020 dataset from Google Drive...")
        import gdown
        
        tmproot = Path(tempfile.mkdtemp())
        tmpzip = tmproot.joinpath('file.zip')
        tmpfolder = tmproot.joinpath('folder')
        
        gdown.download(self.url(), str(tmpzip), quiet=False)
        unzip(tmpzip, tmpfolder)
        
        # Check if unzipped content is in a subfolder
        subfolders = list(tmpfolder.iterdir())
        if len(subfolders) == 1 and subfolders[0].is_dir():
            actual_folder = subfolders[0]
        else:
            actual_folder = tmpfolder
        
        # Create parent directory if it doesn't exist
        folder_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move to final location
        if folder_path.exists():
            shutil.rmtree(folder_path)
        shutil.move(str(actual_folder), str(folder_path))
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
        
        # Filter reviews within 2019 (UTC+0)
        print('Filtering reviews within 2019 (UTC+0)...')
        df = df[(df['timestamp'] >= pd.Timestamp('2019-01-01', tz='UTC').timestamp()) & 
                (df['timestamp'] < pd.Timestamp('2020-01-01', tz='UTC').timestamp())]
        print(f'Reviews within 2019: {len(df):,}')
        print(f'  Users: {df["uid"].nunique():,}, Items: {df["sid"].nunique():,}')
        
        # Apply 5-core filtering using base class method
        print('Applying 5-core filtering...')
        df = self.filter_triplets(df)
        
        print(f'After 5-core filtering:')
        print(f'  Interactions: {len(df):,}')
        print(f'  Users: {df["uid"].nunique():,}, Items: {df["sid"].nunique():,}')
        
        # Load metadata
        meta_raw = self.load_meta_dict()
        
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
        
        # Create metadata dict with fallback for missing names
        meta = {}
        for item_id in smap.values():
            orig_id = [k for k, v in smap.items() if v == item_id][0]
            if orig_id in meta_raw:
                meta[item_id] = meta_raw[orig_id]
            else:
                meta[item_id] = f"Business_{orig_id[:8]}"
        
        dataset = {'train': train, 'val': val, 'test': test, 'meta': meta, 'umap': umap, 'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[0])
        
        print('Loading Yelp2020 reviews...')
        reviews = []
        original_order = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='Reading reviews'):
                original_order += 1
                review = json.loads(line)
                # Parse as UTC+0 explicitly using calendar.timegm
                dt = datetime.strptime(review['date'], '%Y-%m-%d %H:%M:%S')
                timestamp = calendar.timegm(dt.timetuple())
                
                reviews.append({
                    'original_order': original_order,  # Track for stable sorting
                    'uid': review['user_id'],
                    'sid': review['business_id'],
                    'rating': review['stars'],
                    'timestamp': timestamp
                })
        
        df = pd.DataFrame(reviews)
        # Keep original file order for mapping (do NOT sort here)
        df = df.reset_index(drop=True)
        
        print(f'Total reviews loaded: {len(df):,}')
        return df
    
    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[1])
        
        print('Loading business metadata...')
        meta_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='Reading businesses'):
                business = json.loads(line)
                business_id = business['business_id']
                
                # Always add business (use ID as fallback if no name)
                if 'name' in business and business['name']:
                    name = business['name'].strip()
                    if 'categories' in business and business['categories']:
                        category = business['categories'].split(',')[0].strip()
                        meta_dict[business_id] = f"{name} ({category})"
                    else:
                        meta_dict[business_id] = name
                else:
                    meta_dict[business_id] = f"Business {business_id[:8]}"
        
        print(f'Total businesses: {len(meta_dict):,}')
        return meta_dict
