from .ml_100k import ML100KDataset
from .beauty import BeautyDataset
from .games import GamesDataset
from .yelp2020 import Yelp2020Dataset

DATASETS = {
    ML100KDataset.code(): ML100KDataset,
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset,
    Yelp2020Dataset.code(): Yelp2020Dataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
