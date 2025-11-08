"""Quick test for Yelp preprocessing"""
import argparse
from datasets import dataset_factory

args = argparse.Namespace(
    dataset_code='yelp2020',
    min_rating=0,
    min_uc=5,
    min_sc=5
)

dataset = dataset_factory(args)
dataset.preprocess()

# Load and check
result = dataset.load_dataset()
print(f"\n{'='*60}")
print(f"FINAL RESULTS:")
print(f"  Users: {len(result['train']):,}")
print(f"  Items: {len(result['meta']):,}")
total_int = sum(len(result['train'][u]) + len(result['val'][u]) + len(result['test'][u]) for u in result['train'])
print(f"  Total interactions: {total_int:,}")
# print(f"\nYELP2020: users=30,431, items=20,033, interactions=316,354")
# print(f"\nBEAUTY: users=22,363, items=12,101, interactions=198502")
print(f"{'='*60}")