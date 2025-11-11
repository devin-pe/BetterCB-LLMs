#!/usr/bin/env python3
"""
download_echr.py

Download the Hugging Face `ecthr_cases` dataset, save each split as CSV, and
produce a flattened facts CSV (one fact per row) with binary `label` indicating
whether the original case contained any violated articles.

Usage:
    python download_echr.py --out_dir ./data/echr --cache_dir /scratch/hf_cache --use_auth_token

Options:
    --out_dir     directory to write CSVs (created if missing)
    --cache_dir   (optional) HF cache dir
    --use_auth_token  include this flag to pass `use_auth_token=True` to HF loader

WARNING: this script sets `trust_remote_code=True` for the dataset loader.
It will execute dataset-supplied code. Only run with datasets you trust.
"""

import argparse
import os
from datasets import load_dataset
import pandas as pd


def save_splits(ds, out_dir):
    for split in ds.keys():
        print(f"Saving split {split} with {len(ds[split])} rows...")
        df = ds[split].to_pandas()
        out_path = os.path.join(out_dir, f"echr_{split}.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")


def flatten_facts(ds, out_dir):
    records = []
    for split in ds.keys():
        print(f"Flattening {split}...")
        for item in ds[split]:
            facts = item.get("facts", []) or []
            violated_articles = item.get("labels", []) or []
            label = 1 if violated_articles else 0
            for fact in facts:
                if fact and fact.strip():
                    records.append({"split": split, "text": fact, "label": label})
    df = pd.DataFrame(records)
    out_path = os.path.join(out_dir, "echr_facts_flattened.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote flattened facts CSV: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and flatten ECHR dataset to CSVs")
    parser.add_argument("--out_dir", type=str, default="./data/echr", help="Output directory for CSV files")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional HF cache directory")
    parser.add_argument("--use_auth_token", action="store_true", help="Pass use_auth_token=True to HF loader")
    parser.add_argument("--force_redownload", action="store_true", help="Force redownload of dataset")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading ecthr_cases dataset from Hugging Face (trust_remote_code=True)...")
    load_kwargs = {"trust_remote_code": True}
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir
    if args.use_auth_token:
        load_kwargs["use_auth_token"] = True
    if args.force_redownload:
        load_kwargs["download_mode"] = "force_redownload"

    ds = load_dataset("ecthr_cases", **load_kwargs)

    print("Saving raw split CSVs...")
    save_splits(ds, args.out_dir)

    print("Creating flattened facts CSV...")
    flatten_facts(ds, args.out_dir)

    print("Done.")
