import argparse
import json
import pathlib
import random

import torch
import numpy as np
import webdataset as wds
from datasets import load_dataset

# from speech_quantization import quantize


# All 10 FLEURS language configs
FLEURS_LANGUAGES = [
    "ar_eg",
    "cmn_hans_cn",
    "fi_fi", "fil_ph",
    "hi_in", 
    "ja_jp", 
    "ko_kr",
    "ta_in", "tr_tr",
    "yo_ng",
]


def write_shards(
        shards_path: pathlib.Path,
        seed: int,
        samples_per_shard: int,
):
    shards_path.mkdir(exist_ok=True, parents=True)

    data_samples = []

    for lang_code in FLEURS_LANGUAGES:
        print(f"\nLoading language: {lang_code}")
        try:
            dataset = load_dataset("google/fleurs", lang_code, trust_remote_code=True)
        except Exception as e:
            print(f"  Skipping {lang_code}: {e}")
            continue

        split = dataset.get("train", None)
        if split is None:
            print(f"  No train split for {lang_code}, skipping.")
            continue

        for i, sample in enumerate(split):
            try:
             audio = np.array(sample["audio"]["array"], dtype=np.float32)
            except Exception as e:
                print(f"Failed Sample : {sample}")
                print(f"Error : {e}")
                continue
            # audio_quantized = quantize(audio, scheme=1)

            data_samples.append({
                "__key__": f"{lang_code}_{i}",
                "audio.pth": torch.tensor(audio, dtype=torch.float32),
                "language_id": lang_code,
            })

        print(f"  {lang_code}: {i + 1} samples added")

    print(f"\nTotal samples: {len(data_samples)}")

    # Write meta
    meta_dict = {
        "language_ids": FLEURS_LANGUAGES,
        "num_data_samples": len(data_samples),
    }
    with (shards_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    # Shuffle and write shards
    pattern = str(shards_path / "shard") + "-%06d.tar"
    random.seed(seed)
    random.shuffle(data_samples)

    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
        for sample in data_samples:
            sink.write(sample)

    print("Done writing shards.")


# CLI
parser = argparse.ArgumentParser(
    description="Convert FLEURS (all 102 languages) to WebsgDataset shards"
)
parser.add_argument(
    "shards_path", type=pathlib.Path, help="directory to write shards to"
)
parser.add_argument(
    "--seed", type=int, default=12345,
    help="random seed for shuffling before writing shards",
)
parser.add_argument(
    "--samples_per_shard", type=int, default=5000,
    help="max samples per shard tar file",
)

if __name__ == "__main__":
    args = parser.parse_args()
    write_shards(args.shards_path, args.seed, args.samples_per_shard)
