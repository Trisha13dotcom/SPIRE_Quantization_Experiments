import argparse
import json
import pathlib
import random
import re
from collections import defaultdict

import torch
import webdataset as wds

from speechbrain.dataio import audio_io
from pathlib import Path

from datasets import load_dataset
from pathlib import Path
import numpy as np

from speech_quantization import quantize


def write_shards(
        shards_path: pathlib.Path,
        seed: int,
        samples_per_shard: int,
):
     shards_path.mkdir(exist_ok=True, parents=True)
    
     en = load_dataset("google/fleurs", "en_us")
     hi = load_dataset("google/fleurs", "hi_in")

     data_samples = []

    


# English samples
     for i, sample in enumerate(en["train"]):
        audio = sample["audio"]["array"]
        audio = np.array(audio, dtype= np.float32)

        #Applying quantization
        audio_quantized = quantize(audio, scheme=1)

        #Printing the audio arrays to verify the quantization scheme is properly applied
        print("Original Audio English:", audio[:10])
        print("Quantized Audio English", audio_quantized[:10])
        
        data_samples.append({
         "__key__": f"en_{i}",
         "audio.pth": torch.tensor(
            audio_quantized, dtype = torch.float32
        ),
        "language_id": "en",
    })

# Hindi samples
     for i, sample in enumerate(hi["train"]):
        audio = sample["audio"]["array"]
        audio = np.array(audio, dtype = np.float32)

        #Applying quantization 
        audio_quantized = quantize(audio, scheme = 1)
        #Printing the audio arrays to verify the quantization scheme is properly applied
        print("Original Audio Hindi:", audio[:10])
        print("Quantized Audio Hindi:", audio_quantized[:10])
        
        data_samples.append({
         "__key__": f"hi_{i}",
         "audio.pth": torch.tensor(
            audio_quantized,
            dtype=torch.float32,
        ),
        "language_id": "hi",
    })
     meta_dict = {
    "language_ids": ["en", "hi"],
    "num_data_samples": len(data_samples),
}
     # write shards
     pattern = str(shards_path / "shard") + "-%06d.tar"

     with (shards_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_dict, f)
          
     random.seed(seed)
     random.shuffle(data_samples)

     

     with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
      for sample in data_samples:
        sink.write(sample)

#CLI
parser = argparse.ArgumentParser(
    description="Convert Fleurs to WebDataset shards"
)

parser.add_argument(
    "shards_path", type=pathlib.Path, help="directory to write shards to"
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="random seed used for shuffling data before writing to shard",
)
parser.add_argument(
    "--samples_per_shard",
    type=int,
    default=5000,
    help="the maximum amount of samples placed in each shard. The last shard "
    "will most likely contain fewer samples.",
)

################################################################################
# execute script

if __name__ == "__main__":
    args = parser.parse_args()

    write_shards(
        args.shards_path,
        args.seed,
        args.samples_per_shard,
    )

