import os
import csv
import datasets
import pandas as pd
from datasets import Dataset
import hashlib
import base64
import numpy as np
from PIL import Image
import io
from tqdm import tqdm

from_wit_data_path = "/data/wit/image_data_train/image_pixels"
to_wit_data_path = "/data/wit/image_data_train/images"

# use multiprocessing to process csv files in parallel
datasets.set_caching_enabled(False)
import multiprocessing
from multiprocessing import Pool


# load all csv files under from_wit_data_path into huggingface datasets
# for csv_file in tqdm(os.listdir(from_wit_data_path)):
def process_csv(csv_file):
    if not csv_file.endswith(".csv") and not csv_file.endswith(".tsv"):
        return
    csv_file_path = os.path.join(from_wit_data_path, csv_file)
    print(csv_file_path)
    ds = Dataset.from_csv(
        csv_file_path, delimiter="\t", names=["image_url", "b64_bytes", "metadata_url"]
    )
    print("loaded", csv_file_path, len(ds))

    def img_from_base64(imagestring):
        try:
            img = Image.open(io.BytesIO(base64.b64decode(imagestring)))
            return img.convert("RGB")
        except ValueError:
            return None

    def save_to_images(example):
        image_id = hashlib.md5(example["image_url"].encode()).hexdigest()
        save_image_path = os.path.join(to_wit_data_path, f"{image_id}.jpg")
        # if os.path.exists(save_image_path):
        #     return example

        try:
            # decode base64 bytes
            b64_bytes = example["b64_bytes"]
            image_data = img_from_base64(b64_bytes)
            if image_data is None:
                print("error decoding image", example["image_url"])
            else:
                if image_data.size[0] < 10 or image_data.size[1] < 10:
                    print("image too small", example["image_url"], image_data.size)
                    # return example
                # print(image_data.size)
                # image_data.save(save_image_path)
        except Exception as e:
            print("error saving image", example["image_url"], e)

        return example

    ds = ds.map(save_to_images, num_proc=32, keep_in_memory=True)


if __name__ == "__main__":
    with Pool(1) as p:
        print(p.map(process_csv, os.listdir(from_wit_data_path)))
