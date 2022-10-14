from __future__ import annotations

import argparse
import glob
import json
import os
import random

import cv2
import pandas as pd
import tqdm


def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = []
    filenames = glob.glob(os.path.join(glob.escape(args.json_dir), "*.json"))

    for filename in tqdm.tqdm(filenames):
        with open(filename) as fp:
            data = json.load(fp)
        images = {}
        for image in data["images"]:
            img = cv2.imread(os.path.join(args.image_dir, image["file_name"]))
            images[image["id"]] = img
        for annot in data["annotations"]:
            if (
                annot["text"] == "xxx"
                or any(x is None for x in annot["bbox"])
                or annot["image_id"] is None
                or images[annot["image_id"]] is None
            ):
                continue

            x, y, w, h = annot["bbox"]
            img = images[annot["image_id"]][y : y + h, x : x + w]
            if img.size == 0:
                continue

            output = "".join(random.choices("0123456789abcdef", k=16)) + ".jpg"
            output = os.path.join(args.output_dir, output)

            cv2.imwrite(output, img)
            dataset.append({"img_path": output, "text": annot["text"]})

    pd.DataFrame(dataset).to_csv(
        args.output_dataset or (os.path.basename(args.json_dir) + ".csv"),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir")
    parser.add_argument("image_dir")
    parser.add_argument("--output-dir", default="./external/")
    parser.add_argument("--output-dataset")
    main(parser.parse_args())
