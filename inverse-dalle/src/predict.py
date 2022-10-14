from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from data import ImageToTextDataset


def decode_to_text(sequence: list[int], vocab: list[str]) -> str:
    text = ""
    for token_id in sequence:
        if token_id >= len(vocab) or vocab[token_id] in ["<s>", "<pad>"]:
            continue
        elif vocab[token_id] == "</s>":
            break
        text += vocab[token_id]
    return text.strip()


def calculate_confidence(
    sequence: list[int],
    sequence_index: int,
    scores: list[torch.Tensor],
    eos_token_id: int,
) -> float:
    confidence = 0
    for i, token_id in enumerate(sequence):
        if token_id == eos_token_id:
            break
        elif token_id >= scores[i].size(1):
            continue
        confidence += scores[i][sequence_index, token_id].float()
    return confidence if isinstance(confidence, int) else confidence.exp().item()


@torch.no_grad()
def main(args: argparse.Namespace):
    with open(os.path.join(args.model, "vocab.txt")) as fp:
        vocab = [token.strip("\n") for token in fp.readlines()]
        eos_token_id = vocab.index("</s>")
        pad_token_id = vocab.index("<pad>")

    dataset = ImageToTextDataset(
        dataset=pd.read_csv(args.inputs).drop("text", axis=1, errors="ignore"),
        vocab=vocab,
        max_visual_length=args.max_visual_length,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.half().cuda().eval()

    predictions, filenames = [], dataset.dataset["img_path"].to_list()
    for batch in tqdm.tqdm(dataloader):
        output = model.generate(
            batch["input_ids"].cuda(),
            max_new_tokens=args.max_length - 1,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        scores = [F.log_softmax(x[:, : len(vocab)], dim=1) for x in output.scores]
        sequences = output.sequences[:, batch["input_ids"].size(1) :].tolist()

        for i, sequence in enumerate(sequences):
            example = (
                filenames.pop(0),
                decode_to_text(sequence, vocab),
                calculate_confidence(sequence, i, scores, eos_token_id),
            )
            predictions.append(dict(zip(["img_path", "text", "confidence"], example)))

    prediction = pd.DataFrame(predictions)
    if args.merge_tta:
        counts = prediction.groupby("img_path").confidence.count()
        merged = prediction.groupby(["img_path", "text"]).confidence.sum()
        merged = merged.reset_index(name="confidence")

        prediction = pd.merge(merged, counts, left_on="img_path", right_index=True)
        prediction["confidence"] = prediction.confidence_x / prediction.confidence_y
        prediction = prediction[["img_path", "text", "confidence"]]
    prediction.to_csv("prediction.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("inputs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--max-visual-length", type=int, default=128)
    parser.add_argument("--merge-tta", action="store_true", default=False)
    main(parser.parse_args())
