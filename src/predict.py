from transformers import pipeline
import torch
from argparse import ArgumentParser


def make_prediction(model_path, device):
    return pipeline("text-classification", model=model_path, device=device)

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type="str")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    make_prediction(args.model_path, device)
    