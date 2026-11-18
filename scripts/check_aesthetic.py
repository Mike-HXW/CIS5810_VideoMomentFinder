"""Manual check: print the Aesthetic Predictor V2.5 score for one image.

Usage: python scripts/check_aesthetic.py [image]   (default: input/person.jpg)
Not a unit test: it loads (and may download) the full model.
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('image', nargs='?', default=str(PROJECT_DIR / 'input' / 'person.jpg'))
    args = parser.parse_args()

    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

    print('Loading model. The first run requires a download...')
    model, preprocessor = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True)
    model = model.to(device='cpu', dtype=torch.float32).eval()

    with Image.open(args.image) as image:
        pixel_values = preprocessor(images=image.convert('RGB'), return_tensors='pt').pixel_values
    with torch.inference_mode():
        score = model(pixel_values.to(device='cpu', dtype=torch.float32)).logits.squeeze().item()
    print(f'Aesthetic score: {score:.2f}')


if __name__ == '__main__':
    main()
