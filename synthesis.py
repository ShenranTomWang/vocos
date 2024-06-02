from pathlib import Path
import datetime as dt

import numpy as np
import soundfile as sf
import torch

# Vocos imports
from vocos.pretrained import Vocos
from vocos.dataset import VocosDataset
print("Import complete")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_FOLDER = "./synth_output-vocos"
VOCOS_CONFIG = "./configs/vocos-matcha.yaml"
VOCOS_CHECKPOINT = "./logs/lightning_logs/version_5/checkpoints/last.ckpt"

def load_vocoder(config_path, checkpoint_path):
    vocos = Vocos.from_checkpoint(config_path, checkpoint_path)
    print(f"Model loaded!")
    return vocos
vocoder = load_vocoder(VOCOS_CONFIG, VOCOS_CHECKPOINT)

def save_to_folder(filename: str, output, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')

@torch.inference_mode()
def to_waveform(wav, vocoder):
    audio = vocoder(wav)
    return audio

dataset = VocosDataset.from_config(VOCOS_CONFIG)

for i, (input, target) in dataset:
    output = to_waveform(input, vocoder)
    print(f"Saving output {i}...")
    save_to_folder(i, output, OUTPUT_FOLDER)
    print(f"Output {i} saved to {OUTPUT_FOLDER}")
