import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from src.models.adapters import BottleneckAdapter
from src.inference.inference_base import transcribe_audio
import os

def load_finetuned_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

    for layer in model.wav2vec2.encoder.layers:
        layer.adapter = BottleneckAdapter()

    state = torch.load("weights/adapters/adapter_weights.pth")
    model.load_state_dict(state, strict=False)

    return model, processor

def run_inference(test_dir, output_file):
    model, processor = load_finetuned_model()

    with open(output_file, "w", encoding="utf8") as f:
        for file in sorted(os.listdir(test_dir)):
            if file.endswith(".wav"):
                text = transcribe_audio(model, processor, os.path.join(test_dir, file))
                f.write(f"{file}\t{text}\n")

if __name__ == "__main__":
    run_inference("data/test", "finetuned_transcriptions.txt")
