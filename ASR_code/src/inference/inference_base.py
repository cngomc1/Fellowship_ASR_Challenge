import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os

def transcribe_audio(model, processor, path):
    speech, sr = torchaudio.load(path)
    speech = torchaudio.functional.resample(speech, sr, 16000).squeeze()
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    logits = model(inputs.input_values).logits
    pred_ids = logits.argmax(-1)
    return processor.decode(pred_ids[0])

def run_inference(test_dir, output_file):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

    with open(output_file, "w", encoding="utf8") as f:
        for file in sorted(os.listdir(test_dir)):
            if file.endswith(".wav"):
                text = transcribe_audio(model, processor, os.path.join(test_dir, file))
                f.write(f"{file}\t{text}\n")

if __name__ == "__main__":
    run_inference("data/test", "base_transcriptions.txt")
