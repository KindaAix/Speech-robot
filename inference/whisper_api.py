"""Simple inference example for loading saved Whisper model and processor and transcribing a file."""
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
import torch


def transcribe(model_dir, audio_path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    processor = WhisperProcessor.from_pretrained(model_dir)

    # load audio
    audio, sr = sf.read(audio_path)
    if sr != processor.feature_extractor.sampling_rate:
        import librosa
        audio = librosa.resample(audio, sr, processor.feature_extractor.sampling_rate)
    inputs = processor.feature_extractor(audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors='pt')
    input_features = inputs.input_features.to(device)
    generated_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python inference/whisper_api.py <model_dir> <audio.wav>')
        sys.exit(1)
    print(transcribe(sys.argv[1], sys.argv[2]))
