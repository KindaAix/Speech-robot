import argparse
import json
import os
from typing import List, Dict

import torch
from torch.utils.data import Dataset

import librosa

from transformers import (
	WhisperProcessor,
	WhisperForConditionalGeneration,
	TrainingArguments,
	Trainer,
)


class WhisperDataset(Dataset):
	"""PyTorch Dataset for Whisper finetuning using a simple jsonl manifest.

	Each line in the jsonl should be a JSON object with at least:
	  - audio_filepath: absolute or relative path to wav file
	  - text: transcription string

	The dataset returns pre-extracted features and tokenized labels.
	"""

	def __init__(self, manifest_path: str, processor: WhisperProcessor, audio_sr: int = 16000):
		self.processor = processor
		self.audio_sr = audio_sr
		self.samples: List[Dict] = []
		with open(manifest_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					obj = json.loads(line)
				except json.JSONDecodeError:
					continue
				# Expect keys 'audio_filepath' and 'text'
				if 'audio_filepath' in obj and 'text' in obj:
					self.samples.append({'audio': obj['audio_filepath'], 'text': obj['text']})

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx: int):
		sample = self.samples[idx]
		audio_path = sample['audio']
		transcription = sample['text']

		# Load audio
		try:
			audio, sr = librosa.load(audio_path, sr=self.audio_sr)
		except Exception as e:
			raise RuntimeError(f"Failed loading {audio_path}: {e}")

		# Processor.feature_extractor expects a list/array of floats
		input_features = self.processor.feature_extractor(audio, sampling_rate=self.audio_sr).input_features[0]

		# Tokenize transcription into labels (list[int])
		# WhisperProcessor may not have `as_target_processor()` in some transformers versions.
		# Prefer using tokenizer(..., text_target=...) when available, otherwise fall back.
		try:
			# new tokenizers support text_target so we can get label ids directly
			tokenized = self.processor.tokenizer(transcription, text_target=transcription) if 'text_target' in self.processor.tokenizer.__call__.__code__.co_varnames else None
		except Exception:
			tokenized = None

		if tokenized is not None:
			labels_ids = tokenized['input_ids']
		else:
			# Fallback: call tokenizer directly; many tokenizers expect to be used under as_target_processor,
			# but calling tokenizer on the string still returns input_ids for the text.
			labels_ids = self.processor.tokenizer(transcription).input_ids

		return {'input_features': input_features, 'labels': labels_ids}


def data_collator(batch: List[Dict], processor: WhisperProcessor):
	"""Collate list of examples into batch dict suitable for WhisperForConditionalGeneration.

	Returns dict with 'input_features' and 'labels' tensors.
	"""
	import numpy as np

	input_features = [ex['input_features'] for ex in batch]
	labels = [ex['labels'] for ex in batch]

	input_features = torch.tensor(np.stack(input_features), dtype=torch.float32)

	# Pad labels using the tokenizer
	padded = processor.tokenizer.pad({'input_ids': labels}, padding=True, return_tensors='pt')
	labels_tensor = padded['input_ids']

	# Replace padding ids with -100 as ignored index for loss
	labels_tensor[labels_tensor == processor.tokenizer.pad_token_id] = -100

	return {'input_features': input_features, 'labels': labels_tensor}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--manifest', type=str, default='dataset/data.jsonl', help='Path to jsonl manifest')
	parser.add_argument('--model_name', type=str, default='openai/whisper-medium', help='Pretrained Whisper model')
	parser.add_argument('--output_dir', type=str, default='outputs/whisper_finetuned', help='Output dir for checkpoints/logs')
	parser.add_argument('--per_device_train_batch_size', type=int, default=8)
	parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
	parser.add_argument('--num_train_epochs', type=int, default=3)
	parser.add_argument('--learning_rate', type=float, default=1e-5)
	parser.add_argument('--logging_steps', type=int, default=100)
	parser.add_argument('--save_steps', type=int, default=500)
	parser.add_argument('--max_train_samples', type=int, default=None)
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

	print('Loading processor...')
	processor = WhisperProcessor.from_pretrained(args.model_name)

	print('Loading model...')
	model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

	# If the model has a forced_decoder_ids or generation config, you might want to set language/tokens here.
	# Example: processor.tokenizer.set_prefix_tokens_for_generation(language='zh') is not a real API here,
	# but users can set model.config.forced_decoder_ids if needed.

	print('Loading dataset...')
	dataset = WhisperDataset(args.manifest, processor)

	if args.max_train_samples:
		dataset.samples = dataset.samples[: args.max_train_samples]

	# Simple split: small eval split
	n = len(dataset)
	train_size = int(0.95 * n)
	train_dataset = torch.utils.data.Subset(dataset, list(range(0, train_size)))
	eval_dataset = torch.utils.data.Subset(dataset, list(range(train_size, n))) if n - train_size > 0 else None

	def collate_fn(batch):
		return data_collator(batch, processor)

	training_args = TrainingArguments(
		output_dir=args.output_dir,
		per_device_train_batch_size=args.per_device_train_batch_size,
		per_device_eval_batch_size=args.per_device_eval_batch_size,
		num_train_epochs=args.num_train_epochs,
		learning_rate=args.learning_rate,
		logging_steps=args.logging_steps,
		save_steps=args.save_steps,
		eval_strategy='steps' if eval_dataset is not None else 'no',
		eval_accumulation_steps=50,
		save_total_limit=3,
		fp16=torch.cuda.is_available(),
		report_to=['none'],
		remove_unused_columns=False,
		push_to_hub=False,
		load_best_model_at_end=False,
		logging_dir=os.path.join(args.output_dir, 'logs'),
	)

	# Trainer will call model(input_features=..., labels=...)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=collate_fn,
	)

	print('Starting training...')
	trainer.train()

	print('Saving final model...')
	trainer.save_model(args.output_dir)


if __name__ == '__main__':
	main()
