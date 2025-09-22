"""Full training script using Hugging Face Trainer (Seq2SeqTrainer) for Whisper fine-tuning.

This script:
- Loads a JSONL dataset with fields `audio_filepath` and `text`.
- Preprocesses audio to Whisper input features and tokenizes text into labels.
- Uses `Seq2SeqTrainer` for training/evaluation and computes WER with `jiwer`.

Notes:
- Adjust hyperparameters and batch sizes to match your GPU memory.
"""
import os
import argparse
import soundfile as sf
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl', type=str, default='dataset/data.jsonl')
    p.add_argument('--model', type=str, default='openai/whisper-base')
    p.add_argument('--output_dir', type=str, default='outputs/whisper_finetuned')
    p.add_argument('--per_device_train_batch_size', type=int, default=8)
    p.add_argument('--per_device_eval_batch_size', type=int, default=2)
    p.add_argument('--learning_rate', type=float, default=1e-5)
    p.add_argument('--num_train_epochs', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    # LoRA / PEFT options
    p.add_argument('--use_lora', action='store_true', help='Enable LoRA fine-tuning via PEFT')
    p.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    p.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    p.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    p.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj',
                   help='Comma-separated target modules for LoRA (e.g. "q_proj,k_proj,v_proj")')
    return p.parse_args()


def prepare_dataset(jsonl_path, processor, sampling_rate=16000, split={'train':0.95,'validation':0.05}):
    # 1. 加载 jsonl，但不要 cast_column
    ds = load_dataset('json', data_files=jsonl_path, split='train')

    # 2. 切分
    if isinstance(split, dict):
        total = len(ds)
        n_val = max(1, int(total * split.get('validation', 0.05)))
        ds = ds.train_test_split(test_size=n_val, seed=42)
        ds = DatasetDict({'train': ds['train'], 'validation': ds['test']})

    # 3. map 函数：手动用 soundfile/ librosa 读 wav
    def speech_map(batch, processor, sampling_rate, max_target_length):
        speech, sr = sf.read(batch['audio_filepath'])
        if sr != sampling_rate:
            import librosa
            speech = librosa.resample(speech, orig_sr=sr, target_sr=sampling_rate)
        input_features = processor.feature_extractor(speech, sampling_rate=sampling_rate).input_features[0]
        labels = processor.tokenizer(batch['text']).input_ids
        if len(labels) > max_target_length:
            labels = labels[:max_target_length]
        return {'input_features': input_features, 'labels': labels}

    from functools import partial
    map_fn = partial(speech_map, processor=processor, sampling_rate=sampling_rate, max_target_length=128)

    ds_proc = {k: v.map(map_fn, remove_columns=['audio_filepath','text'], batched=False) for k, v in ds.items()}
    return DatasetDict(ds_proc)



class DataCollatorSpeechSeq2Seq:
    """Data collator to pad input_features and labels for seq2seq training."""

    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_features = [f['input_features'] for f in features]
        labels = [f['labels'] for f in features]
        batch = {}
        batch_inputs = self.processor.feature_extractor.pad(
            {'input_features': input_features}, return_tensors='pt'
        )
        # pad labels using tokenizer
        batch_labels = self.processor.tokenizer.pad(
            {'input_ids': labels}, return_tensors='pt'
        )
        # replace padding token id's in labels by -100 so they are ignored by loss
        labels_ids = batch_labels['input_ids'].masked_fill(batch_labels['attention_mask'] == 0, -100)
        batch['input_features'] = batch_inputs['input_features']
        batch['labels'] = labels_ids
        return batch


def compute_metrics(pred, processor):
    """Decode generated token ids and compute WER against labels.

    This function is intended to be used with `predict_with_generate=True`, so
    `pred.predictions` are the generated token ids from `model.generate()`.
    """
    wer_metric = evaluate.load('wer')
    preds = pred.predictions
    # some trainers return a tuple (preds, scores)
    if isinstance(preds, tuple):
        preds = preds[0]
    # decode predictions
    try:
        pred_str = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
    except Exception:
        # fallback: ensure numpy
        pred_str = processor.tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)

    # prepare labels: replace -100 with pad_token_id before decoding
    labels = pred.label_ids
    labels[np.where(labels == -100)] = processor.tokenizer.pad_token_id
    try:
        label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception:
        label_str = processor.tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {'wer': wer}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print('Loading processor and model:', args.model)
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    # optionally wrap with PEFT/LoRA
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except Exception as e:
            raise ImportError('peft not installed. Run: pip install peft')
        # validate target modules against model parameter names; if not found try to auto-detect common names
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(','),
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        print('Applying LoRA with config:', lora_config)
        model = get_peft_model(model, lora_config)

    print('Preparing dataset...')
    ds = prepare_dataset(args.jsonl, processor)
    print('Dataset sizes:', {k: len(v) for k, v in ds.items()})

    # data collator
    data_collator = DataCollatorSpeechSeq2Seq(processor=processor)

    # training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        fp16=torch.cuda.is_available(),
        save_total_limit=3,
        eval_strategy='steps',
        eval_steps=500,
        save_steps=200,
        logging_steps=100,
        remove_unused_columns=False,
        dataloader_num_workers=0
    )

    # trainer
    # wrap compute_metrics so it has access to processor
    def _compute(pred):
        return compute_metrics(pred, processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        data_collator=data_collator,
        processing_class=processor.tokenizer,
        compute_metrics=_compute,
    )

    # train
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
