# Whisper + PEFT/LoRA + Transformers Trainer 的常见坑

1. 原因分析
a. Whisper 的 forward() 签名 和标准 Seq2Seq 不完全一样：
Whisper 不接受 input_ids，而 Trainer 默认会把 batch 里的 labels 和 input_ids 直接传给模型。
b. 现在的 data collator 返回的是
"""
batch = {
    "input_features": ...,
    "labels": ...
}
"""
但是 Trainer 可能会把 tokenizer 也加进去，把 input_ids 传给模型 → 导致 forward() 收到 input_ids → 报错。

2. 解决方案
a. 重写 Seq2SeqTrainer.compute_loss 或 model.forward 参数映射
可以自定义 Trainer
b. 确保 batch 中没有 input_ids
"""
batch = {
    "input_features": batch_inputs["input_features"],
    "labels": labels_ids
}
"""
不要给 Trainer 传 tokenizer（或者传 None），这样 Trainer 就不会把 input_ids 自动加进去。
