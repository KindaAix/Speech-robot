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

## 本次调试与修复（简要记录）

- 问题复现：在启用 LoRA（PEFT）并用 `Seq2SeqTrainer` 训练时，出现过一次性错误：
    `TypeError: WhisperForConditionalGeneration.forward() got an unexpected keyword argument 'input_ids'`。

- 排查步骤：
    1. 在 `scripts/train_whisper.py` 的 `DataCollatorSpeechSeq2Seq` 中打印返回的 keys 和形状，确认 collator 只返回 `['input_features','labels']`。
    2. 在 `Trainer` 创建完成后取 `trainer.get_train_dataloader()` 的第一批打印 keys，确认 DataLoader 也只产生 `['input_features','labels']`。
    3. 重写了 `WhisperSeq2SeqTrainer._sanitize_inputs` 和 `compute_loss`，添加 debug 日志，打印 compute_loss 接收到的 keys 与 kwargs（例如 `num_items_in_batch`）。

- 发现：在训练循环中，compute_loss 接收到的 inputs 只包含 `input_features` 与 `labels`，kwargs 中有 `num_items_in_batch`（Transformers 内部行为）。因此数据端没有多出的 `input_ids`。

- 临时/保守修复：

  - 在 `compute_loss` 中先过滤掉 inputs 中的非允许键（allowed set），然后把过滤后的字典作为 `call_kwargs` 传给模型 wrapper（`model(**call_kwargs)`）。
  - 另外在 `compute_metrics` 中确保将 predictions/labels 移动到 CPU 并复制后再做 in-place 替换与解码，避免在评估时占用 GPU 内存导致 OOM。

- 结果：在 conda 环境中重新运行脚本，训练能正常进行并产生 loss，未再复现一次性报错；并且评估路径的内存更安全（限制生成长度与 beam，且在 CPU 上解码）。

注：长期方案建议升级/对齐 `transformers` 与 `peft` 的兼容版本，或在 PEFT wrapper 层面改造 forward 以安全接收并忽略额外 kwargs（需要修改 peft 源码或等待 upstream 修复）。
但是 Trainer 可能会把 tokenizer 也加进去，把 input_ids 传给模型 → 导致 forward() 收到 input_ids → 报错。

1. 解决方案
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
