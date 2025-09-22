# ASR

1. model: Whisper Small
2. data: magichub shanghai/sichuan
3. 下面是项目中与 ASR 微调与推理相关的 Python 脚本简要说明：

- `scripts/inspect_data.py`：快速校验脚本，读取 `dataset/data.jsonl`，检查每行是否包含 `audio_filepath` 和 `text`，并尝试用 `soundfile` 打开音频文件以确保文件可读。输出示例：`Total entries: 9870, Bad entries: 0`。

- `scripts/train_whisper.py`：微调模板脚本（起点）。演示如何使用 `transformers` 的 `WhisperProcessor` 和 `WhisperForConditionalGeneration` 来加载模型并对 dataset 进行简单预处理（将音频转换为特征并把文本 token 化）。当前脚本会将处理后的 dataset 和模型/processor 保存到指定的 `--output_dir`，便于后续使用 `Trainer` 或 `accelerate` 实施正式训练。

- `inference/whisper_api.py`：推理示例，演示如何加载保存的模型/processor 并对单个 `.wav` 文件进行转录（含重采样步骤）。可直接在 CLI 中运行：

"""
bash
python inference/whisper_api.py <model_dir> <audio.wav>
"""

- 先运行 `scripts/inspect_data.py` 验证 dataset：

 """
 bash
 conda run --name kinda python scripts/inspect_data.py
 """

- 使用 `scripts/train_whisper.py` 做数据预处理并保存（示例）：

 """
 bash
 conda run --name kinda python scripts/train_whisper.py --jsonl dataset/data.jsonl --output_dir outputs/whisper_prep
 """
