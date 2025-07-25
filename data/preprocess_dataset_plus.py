"""
一键把原始结构的数据处理成可用于dataloader的数据结构

"""
import json
import csv
from config import RAW_DATA_DIR, NOW_DATASET, PROCESSED_DATASET_DIR, RAW_LABELS_FILE
from config import MFA_ZIP, MFA_DICT
from tqdm import tqdm
import shutil
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor
import random
import re


# ======================== 采样率与频道统一 ========================
def check_ffmpeg():
    """检查 ffmpeg 是否可用"""
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("❌ ffmpeg 未安装或未加入系统环境变量 PATH")


def _convert_single_wav(wav_file: Path, out_path: Path, sr: int, channels: int) -> bool:
    """处理单个音频文件的转换逻辑"""
    if out_path.exists():
        return True  # 已存在，视为成功跳过
    command = [
        "ffmpeg", "-y",
        "-i", str(wav_file),
        # 单声道
        "-ac", str(channels),
        # 采样率 16kHz
        "-ar", str(sr),
        str(out_path)
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0  # True 表示成功


def convert_wavs(src_dir: Path, dst_dir: Path, sr: int = 16000, channels: int = 1, num_workers: int = 8):
    """
    将 src_dir 中的 .wav 音频转换为指定采样率和通道数，保存至 dst_dir
    支持多进程加速处理（默认开启8个并发进程）
    """
    wav_files = list(src_dir.glob("*.wav"))
    tasks = []
    for wav_file in wav_files:
        out_path = dst_dir / wav_file.name
        tasks.append((wav_file, out_path, sr, channels))
    success_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_convert_single_wav, *task) for task in tasks]
        for i, future in enumerate(tqdm(futures, desc="🎧 转换音频", total=len(futures))):
            success = future.result()
            if not success:
                print(f"⚠️ 转换失败: {tasks[i][0].name}")
            else:
                success_count += 1
    return success_count


# ======================== JSONL文件编写 ========================
def save_polarity_map(label_map_path: Path):
    """
    保存情绪标签与极性数值的映射表。
    当前仅支持五类，未来可扩展为更多细致情绪。
    """
    emotion_to_polarity = {
        "angry": -1.0,
        "sad": -0.5,
        "neutral": 0.0,
        "happy": 0.5,
        "excited": 1.0
    }
    # 保存 emotion → polarity（只用于可视化）
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(emotion_to_polarity, f, ensure_ascii=False, indent=2)
    print(f"✅ 情绪标签映射表已保存到: {label_map_path}")
    # 构造 polarity → emotion 的映射，用于实际判断
    polarity_to_emotion = {f"{v:.1f}": k for k, v in emotion_to_polarity.items()}
    return polarity_to_emotion


def convert_labels_to_jsonl(wav_dir: Path, csv_path: Path, jsonl_path: Path, polarity_map: dict):
    """读取 CSV 标签并结合 wav_dir 生成 JSONL 数据集"""
    num_written = 0
    with open(csv_path, 'r', encoding='utf-8') as csvfile, \
            open(jsonl_path, 'w', encoding='utf-8') as jsonlfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            wav_filename = row['filename']

            def remove_chinese_punctuation(text: str) -> str:
                """
                移除中文及全角标点符号
                """
                punctuation_pattern = r"[。？！，、；：“”‘’（）《》〈〉【】『』「」﹏…—～·]"
                return re.sub(punctuation_pattern, "", text)

            # 处理文本内容
            text = remove_chinese_punctuation(row['text'].strip())
            try:
                polarity_float = float(row['polarity_label'])
                polarity_key = f"{polarity_float:.1f}"
            except Exception as e:
                print(f"[错误] 行解析失败，文件: {wav_filename}，异常: {e}，已跳过")
                continue
            if polarity_key not in polarity_map:
                print(f"[警告] 未知极性标签: {polarity_key}，文件名: {wav_filename}，已跳过")
                continue
            emotion_class = polarity_map[polarity_key]
            wav_path = (wav_dir / wav_filename).as_posix()

            sample = {
                "text": text,
                "emotion_polarity": polarity_float,
                "emotion_class": emotion_class,
                "wav_path": wav_path
            }

            jsonlfile.write(json.dumps(sample, ensure_ascii=False) + '\n')
            num_written += 1
    print(f"✅ 处理完成，共写入 {num_written} 条样本到: {jsonl_path}")


# ======================== 划分数据集 ========================
def split_jsonl_dataset(source_jsonl: Path, save_dir: Path, train_ratio: float = 0.7, val_ratio: float = 0.2,
                        test_ratio: float = 0.1, seed: int = 42):
    """
    将一个 JSONL 格式的完整数据集划分为 train / val / test 三个子集
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例加起来应为1"
    # ========== 加载原始 JSONL 数据 ==========
    with open(source_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"原始数据总数: {len(lines)} 条")
    # ========== 随机打乱并划分 ==========
    random.seed(seed)
    random.shuffle(lines)
    total = len(lines)
    num_train = int(total * train_ratio)
    num_val = int(total * val_ratio)
    num_test = total - num_train - num_val  # 剩余全部划入 test
    # 拆分三部分数据
    train_data = lines[:num_train]
    val_data = lines[num_train:num_train + num_val]
    test_data = lines[num_train + num_val:]

    #  保存函数
    def save_jsonl(filepath: Path, data):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(data)
        print(f"✅ 已保存 {filepath}（{len(data)} 条样本）")

    #  保存至目标目录
    save_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(save_dir / "train.jsonl", train_data)
    save_jsonl(save_dir / "val.jsonl", val_data)
    save_jsonl(save_dir / "test.jsonl", test_data)


# ======================== 转化MFA适用格式 ========================
def convert_jsonl_to_mfa_format(jsonl_path: Path, output_dir: Path):
    """
    将 jsonl 文件转换为 MFA 格式的 wav + lab 文件。
    文件编号统一为 0000.wav / 0000.lab 格式。
    """
    # 创建输出目录
    # MFA一定要把wav和lab放到一起啊！！！
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc=f"转换中: {jsonl_path.name}")):
            data = json.loads(line)
            audio_path = Path(data["wav_path"]).resolve(strict=False)
            text = data["text"]
            if not audio_path.exists():
                print(f"⚠️ 音频文件不存在: {audio_path}，已跳过")
                continue
            file_id = f"{idx:04d}"
            # 拷贝音频
            wav_dst = output_dir / f"{file_id}.wav"
            shutil.copy(audio_path, wav_dst)
            # 写入 lab 文件
            lab_dst = output_dir / f"{file_id}.lab"
            with open(lab_dst, "w", encoding="utf-8") as lab_file:
                lab_file.write(text.strip())


# ======================== MFA对齐 ========================
def run_mfa_align(mfa_input_dir: Path, dict, model, output_dir: Path):
    """
    调用 MFA 对音频 + 文本进行强制对齐
    """
    command = [
        "mfa", "align",
        str(mfa_input_dir),
        str(dict),
        str(model),
        str(output_dir),
        "--clean", "--verbose"
    ]
    print(f"🚀 正在运行 MFA 对齐命令: {' '.join(command)}")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print(f"✅ MFA 对齐完成，输出目录: {output_dir}")
    else:
        print(f"❌ MFA 对齐失败:\n{result.stderr}")


# ======================== 111 ========================


# ======================== 主函数 ========================
def main():
    # 转码输出目录
    dst_dir = NOW_DATASET / "wavs16k"
    dst_dir.mkdir(parents=True, exist_ok=True)
    # 检测转换功能是否可用
    check_ffmpeg()
    # 转换采样率
    num_converted = convert_wavs(RAW_DATA_DIR, dst_dir, num_workers=8)
    print(f"\n✅ 音频转换完成，共处理 {num_converted} 条音频。\n")

    # 保存jsonl
    jsonl_path = PROCESSED_DATASET_DIR / "metadata_emovie.jsonl"
    # 保存情绪映射
    label_map_path = PROCESSED_DATASET_DIR / "polarity_map.json"
    # 处理情感映射
    polarity_map = save_polarity_map(label_map_path)
    # 处理音频文字对应JSONL
    convert_labels_to_jsonl(dst_dir, RAW_LABELS_FILE, jsonl_path, polarity_map)

    # 划开数据集为三种
    split_jsonl_dataset(
        source_jsonl=jsonl_path,
        save_dir=PROCESSED_DATASET_DIR
    )

    # 处理训练集
    train_jsonl = PROCESSED_DATASET_DIR / "train.jsonl"
    train_mfa_out_dir = PROCESSED_DATASET_DIR / "mfa_train_data"
    convert_jsonl_to_mfa_format(train_jsonl, train_mfa_out_dir)
    # 处理验证集
    val_jsonl = PROCESSED_DATASET_DIR / "val.jsonl"
    val_mfa_out_dir = PROCESSED_DATASET_DIR / "mfa_val_data"
    convert_jsonl_to_mfa_format(val_jsonl, val_mfa_out_dir)

    # 运行 MFA
    mfa_output_val = PROCESSED_DATASET_DIR / "mfa_output_val"
    mfa_output_val.mkdir(parents=True, exist_ok=True)
    run_mfa_align(
        mfa_input_dir=val_mfa_out_dir,  # 包含 wav/lab 的路径
        dict=MFA_DICT,  # ✅ 直接用模型名字符串
        model=MFA_ZIP,  # ✅ 直接用模型名字符串
        output_dir=mfa_output_val  # 输出 TextGrid 路径
    )
    mfa_output_train = PROCESSED_DATASET_DIR / "mfa_output_train"
    mfa_output_train.mkdir(parents=True, exist_ok=True)
    run_mfa_align(
        mfa_input_dir=train_mfa_out_dir,  # 包含 wav/lab 的路径
        dict=MFA_DICT,  # ✅ 直接用模型名字符串
        model=MFA_ZIP,  # ✅ 直接用模型名字符串
        output_dir=mfa_output_train  # 输出 TextGrid 路径
    )


# ======================== 一键执行 ========================
if __name__ == "__main__":
    main()
