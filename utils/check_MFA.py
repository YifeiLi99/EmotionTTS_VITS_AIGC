from pathlib import Path
import chardet

# ===== 用户配置部分 =====
USE_SUBFOLDERS = False  # 如果是 wav/ 和 lab/ 混在一起，请设为 False
ROOT_DIR = Path("D:/lyf/EmotionTTS_VITS_AIGC/data/processed/EMOVIE_DATASET/mfa_train_data")
DICT_PATH = Path("C:/Users/admin/Documents/MFA/mfa_train_data/dictionary/1_mandarin_mfa/words.txt")

if USE_SUBFOLDERS:
    WAV_DIR = ROOT_DIR / "wav"
    LAB_DIR = ROOT_DIR / "lab"
else:
    WAV_DIR = LAB_DIR = ROOT_DIR

assert WAV_DIR.exists(), f"WAV 路径不存在: {WAV_DIR}"
assert LAB_DIR.exists(), f"LAB 路径不存在: {LAB_DIR}"

# 收集文件名
wav_files = {p.stem for p in WAV_DIR.glob("*.wav")}
lab_files = {p.stem for p in LAB_DIR.glob("*.lab")}

# 找出缺失文件
missing_labs = wav_files - lab_files
missing_wavs = lab_files - wav_files

print("========== 文件匹配检查 ==========")
print(f"✅ WAV 文件数量: {len(wav_files)}")
print(f"✅ LAB 文件数量: {len(lab_files)}")
print(f"❌ 缺失 .lab 文件: {len(missing_labs)} 条，如: {list(missing_labs)[:5]}")
print(f"❌ 缺失 .wav 文件: {len(missing_wavs)} 条，如: {list(missing_wavs)[:5]}")

# 加载合法汉字集合
with open(DICT_PATH, 'r', encoding='utf-8') as f:
    dict_words = set(line.strip().split()[0] for line in f if not line.startswith('<'))

# 检查 lab 文件内容
print("\n========== .lab 内容合法性检查 ==========")
illegal_labs = []
empty_labs = []
bad_encoding = []

for lab in LAB_DIR.glob("*.lab"):
    raw_bytes = lab.read_bytes()
    detected = chardet.detect(raw_bytes)
    encoding = detected['encoding']
    try:
        text = raw_bytes.decode('utf-8').strip()
        if not text:
            empty_labs.append(lab.name)
        else:
            chars = text.split()
            illegal_chars = [c for c in chars if c not in dict_words]
            if illegal_chars:
                illegal_labs.append((lab.name, text, illegal_chars))
    except UnicodeDecodeError as e:
        bad_encoding.append((lab.name, encoding))

print(f"❌ 空白 .lab 文件: {len(empty_labs)}")
if empty_labs:
    print("--- 空白文件列表 ---")
    for name in empty_labs:
        print(f"[空白] {name}")

print(f"\n❌ 包含词典缺失字符的 .lab 文件: {len(illegal_labs)}")
if illegal_labs:
    print("--- 缺失字符文件列表（全部列出） ---")
    for name, text, missing in illegal_labs:
        print(f"[缺失] {name}: {text} ｜ 缺失字: {missing}")

print(f"\n❌ 非 UTF-8 编码 .lab 文件: {len(bad_encoding)}")
if bad_encoding:
    print("--- 编码异常文件列表（全部列出） ---")
    for name, reason in bad_encoding:
        print(f"[编码异常] {name}: {reason}")

print("\n✅ 如果全部为 0，即可安全进行对齐任务。")
