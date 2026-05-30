"""
    将所有的通用配置放入到这个文件中
"""

from pathlib import Path # 找路径

ROOT_DIR = Path(__file__).parent.parent
# 数据目录
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

MAX_SEQ_LEN = 128
SEQ_LEN = 128
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256 # 理论上要比 embedding_dim 大
LEARNING_RATE = 1e-3
EPOCHS = 10