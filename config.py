# -*- coding: utf-8 -*-
"""
Level2-Insight Agent - Global Configuration Center
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load optional .env file for local development
load_dotenv(BASE_DIR / ".env")

# Core Directory Layout (Pure English Standard)
FEATURES_DIR = BASE_DIR / "features"
TRAIN_FEAT_DIR = FEATURES_DIR / "train_features"
MODEL_TRAIN_DIR = BASE_DIR / "training"
MODEL_FILES_DIR = BASE_DIR / "models"
INFER_DIR = BASE_DIR / "inference"
INFER_FEAT_DIR = FEATURES_DIR / "infer_features" if (FEATURES_DIR / "infer_features").exists() else INFER_DIR / "infer_features"
OUTPUT_DIR = BASE_DIR / "output"
SUBMISSION_DIR = BASE_DIR / "submission"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODEL_FILES_DIR, exist_ok=True)

RANDOM_STATE = 42

# =====================================================================
# LLM API Configuration (OpenAI Compatible: DeepSeek / Qwen / OpenAI)
# Loaded from environment variables or .env file (Safe for Public Git)
# =====================================================================
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# FastDTW Clustering Hyperparameters (Task 1)
CLUSTER_K_MIN = 8
CLUSTER_K_MAX = 30
CLUSTER_K_PREF = 18
MAX_SINGLETONS = 5

WDTW_WEIGHT_W = 0.70
WDTW_WEIGHT_SIL = 0.15
WDTW_WEIGHT_CH = 0.15

# Ensemble Classifier Hyperparameters (Task 2)
N_FOLDS = 3
N_TOP_FEATURES = 120
EM_ROUNDS = 2

TAIL_THRESHOLD = 0.95
STRETCH_FACTOR = 0.15

PURIFIED_WEIGHT = 2.0
CONFUSION_WEIGHT = 0.5

CAPITAL_TYPES = ["游资", "量化", "散户"]
