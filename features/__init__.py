"""特征工程包入口。包内所有 API 通过 main 暴露（向后兼容）。"""
from .main import (
    # 版本
    __version__,
    # 核心函数
    run, main, postprocess, save_meta, load_meta,
    # 特征清单
    P0_FEATS, P1_FEATS,
    NEUTRALIZE_COLS, FINAL_FEAT_LIST, OUTPUT_COLUMNS,
    ENH_FEATS, L3_WINDOWS,
    RAW_QUOTE_COLS, RAW_TRADE_COLS, RAW_ORDER_COLS, NEU_FEAT_LIST,
)

__all__ = [
    "run", "main", "postprocess", "save_meta", "load_meta",
    "P0_FEATS", "P1_FEATS",
    "NEUTRALIZE_COLS", "FINAL_FEAT_LIST", "OUTPUT_COLUMNS",
    "ENH_FEATS", "L3_WINDOWS",
    "RAW_QUOTE_COLS", "RAW_TRADE_COLS", "RAW_ORDER_COLS", "NEU_FEAT_LIST",
]
