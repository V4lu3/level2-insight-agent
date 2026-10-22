"""端到端运行：训练特征（110 日）+ 推理特征（9 日）。

- 训练：训练期全部日期 → 训练\训练特征\Task2_features_train.csv
- 推理：9 个推理日（7.10~7.22）→ 推理\推理特征\{7.xx}\Task1_features_infer.csv
"""
import sys
import time
import shutil
from pathlib import Path

ROOT = Path(r'd:\lwj2333\Downloads\比赛\AFAC2026金融智能创新大赛')
sys.path.insert(0, str(ROOT))

from features.main import run

# ============================================================
# 配置
# ============================================================
PARQUET_DIR = ROOT / "原始数据"
MARKET_CAP = str(ROOT / "features" / "market_cap.csv")

TRAIN_DATES = [
    # 历史训练期（110 个交易日，2026-01-05 ~ 2026-06-30）
    "20260105", "20260106", "20260107", "20260113", "20260114", "20260115",
    "20260116", "20260119", "20260120", "20260121", "20260122", "20260123",
    "20260126", "20260127", "20260128", "20260129", "20260130",
    "20260202", "20260203", "20260204", "20260205", "20260206",
    "20260209", "20260210", "20260211", "20260212", "20260213",
    "20260224", "20260226", "20260227",
    "20260302", "20260303", "20260304", "20260305", "20260306",
    "20260309", "20260310", "20260311", "20260312", "20260313",
    "20260316", "20260317", "20260318", "20260319", "20260320",
    "20260323", "20260324", "20260325", "20260326", "20260327",
    "20260330", "20260331",
    "20260401", "20260402", "20260403",
    "20260407", "20260408", "20260409", "20260410",
    "20260413", "20260414", "20260415", "20260416", "20260417",
    "20260420", "20260421", "20260422", "20260423", "20260424",
    "20260427", "20260428", "20260429", "20260430",
    "20260508",
    "20260511", "20260512", "20260513", "20260514", "20260515",
    "20260518", "20260519", "20260520", "20260521", "20260522",
    "20260525", "20260526", "20260527", "20260528", "20260529",
    "20260601", "20260602", "20260603", "20260604", "20260605",
    "20260608", "20260609", "20260610", "20260611", "20260612",
    "20260615", "20260616", "20260617", "20260618",
    "20260622", "20260623", "20260624", "20260625", "20260626",
    "20260629", "20260630",
]

# 推理 9 天
INFER_DATES = [
    ("20260710", "7.10"),
    ("20260713", "7.13"),
    ("20260714", "7.14"),
    ("20260715", "7.15"),
    ("20260716", "7.16"),
    ("20260717", "7.17"),
    ("20260720", "7.20"),
    ("20260721", "7.21"),
    ("20260722", "7.22"),
]


def main():
    # ============================================================
    # 1. 训练特征（多日合并，单次 run）
    # ============================================================
    print("=" * 80)
    print(f"Step 1: 训练特征生成  ({len(TRAIN_DATES)} 个交易日)")
    print("=" * 80)
    TRAIN_OUT = ROOT / "训练" / "训练特征"
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    TRAIN_TMP = ROOT / "训练" / "训练特征" / "_tmp_full"
    TRAIN_TMP.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df_train = run(
        parquet_daily_dir=str(PARQUET_DIR),
        output_dir=str(TRAIN_TMP),
        dates=TRAIN_DATES,
        mode="train",
        n_jobs=6,         # 6 天并行（分批读取后每 worker ~0.6GB，6×0.6=3.6GB 安全）
        do_neutralize=True,
        market_cap_csv=MARKET_CAP,
        max_stocks=0,
    )
    print(f"\n训练特征计算完成, shape = {df_train.shape}, 耗时 {time.time()-t0:.1f}s")

    # 输出到 Task2_features_train.csv
    train_csv = TRAIN_OUT / "Task2_features_train.csv"
    df_train.to_csv(train_csv, index=False, encoding="utf-8-sig")
    print(f"已写出: {train_csv}")

    # 清理 tmp
    shutil.rmtree(TRAIN_TMP, ignore_errors=True)

    # ============================================================
    # 2. 推理特征（每天一个文件）
    # ============================================================
    print()
    print("=" * 80)
    print(f"Step 2: 推理特征生成  ({len(INFER_DATES)} 天)")
    print("=" * 80)
    INFER_ROOT = ROOT / "推理" / "推理特征"
    INFER_TMP = ROOT / "推理" / "_tmp_infer"
    INFER_TMP.mkdir(parents=True, exist_ok=True)

    for d, label in INFER_DATES:
        out_dir = INFER_TMP / label
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        df = run(
            parquet_daily_dir=str(PARQUET_DIR),
            output_dir=str(out_dir),
            dates=[d],
            mode="train",      # 单日也用 train 模式（没有历史 meta）
            n_jobs=1,
            do_neutralize=True,
            market_cap_csv=None,  # 自动扫描 特征工程/*market_cap.csv（含 7.xx 分日期文件）
            max_stocks=0,
        )
        # 输出到目标目录
        target_dir = INFER_ROOT / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target_csv = target_dir / "Task1_features_infer.csv"
        df.to_csv(target_csv, index=False, encoding="utf-8-sig")
        print(f"  [{label}] shape={df.shape}, 耗时 {time.time()-t0:.1f}s, → {target_csv}")

    # 清理 tmp
    shutil.rmtree(INFER_TMP, ignore_errors=True)

    print()
    print("=" * 80)
    print("[DONE] 训练特征 + 推理特征 全部生成完毕")
    print("=" * 80)


if __name__ == '__main__':
    main()
