"""타슈 재고 예측 모델 학습.

수집된 스냅샷(10분 간격)으로 "N분 뒤 재고"를 예측하는 모델을 만든다.

설계 근거 (실측으로 정한 것):
  - 재고 자체가 아니라 '변화량'을 예측한다. 재고를 직접 맞히게 하면 아무 일도
    없는 대부분의 순간에 잔오차가 쌓여 '그대로 유지'보다 나빠졌다.
  - 예측값은 반올림한다. 재고는 정수라 소수점을 남기면 오차만 늘어난다.
  - 0 아래로만 자른다. name_cn 의 숫자는 상한이 아니다 — 관측값의 8%가 그
    숫자를 넘었고, 상한으로 쓰면 만석 근처 오차가 4배로 뛴다.
  - 예측 시점차(10~120분)를 피처로 넣어 모델 하나로 전 구간을 커버한다.
  - 변화량이 0.8대에 못 미치면 움직이지 않는다. 확신 없이 흔들면 '그대로
    유지'보다 적중률이 떨어졌고, 둔감폭을 두자 적중률은 그 수준으로
    돌아오면서 큰 변화 감지는 3분의 1이 남았다.

실행: python3 train.py <스냅샷 폴더>
"""

import glob
import gzip
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

HORIZONS = [1, 2, 3, 6, 9, 12]          # 10,20,30,60,90,120분 (1칸=10분)
DEADBAND = 0.8                          # 이만큼 안 움직인다고 보면 그대로 둔다
LAGS = [1, 3, 6, 144]                   # 10분, 30분, 1시간, 24시간 전
MAXLAG = max(LAGS)
OUT = os.path.dirname(os.path.abspath(__file__))


def load_snapshots(data_dir):
    snaps = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*", "*.json.gz"))):
        try:
            d = json.load(gzip.open(f, "rt"))
        except Exception:
            continue
        ts = d.get("collected_at") or d.get("fetched_at")
        if not ts:
            continue
        t = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
        snaps.append((t, {r["id"]: r["parking_count"] for r in d["data"]["results"]}))
    snaps.sort()
    return snaps


def build_grid(snaps):
    """10분 격자에 스냅. 수집이 밀려도 같은 칸으로 모인다."""
    sids = sorted(set().union(*[set(s) for _, s in snaps]))
    idx = {s: i for i, s in enumerate(sids)}
    t0 = snaps[0][0]
    n = int(round((snaps[-1][0] - t0).total_seconds() / 600)) + 1
    grid = np.full((n, len(sids)), np.nan, dtype=np.float32)
    for t, stock in snaps:
        k = int(round((t - t0).total_seconds() / 600))
        if 0 <= k < n:
            for sid, v in stock.items():
                grid[k, idx[sid]] = v
    return grid, sids, t0


def make_rows(grid, t0, lo, hi, prof_mean, prof_std, cap_rows, rng):
    """(대여소, 시각, 예측시점차) 조합을 피처 행렬로 편다."""
    n = grid.shape[0]
    kst = [t0 + timedelta(minutes=10 * k, hours=9) for k in range(n)]
    X, y, base, hors = [], [], [], []
    for h in HORIZONS:
        for t in range(max(lo, MAXLAG), min(hi, n - h)):
            cur, nxt = grid[t], grid[t + h]
            ok = ~np.isnan(cur) & ~np.isnan(nxt)
            for lg in LAGS:
                ok &= ~np.isnan(grid[t - lg])
            si = np.where(ok)[0]
            if si.size == 0:
                continue
            if si.size > cap_rows:
                si = rng.choice(si, cap_rows, replace=False)
            hour = kst[t].hour + kst[t].minute / 60
            dow = kst[t].weekday()
            cols = [cur[si]]
            cols += [cur[si] - grid[t - lg][si] for lg in LAGS]
            cols += [
                np.full(si.size, np.sin(2 * np.pi * hour / 24)),
                np.full(si.size, np.cos(2 * np.pi * hour / 24)),
                np.full(si.size, float(dow)),
                np.full(si.size, float(dow >= 5)),
                np.full(si.size, h * 10.0),
                prof_mean[si],
                prof_std[si],
            ]
            X.append(np.column_stack(cols))
            y.append(nxt[si])
            base.append(cur[si])
            hors.append(np.full(si.size, h * 10))
    return (np.vstack(X), np.concatenate(y),
            np.concatenate(base), np.concatenate(hors))


FEATURES = (["현재재고"] + [f"{m}전차이" for m in ("10분", "30분", "1시간", "24시간")]
            + ["시각sin", "시각cos", "요일", "주말", "예측시점차", "평소평균", "평소변동"])


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    rng = np.random.default_rng(42)

    snaps = load_snapshots(data_dir)
    grid, sids, t0 = build_grid(snaps)
    n = grid.shape[0]
    print(f"스냅샷 {len(snaps)}개 / 대여소 {len(sids)}개소")
    print(f"기간(KST) {t0 + timedelta(hours=9):%m-%d %H시} ~ "
          f"{t0 + timedelta(minutes=10*(n-1), hours=9):%m-%d %H시}")

    prof_mean = np.nan_to_num(np.nanmean(grid, axis=0))
    prof_std = np.nan_to_num(np.nanstd(grid, axis=0))

    from xgboost import XGBRegressor
    params = dict(n_estimators=400, max_depth=6, learning_rate=0.06,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                  tree_method="hist", n_jobs=-1, random_state=42)

    # 1) 뒤쪽을 떼어 정확도를 잰다
    split = int(n * 0.8)
    Xtr, ytr, btr, _ = make_rows(grid, t0, 0, split, prof_mean, prof_std, 120, rng)
    Xva, yva, bva, hva = make_rows(grid, t0, split, n, prof_mean, prof_std, 400, rng)
    val = XGBRegressor(**params).fit(Xtr, ytr - btr)
    dv = val.predict(Xva)
    dv = np.where(np.abs(dv) < DEADBAND, 0.0, dv)
    pred = np.round(np.clip(bva + dv, 0, None))

    print(f"\n검증 {Xva.shape[0]:,}건 — '탈 수 있나' 적중률")
    print(f"  {'시점':>6} {'그대로유지':>9} {'모델':>7}")
    for h in (10, 30, 60, 120):
        s = hva == h
        hit = lambda v: ((v[s] >= 1) == (yva[s] >= 1)).mean()
        print(f"  {h:5d}분 {hit(bva):8.0%} {hit(pred):7.0%}")

    chg = (hva == 60) & (np.abs(yva - bva) >= 3)
    det = ((np.sign(pred[chg] - bva[chg]) == np.sign(yva[chg] - bva[chg]))
           & (np.abs(pred[chg] - bva[chg]) >= 1)).mean()
    print(f"  1시간 뒤 큰 변화 감지율: {det:.0%}")

    # 2) 전체 구간으로 다시 학습해 배포용 모델을 만든다
    Xall, yall, ball, _ = make_rows(grid, t0, 0, n, prof_mean, prof_std, 120, rng)
    final = XGBRegressor(**params).fit(Xall, yall - ball)

    final.save_model(os.path.join(OUT, "model.json"))
    with open(os.path.join(OUT, "station_stats.json"), "w") as fh:
        json.dump({s: {"mean": round(float(prof_mean[i]), 3),
                       "std": round(float(prof_std[i]), 3)}
                   for i, s in enumerate(sids)}, fh)
    with open(os.path.join(OUT, "model_meta.json"), "w") as fh:
        json.dump({"features": FEATURES, "lags_slots": LAGS,
                   "horizons_min": [h * 10 for h in HORIZONS],
                   "trained_rows": int(Xall.shape[0]),
                   "trained_at": datetime.now().isoformat(timespec="seconds")},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n학습 {Xall.shape[0]:,}건 → model.json / station_stats.json 저장")


if __name__ == "__main__":
    main()
