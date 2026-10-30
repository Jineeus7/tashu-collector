"""큰 변화(1시간 3대+) 감지율을 비교 기준과 함께. 9일간(9/23~10/2).
- 지금 값 그대로 / 평소 흐름 규칙(지금 재고 + 대여 이력 평소 순증) / 재고만 학습 / 우리 모델
- 감지율(재현율): 실제 큰 변화 중 방향 맞게 1대+ 예측
- 정밀도: '1시간에 3대+ 바뀐다'고 예측한 것 중 실제로 같은 방향 3대+ 바뀐 비율

실행 (저장소 루트에서): python3 model/experiments/exp13_big_change.py
결과: results/exp13_big_change.txt"""
import os
import sys
from datetime import timedelta
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import flow, train as T
from exp12_final_eval import EVAL, SETS, BASE_PARAMS, RAW, build, matrix, to_count
from xgboost import XGBRegressor

NEW = {**BASE_PARAMS, "max_depth": 10, "n_estimators": 800, "learning_rate": 0.04, "min_child_weight": 100}
snaps = T.load_snapshots("data"); grid, sids, t0 = T.build_grid(snaps); n = grid.shape[0]
kst = [t0 + timedelta(minutes=10 * k, hours=9) for k in range(n)]
idx_of = lambda dt: next((k for k in range(n) if kst[k] >= dt), n)
out, inn = flow.load("model/flow_profile.npz", sids)
A = build(grid, kst, 0, n, out, inn, 120, np.random.default_rng(42))
B = build(grid, kst, idx_of(EVAL[0][0]), n, out, inn, 400, np.random.default_rng(7))
R = {k: [] for k in ("stock", "new", "y", "b", "h", "flow")}
for lo_dt, hi_dt, lab in EVAL:
    lo, hi = idx_of(lo_dt), idx_of(hi_dt)
    pm = np.nan_to_num(np.nanmean(grid[:lo], 0)); ps = np.nan_to_num(np.nanstd(grid[:lo], 0))
    tr = (A["t"] + A["hz"] // 10) < lo; te = (B["t"] >= lo) & (B["t"] < hi)
    for key, names, prm, db in (("stock", RAW[:10], BASE_PARAMS, 0.8), ("new", SETS["+전부"], NEW, 0.7)):
        m = XGBRegressor(**prm).fit(matrix(A, tr, names, pm, ps), (A["nxt"] - A["cur"])[tr])
        R[key].append(to_count(m.predict(matrix(B, te, names, pm, ps)), B["cur"][te], db))
    R["flow"].append(np.round(np.clip(B["cur"][te] + np.nan_to_num(B["flow2"][te]), 0, None)))
    for k, s in (("y", "nxt"), ("b", "cur"), ("h", "hz")): R[k].append(B[s][te])
    print(lab, "완료", flush=True)
P = {k: np.concatenate(v) for k, v in R.items()}
y, b, h = P["y"], P["b"], P["h"]
s = h == 60; big = s & (np.abs(y - b) >= 3)
print(f"\n1시간 뒤, 전체 {s.sum():,}건 중 실제 큰 변화(3대+) {big.sum():,}건 ({big.sum()/s.sum():.1%})")
print(f"{'방법':22}{'감지율':>8}{'정밀도':>8}{'큰 변화 예측 건수':>16}")
for name, p in (("지금 값 그대로", b), ("평소 흐름 규칙", P["flow"]), ("재고만 학습", P["stock"]), ("우리 모델", P["new"])):
    rec = ((np.sign(p[big]-b[big]) == np.sign(y[big]-b[big])) & (np.abs(p[big]-b[big]) >= 1)).mean()
    pb = s & (np.abs(p - b) >= 3)
    prec = (np.sign(p[pb]-b[pb]) == np.sign(y[pb]-b[pb])) & (np.abs(y[pb]-b[pb]) >= 3)
    print(f"{name:22}{rec:8.0%}{(prec.mean() if pb.any() else float('nan')):8.0%}{pb.sum():16,}")
