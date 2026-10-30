"""날씨 변수를 넣으면 나아지는지 본다. 9일간(9/23~10/2) 3개 기간 채점.
변수는 서울 따릉이 대여량 회귀에서 흔히 쓰는 것과 같게 골랐다.
- 기준: 지금 배포 모델(23개 피처)
- +날씨(지금): 기온·습도·이슬점·풍속·일사량·강수·구름·지면온도 (가시거리는 과거 기록 없음, 적설은 이 시기 0)
- +날씨(지금+도착 시각): 도착 시각 날씨까지 정확히 안다고 가정 (예보가 완벽할 때의 상한)
날씨: Open-Meteo 과거 기록, 대전 시청 근처 1시간 단위. 처음 실행하면 받아서 wx_daejeon.json 으로 저장한다.

실행 (저장소 루트에서): python3 model/experiments/exp14_weather.py
결과: results/exp14_weather.txt"""
import json
import os
import sys
import urllib.request
from datetime import timedelta
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import flow, train as T
from exp12_final_eval import EVAL, SETS, BASE_PARAMS, build, matrix, to_count
from xgboost import XGBRegressor

NEW = {**BASE_PARAMS, "max_depth": 10, "n_estimators": 800, "learning_rate": 0.04, "min_child_weight": 100}
WX = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_speed_10m",
      "shortwave_radiation", "precipitation", "cloud_cover", "soil_temperature_0_to_7cm"]
WX_FILE = os.path.join(HERE, "wx_daejeon.json")
if not os.path.exists(WX_FILE):
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=36.35&longitude=127.385"
           "&start_date=2026-08-30&end_date=2026-10-03&hourly=" + ",".join(WX) +
           "&timezone=Asia%2FSeoul&wind_speed_unit=ms")
    urllib.request.urlretrieve(url, WX_FILE)
w = json.load(open(WX_FILE))["hourly"]
wx_at = {t: np.array([w[k][i] for k in WX], np.float32) for i, t in enumerate(w["time"])}

snaps = T.load_snapshots("data"); grid, sids, t0 = T.build_grid(snaps); n = grid.shape[0]
kst = [t0 + timedelta(minutes=10 * k, hours=9) for k in range(n + 12)]
idx_of = lambda dt: next((k for k in range(n) if kst[k] >= dt), n)
W = np.stack([wx_at.get(kst[k].strftime("%Y-%m-%dT%H:00"), np.full(len(WX), np.nan, np.float32)) for k in range(n + 12)])
out, inn = flow.load("model/flow_profile.npz", sids)
A = build(grid, kst, 0, n, out, inn, 120, np.random.default_rng(42))
B = build(grid, kst, idx_of(EVAL[0][0]), n, out, inn, 400, np.random.default_rng(7))
for R in (A, B):
    now, tgt = W[R["t"]], W[R["t"] + R["hz"] // 10]
    for j, k in enumerate(WX):
        R["wn_" + k], R["wt_" + k] = now[:, j], tgt[:, j]
print(f"학습 {len(A['cur']):,}행 · 채점 {len(B['cur']):,}행", flush=True)

BASE = SETS["+전부"]
VARS = {"기준(배포 모델)": BASE,
        "+날씨(지금)": BASE + ["wn_" + k for k in WX],
        "+날씨(지금+도착 시각)": BASE + ["wn_" + k for k in WX] + ["wt_" + k for k in WX]}
P = {k: [] for k in VARS}; M = {k: [] for k in ("y", "b", "h", "rain")}
for lo_dt, hi_dt, lab in EVAL:
    lo, hi = idx_of(lo_dt), idx_of(hi_dt)
    pm = np.nan_to_num(np.nanmean(grid[:lo], 0)); ps = np.nan_to_num(np.nanstd(grid[:lo], 0))
    tr = (A["t"] + A["hz"] // 10) < lo; te = (B["t"] >= lo) & (B["t"] < hi)
    for name, cols in VARS.items():
        m = XGBRegressor(**NEW).fit(matrix(A, tr, cols, pm, ps), (A["nxt"] - A["cur"])[tr])
        P[name].append(to_count(m.predict(matrix(B, te, cols, pm, ps)), B["cur"][te], 0.7))
    M["y"].append(B["nxt"][te]); M["b"].append(B["cur"][te]); M["h"].append(B["hz"][te])
    M["rain"].append((B["wn_precipitation"][te] >= 0.5) | (B["wt_precipitation"][te] >= 0.5))
    print(lab, "완료", flush=True)

P = {k: np.concatenate(v) for k, v in P.items()}; M = {k: np.concatenate(v) for k, v in M.items()}
y, b, h, rain = M["y"], M["b"], M["h"], M["rain"]
s = h == 120; mid = s & (b >= 1) & (b <= 2); sr = s & rain; big = (h == 60) & (np.abs(y - b) >= 3)
print(f"\n9일간 2시간 뒤 {s.sum():,}건 (비 온 경우 {sr.sum():,}건 = {sr.sum()/s.sum():.1%})")
print(f"{'':24}{'평균오차':>8}{'2대 적중':>9}{'1~2대 적중':>11}{'비 올 때 오차':>12}{'큰 변화 감지':>11}")
for name, p in [("지금 값 그대로", b)] + list(P.items()):
    hit = (p >= 2) == (y >= 2)
    det = ((np.sign(p[big] - b[big]) == np.sign(y[big] - b[big])) & (np.abs(p[big] - b[big]) >= 1)).mean()
    print(f"{name:24}{np.abs(p-y)[s].mean():8.3f}{hit[s].mean():9.1%}{hit[mid].mean():11.1%}"
          f"{np.abs(p-y)[sr].mean():12.3f}{det:11.0%}")
