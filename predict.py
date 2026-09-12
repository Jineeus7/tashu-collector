"""최근 스냅샷으로 10~120분 뒤 재고를 예측해 site/ 에 JSON으로 내놓는다.

웹앱은 이 두 파일만 받아 간다.

    site/stations.json     대여소 이름·좌표. 거의 안 바뀐다.
    site/predictions.json  현재 재고와 12개 시점 예측. 10분마다 갱신.

예측을 미리 계산해두는 이유는 슬라이더 때문이다. 사용자가 슬라이더를 움직일
때마다 서버를 부르면 버벅이므로, 12개 시점을 한 번에 내려보내 브라우저가
즉시 바꿔 그리게 한다. 12시점 × 1,372개소를 다 담아도 40KB 안쪽이다.

예측값은 반올림하고 0 아래로만 자른다. name_cn 의 숫자는 상한이 아니다 —
관측값의 8%가 그 숫자를 넘는다.

변화량이 둔감폭에 못 미치면 지금 값을 그대로 둔다. 모델이 아무 일도
없는 순간까지 조금씩 흔들어서 '그대로 유지'보다 적중률이 낮았다.
"""

import gzip
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("TASHU_DATA", os.path.join(HERE, "data"))
SITE = os.path.join(HERE, "site")
MODEL = os.path.join(HERE, "model")

HORIZONS = [10, 20, 30, 60, 90, 120]     # 분
LAGS = [1, 3, 6, 144]                    # 10분, 30분, 1시간, 24시간 전 (1칸=10분)
NEED_SLOTS = max(LAGS) + 2
DEADBAND = 0.8                           # 이만큼 안 움직인다고 보면 그대로 둔다
KST = timezone(timedelta(hours=9))


def recent_files(count):
    """최근 스냅샷 파일을 시간순으로. 하루치가 144개라 넉넉히 훑는다."""
    days = sorted(os.listdir(DATA))[-3:]
    files = []
    for d in days:
        p = os.path.join(DATA, d)
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p))
                      if f.endswith(".json.gz")]
    return files[-count:]


def load_recent():
    snaps = []
    for f in recent_files(NEED_SLOTS * 2):
        try:
            d = json.load(gzip.open(f, "rt"))
        except Exception:
            continue
        ts = d.get("collected_at") or d.get("fetched_at")
        if not ts:
            continue
        t = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        snaps.append((t.replace(tzinfo=None), d["data"]["results"]))
    snaps.sort()
    return snaps


def main():
    snaps = load_recent()
    if not snaps:
        raise SystemExit("스냅샷이 없다")

    latest_t, latest_rows = snaps[-1]
    sids = [r["id"] for r in latest_rows]
    idx = {s: i for i, s in enumerate(sids)}

    # 최신 시각 기준 10분 격자에 과거를 얹는다. 수집이 밀려도 같은 칸으로 모인다.
    n = NEED_SLOTS
    grid = np.full((n, len(sids)), np.nan, dtype=np.float32)
    for t, rows in snaps:
        k = n - 1 - int(round((latest_t - t).total_seconds() / 600))
        if 0 <= k < n:
            for r in rows:
                if r["id"] in idx:
                    grid[k, idx[r["id"]]] = r["parking_count"]

    cur = grid[-1]
    stats = json.load(open(os.path.join(MODEL, "station_stats.json")))
    pmean = np.array([stats.get(s, {}).get("mean", 0.0) for s in sids], dtype=np.float32)
    pstd = np.array([stats.get(s, {}).get("std", 0.0) for s in sids], dtype=np.float32)

    kst_now = latest_t.replace(tzinfo=timezone.utc).astimezone(KST)
    hour = kst_now.hour + kst_now.minute / 60
    dow = kst_now.weekday()

    # 과거 값이 비면 변화량을 0으로 둔다. 결측 자체를 모델에 넘기는 것보다
    # "변화 없었다"로 보는 편이 예측이 튀지 않는다.
    diffs = []
    for lg in LAGS:
        past = grid[-1 - lg] if n > lg else np.full_like(cur, np.nan)
        d = cur - past
        diffs.append(np.where(np.isnan(d), 0.0, d))

    booster = xgb.Booster()
    booster.load_model(os.path.join(MODEL, "model.json"))

    preds = {}
    for h in HORIZONS:
        cols = [cur] + diffs + [
            np.full(len(sids), np.sin(2 * np.pi * hour / 24)),
            np.full(len(sids), np.cos(2 * np.pi * hour / 24)),
            np.full(len(sids), float(dow)),
            np.full(len(sids), float(dow >= 5)),
            np.full(len(sids), float(h)),
            pmean, pstd,
        ]
        X = np.column_stack(cols).astype(np.float32)
        delta = booster.inplace_predict(X)
        delta = np.where(np.abs(delta) < DEADBAND, 0.0, delta)
        preds[h] = np.maximum(np.round(cur + delta), 0)

    os.makedirs(SITE, exist_ok=True)

    # 대여소 정보는 좌표가 문자열로 오고 x/y 순서가 위경도와 반대다.
    stations = {}
    for r in latest_rows:
        try:
            lat, lon = float(r["x_pos"]), float(r["y_pos"])
        except (TypeError, ValueError):
            continue
        stations[r["id"]] = {"n": r.get("name"), "la": round(lat, 6), "lo": round(lon, 6)}
    with open(os.path.join(SITE, "stations.json"), "w") as fh:
        json.dump(stations, fh, ensure_ascii=False, separators=(",", ":"))

    out = {
        "at": latest_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizons": HORIZONS,
        "stock": {s: int(cur[i]) for i, s in enumerate(sids) if not np.isnan(cur[i])},
        "pred": {s: [int(preds[h][i]) for h in HORIZONS]
                 for i, s in enumerate(sids) if not np.isnan(cur[i])},
    }
    with open(os.path.join(SITE, "predictions.json"), "w") as fh:
        json.dump(out, fh, separators=(",", ":"))

    size = os.path.getsize(os.path.join(SITE, "predictions.json")) / 1024
    print(f"{kst_now:%m-%d %H:%M} KST — {len(out['stock'])}개소 예측 ({size:.0f}KB)")


if __name__ == "__main__":
    main()
