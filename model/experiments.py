"""성능을 올릴 만한 후보들을 같은 잣대로 재본다.

검증 구간을 한 번만 떼어 놓고, 피처만 바꿔 가며 같은 행 위에서 비교한다.
그래야 "좋아졌다"가 피처 덕인지 우연인지 갈린다.

후보:
  base   지금 쓰는 것
  +hour  대여소×시간대 평균 (지금은 대여소 전체 평균만 쓴다)
  +near  가까운 대여소 5곳의 재고 합
  +lag   2시간/3시간 전 차이와 1시간 이동평균
  all    위 셋 전부
  clf    all 을 분류로 (탈 수 있나/없나를 직접 맞히게)

실행: python3 experiments.py <스냅샷 폴더>
"""

import glob
import gzip
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

HORIZONS = [1, 3, 6, 12]                # 10, 30, 60, 120분 (1칸=10분)
LAGS = [1, 3, 6, 144]
EXTRA_LAGS = [12, 18]                   # 2시간, 3시간 전
MAXLAG = max(LAGS + EXTRA_LAGS)
K_NEAR = 5

PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.06,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
              tree_method="hist", n_jobs=-1, random_state=42)


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
        rows = d["data"]["results"]
        snaps.append((t, rows))
    snaps.sort()
    return snaps


def build_grid(snaps):
    sids = sorted({r["id"] for _, rows in snaps for r in rows})
    idx = {s: i for i, s in enumerate(sids)}
    t0 = snaps[0][0]
    n = int(round((snaps[-1][0] - t0).total_seconds() / 600)) + 1
    grid = np.full((n, len(sids)), np.nan, dtype=np.float32)
    coord = np.full((len(sids), 2), np.nan, dtype=np.float64)
    for t, rows in snaps:
        k = int(round((t - t0).total_seconds() / 600))
        if not (0 <= k < n):
            continue
        for r in rows:
            i = idx[r["id"]]
            grid[k, i] = r["parking_count"]
            if np.isnan(coord[i, 0]):
                try:
                    coord[i] = (float(r["x_pos"]), float(r["y_pos"]))
                except (TypeError, ValueError):
                    pass
    return grid, sids, t0, coord


def near_matrix(coord):
    """가까운 K곳의 인덱스. 좌표가 없는 곳은 자기 자신으로 채운다."""
    m = len(coord)
    ok = ~np.isnan(coord[:, 0])
    la = np.radians(np.nan_to_num(coord[:, 0]))
    lo = np.radians(np.nan_to_num(coord[:, 1]))
    x = lo * np.cos(la.mean())
    pts = np.column_stack([la, x])
    out = np.zeros((m, K_NEAR), dtype=np.int32)
    for i in range(m):
        if not ok[i]:
            out[i] = i
            continue
        d = ((pts - pts[i]) ** 2).sum(1)
        d[~ok] = np.inf
        d[i] = np.inf
        out[i] = np.argsort(d)[:K_NEAR]
    return out


def hour_profile(grid, kst, lo, hi):
    """대여소 × 시각(정시 24칸) 평균. 학습 구간만 보고 만든다."""
    prof = np.zeros((24, grid.shape[1]), dtype=np.float32)
    for h in range(24):
        ks = [k for k in range(lo, hi) if kst[k].hour == h]
        if ks:
            prof[h] = np.nan_to_num(np.nanmean(grid[ks], axis=0))
    return prof


def make_rows(grid, kst, lo, hi, feats, cap, rng):
    """(대여소, 시각, 예측시점차) 조합을 피처 행렬로 편다."""
    n = grid.shape[0]
    X, y, base, hors = [], [], [], []
    for h in HORIZONS:
        for t in range(max(lo, MAXLAG), min(hi, n - h)):
            cur, nxt = grid[t], grid[t + h]
            ok = ~np.isnan(cur) & ~np.isnan(nxt)
            for lg in LAGS + EXTRA_LAGS:
                ok &= ~np.isnan(grid[t - lg])
            si = np.where(ok)[0]
            if si.size == 0:
                continue
            if si.size > cap:
                si = rng.choice(si, cap, replace=False)
            X.append(np.column_stack(feats(grid, kst, t, h, si)))
            y.append(nxt[si])
            base.append(cur[si])
            hors.append(np.full(si.size, h * 10))
    return (np.vstack(X), np.concatenate(y),
            np.concatenate(base), np.concatenate(hors))


def make_featfn(prof_mean, prof_std, prof_hour, near, use_hour, use_near, use_lag):
    def feats(grid, kst, t, h, si):
        cur = grid[t]
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
        if use_hour:
            p = prof_hour[kst[t].hour][si]
            cols += [p, cur[si] - p]          # 평소보다 몇 대 많은가
        if use_near:
            nb = near[si]                      # (행, K)
            s_now = np.nan_to_num(grid[t][nb]).sum(1)
            s_old = np.nan_to_num(grid[t - 6][nb]).sum(1)
            cols += [s_now, s_now - s_old]
        if use_lag:
            cols += [cur[si] - grid[t - lg][si] for lg in EXTRA_LAGS]
            win = np.nan_to_num(grid[t - 6:t + 1][:, si]).mean(0)
            cols += [cur[si] - win]
        return cols
    return feats


def report(name, bva, pred, yva, hva):
    line = [f"{name:7s}"]
    for h in (10, 30, 60, 120):
        s = hva == h
        line.append(f"{((pred[s] >= 1) == (yva[s] >= 1)).mean():7.1%}")
    chg = (hva == 60) & (np.abs(yva - bva) >= 3)
    det = ((np.sign(pred[chg] - bva[chg]) == np.sign(yva[chg] - bva[chg]))
           & (np.abs(pred[chg] - bva[chg]) >= 1)).mean()
    mae = np.abs(pred - yva).mean()
    line.append(f"{det:9.0%}{mae:8.3f}")
    print("".join(line))


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    rng = np.random.default_rng(42)

    snaps = load_snapshots(data_dir)
    grid, sids, t0, coord = build_grid(snaps)
    n = grid.shape[0]
    kst = [t0 + timedelta(minutes=10 * k, hours=9) for k in range(n)]
    print(f"스냅샷 {len(snaps)}개 / 대여소 {len(sids)}개소 / "
          f"{kst[0]:%m-%d %H시} ~ {kst[-1]:%m-%d %H시} (KST)")

    split = int(n * 0.8)
    prof_mean = np.nan_to_num(np.nanmean(grid[:split], axis=0))
    prof_std = np.nan_to_num(np.nanstd(grid[:split], axis=0))
    prof_hour = hour_profile(grid, kst, 0, split)
    near = near_matrix(coord)
    print(f"학습 {kst[0]:%m-%d} ~ {kst[split-1]:%m-%d} / "
          f"검증 {kst[split]:%m-%d} ~ {kst[-1]:%m-%d}\n")

    from xgboost import XGBRegressor, XGBClassifier

    print(f"{'':7s}{'10분':>7}{'30분':>7}{'1시간':>7}{'2시간':>7}"
          f"{'큰변화':>10}{'MAE':>8}")

    configs = [
        ("base",  dict(use_hour=False, use_near=False, use_lag=False)),
        ("+hour", dict(use_hour=True,  use_near=False, use_lag=False)),
        ("+near", dict(use_hour=False, use_near=True,  use_lag=False)),
        ("+lag",  dict(use_hour=False, use_near=False, use_lag=True)),
        ("all",   dict(use_hour=True,  use_near=True,  use_lag=True)),
    ]

    shown_base = False
    for name, kw in configs:
        fn = make_featfn(prof_mean, prof_std, prof_hour, near, **kw)
        r = np.random.default_rng(42)
        Xtr, ytr, btr, _ = make_rows(grid, kst, 0, split, fn, 120, r)
        r = np.random.default_rng(7)
        Xva, yva, bva, hva = make_rows(grid, kst, split, n, fn, 400, r)
        if not shown_base:
            report("유지", bva, bva, yva, hva)      # 그대로 유지 기준선
            shown_base = True
        m = XGBRegressor(**PARAMS).fit(Xtr, ytr - btr)
        pred = np.round(np.clip(bva + m.predict(Xva), 0, None))
        report(name, bva, pred, yva, hva)

        if name == "all":
            # 같은 피처로 "탈 수 있나"를 직접 맞히게 해 본다
            c = XGBClassifier(**PARAMS, eval_metric="logloss")
            c.fit(Xtr, (ytr >= 1).astype(int))
            p = c.predict_proba(Xva)[:, 1] >= 0.5
            line = ["clf    "]
            for h in (10, 30, 60, 120):
                s = hva == h
                line.append(f"{(p[s] == (yva[s] >= 1)).mean():7.1%}")
            print("".join(line) + f"{'-':>10}{'-':>8}")

    print("\n큰변화 = 1시간 뒤 3대 이상 움직인 건을 방향까지 맞힌 비율")


if __name__ == "__main__":
    main()
