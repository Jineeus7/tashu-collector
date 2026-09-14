"""'그대로 유지'가 이기는 건 2시간 안에 아무 일도 안 일어나기 때문이다.
그렇다면 뭔가 일어나는 곳에서 재야 한다. 두 방향으로 넓혀 본다.

  1) 더 먼 시점 — 1, 2, 6, 12, 24시간 뒤
  2) 변화가 잦은 대여소 — 10분 변화량 상위 20%

시간대 프로필을 피처에 넣는다. 먼 시점일수록 "지금 몇 대"보다 "이 대여소가
그 시각에 보통 몇 대"가 중요해질 것이기 때문이다.

실행: python3 experiments3.py <스냅샷 폴더>
"""

import os
import sys
from datetime import timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import (PARAMS, build_grid, hour_profile,  # noqa: E402
                         load_snapshots, near_matrix)

LAGS = [1, 3, 6, 144]
HORIZONS = [6, 12, 36, 72, 144]          # 1, 2, 6, 12, 24시간 (1칸=10분)
MAXLAG = max(LAGS)
DEADBAND = 0.8


def rows(grid, kst, prof_mean, prof_std, prof_hour, lo, hi, cap, rng):
    n = grid.shape[0]
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
            if si.size > cap:
                si = rng.choice(si, cap, replace=False)
            hour = kst[t].hour + kst[t].minute / 60
            dow = kst[t].weekday()
            # 도착 시각의 시간대 프로필. 먼 시점일수록 이쪽이 답에 가깝다.
            arr = kst[t + h]
            pa = prof_hour[arr.hour][si]
            cols = [cur[si]]
            cols += [cur[si] - grid[t - lg][si] for lg in LAGS]
            cols += [
                np.full(si.size, np.sin(2 * np.pi * hour / 24)),
                np.full(si.size, np.cos(2 * np.pi * hour / 24)),
                np.full(si.size, float(dow)),
                np.full(si.size, float(dow >= 5)),
                np.full(si.size, h * 10.0),
                prof_mean[si], prof_std[si],
                np.full(si.size, np.sin(2 * np.pi * arr.hour / 24)),
                np.full(si.size, np.cos(2 * np.pi * arr.hour / 24)),
                pa, cur[si] - pa,
            ]
            X.append(np.column_stack(cols))
            y.append(nxt[si])
            base.append(cur[si])
            hors.append(np.full(si.size, h))
    return (np.vstack(X), np.concatenate(y),
            np.concatenate(base), np.concatenate(hors))


def hit(v, y):
    return ((v >= 1) == (y >= 1)).mean()


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    snaps = load_snapshots(data_dir)
    grid, sids, t0, coord = build_grid(snaps)
    n = grid.shape[0]
    kst = [t0 + timedelta(minutes=10 * k, hours=9) for k in range(n)]
    split = int(n * 0.8)

    prof_mean = np.nan_to_num(np.nanmean(grid[:split], axis=0))
    prof_std = np.nan_to_num(np.nanstd(grid[:split], axis=0))
    prof_hour = hour_profile(grid, kst, 0, split)

    # 변화가 잦은 대여소: 10분 변화량 절대값 평균 상위 20%
    d10 = np.abs(np.diff(grid[:split], axis=0))
    churn = np.nan_to_num(np.nanmean(d10, axis=0))
    busy = churn >= np.quantile(churn, 0.8)
    print(f"스냅샷 {len(snaps)}개 / 대여소 {len(sids)}개소 "
          f"(변화 잦은 곳 {busy.sum()}개소)")
    print(f"학습 {kst[0]:%m-%d} ~ {kst[split-1]:%m-%d} / "
          f"검증 {kst[split]:%m-%d} ~ {kst[-1]:%m-%d}\n")

    Xtr, ytr, btr, _ = rows(grid, kst, prof_mean, prof_std, prof_hour,
                            0, split, 120, np.random.default_rng(42))
    Xva, yva, bva, hva = rows(grid, kst, prof_mean, prof_std, prof_hour,
                              split, n, 400, np.random.default_rng(7))

    # 검증행이 어느 대여소인지 다시 만든다 (busy 구분용)
    sel = []
    r = np.random.default_rng(7)
    for h in HORIZONS:
        for t in range(max(split, MAXLAG), min(n, n - h)):
            cur, nxt = grid[t], grid[t + h]
            ok = ~np.isnan(cur) & ~np.isnan(nxt)
            for lg in LAGS:
                ok &= ~np.isnan(grid[t - lg])
            si = np.where(ok)[0]
            if si.size == 0:
                continue
            if si.size > 400:
                si = r.choice(si, 400, replace=False)
            sel.append(si)
    sel = np.concatenate(sel)
    assert sel.size == yva.size

    from xgboost import XGBRegressor
    m = XGBRegressor(**PARAMS).fit(Xtr, ytr - btr)
    dv = m.predict(Xva)
    dv = np.where(np.abs(dv) < DEADBAND, 0.0, dv)
    pred = np.round(np.clip(bva + dv, 0, None))

    isbusy = busy[sel]
    print(f"{'':8s}{'전체':^17}{'변화 잦은 곳':^19}")
    print(f"{'':8s}{'유지':>8}{'모델':>8}{'유지':>10}{'모델':>8}{'변화율':>9}")
    for h in HORIZONS:
        s = hva == h
        sb = s & isbusy
        chg = (np.abs(yva[s] - bva[s]) >= 1).mean()
        label = f"{h//6}시간"
        print(f"{label:8s}{hit(bva[s], yva[s]):8.1%}{hit(pred[s], yva[s]):8.1%}"
              f"{hit(bva[sb], yva[sb]):10.1%}{hit(pred[sb], yva[sb]):8.1%}"
              f"{chg:9.0%}")
    print("\n변화율 = 그 시점에 재고가 실제로 달라진 건의 비율")


if __name__ == "__main__":
    main()
