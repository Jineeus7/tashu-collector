"""공휴일을 일요일처럼 다루면 연휴 오차가 줄어드는지 본다.

모델에는 공휴일 개념이 없어서 추석(9/24~26)을 평소 목·금·토로 보고 출퇴근
흐름을 예측했다. 추석 당일 2시간 뒤 평균 오차가 그대로 유지보다 35% 컸다.

실험: 모델은 그대로 두고, 예측할 때만 공휴일이면 요일·주말 피처와 대여 이력
흐름 프로필을 '일요일' 것으로 바꿔 넣는다. 학습은 추석 전 데이터로만 해서
연휴를 한 번도 본 적 없는 모델로 잰다 — 다가오는 개천절·한글날과 같은 조건.

실행: python3 experiments11.py ../data flow_profile.npz
"""

import os
import sys
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402
import train as T  # noqa: E402

HOLIDAYS = {date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 26)}
SUNDAY = 6
PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.06, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=50, tree_method="hist",
              n_jobs=-1, random_state=42)


def dow(d, as_sunday):
    return SUNDAY if as_sunday and d.date() in HOLIDAYS else d.weekday()


def flow_cols(now, h, si, out, inn, as_sunday):
    """flow.cols 와 같되, 공휴일 칸은 일요일 프로필을 쓴다."""
    o = np.zeros(len(si), np.float32)
    i_ = np.zeros(len(si), np.float32)
    for k in range(1, h + 1):
        d = now + timedelta(minutes=10 * k)
        w = dow(d, as_sunday)
        o += out[w, d.hour][si] / 6
        i_ += inn[w, d.hour][si] / 6
    w0 = dow(now, as_sunday)
    return [o, i_, i_ - o, out[w0, now.hour][si], inn[w0, now.hour][si]]


def rows(grid, kst, lo, hi, pm, ps, out, inn, cap, rng, as_sunday):
    n = grid.shape[0]
    X, y, base, hors, tt = [], [], [], [], []
    for h in T.HORIZONS:
        for t in range(max(lo, T.MAXLAG), min(hi, n - h)):
            cur, nxt = grid[t], grid[t + h]
            ok = ~np.isnan(cur) & ~np.isnan(nxt)
            for lg in T.LAGS:
                ok &= ~np.isnan(grid[t - lg])
            si = np.where(ok)[0]
            if si.size == 0:
                continue
            if si.size > cap:
                si = rng.choice(si, cap, replace=False)
            hour = kst[t].hour + kst[t].minute / 60
            dw = dow(kst[t], as_sunday)
            cols = [cur[si]] + [cur[si] - grid[t - lg][si] for lg in T.LAGS] + [
                np.full(si.size, np.sin(2 * np.pi * hour / 24)),
                np.full(si.size, np.cos(2 * np.pi * hour / 24)),
                np.full(si.size, float(dw)), np.full(si.size, float(dw >= 5)),
                np.full(si.size, h * 10.0), pm[si], ps[si]]
            cols += flow_cols(kst[t], h, si, out, inn, as_sunday)
            X.append(np.column_stack(cols))
            y.append(nxt[si]); base.append(cur[si])
            hors.append(np.full(si.size, h * 10)); tt.append(np.full(si.size, t))
    c = np.concatenate
    return np.vstack(X), c(y), c(base), c(hors), c(tt)


def main():
    data_dir, prof = sys.argv[1], sys.argv[2]
    snaps = T.load_snapshots(data_dir)
    grid, sids, t0 = T.build_grid(snaps)
    n = grid.shape[0]
    kst = [t0 + timedelta(minutes=10 * k, hours=9) for k in range(n)]
    split = next(k for k in range(n) if kst[k].date() >= date(2026, 9, 22))
    out, inn = flow.load(prof, sids)
    pm = np.nan_to_num(np.nanmean(grid[:split], 0))
    ps = np.nan_to_num(np.nanstd(grid[:split], 0))

    from xgboost import XGBRegressor
    Xtr, ytr, btr, _, _ = rows(grid, kst, 0, split, pm, ps, out, inn, 120,
                               np.random.default_rng(42), False)
    m = XGBRegressor(**PARAMS).fit(Xtr, ytr - btr)
    print(f"학습 {kst[0]:%m/%d} ~ {kst[split-1]:%m/%d} (연휴 없음) · 채점 {kst[split]:%m/%d} ~ {kst[-1]:%m/%d %H시}\n")

    res = {}
    for tag, flag in (("지금", False), ("공휴일=일요일", True)):
        X, y, b, h, t = rows(grid, kst, split, n, pm, ps, out, inn, 10**6,
                             np.random.default_rng(0), flag)
        d = m.predict(X)
        p = np.round(np.clip(b + np.where(np.abs(d) < T.DEADBAND, 0, d), 0, None))
        res[tag] = (p, y, b, h, t)

    _, y, b, h, t = res["지금"]
    days = np.array([kst[i].date() for i in t])
    print(f"2시간 뒤 평균 오차 (대)")
    print(f"{'날짜':12}{'':5}{'그대로 유지':>10}{'지금 모델':>10}{'공휴일=일요일':>14}")
    for d0 in sorted(set(days)):
        s = (h == 120) & (days == d0)
        if not s.any():
            continue
        tag = "추석" if d0 in HOLIDAYS else ("주말" if d0.weekday() >= 5 else "평일")
        mk = lambda p: np.abs(p[s] - y[s]).mean()
        print(f"{d0:%m/%d}({'월화수목금토일'[d0.weekday()]}) {tag:4}{mk(b):10.2f}"
              f"{mk(res['지금'][0]):10.2f}{mk(res['공휴일=일요일'][0]):14.2f}")

    hol = np.isin(days, list(HOLIDAYS))
    print()
    for lab, sel in (("추석 사흘", hol), ("나머지 날", ~hol)):
        for hm in (60, 120):
            s = (h == hm) & sel
            mk = lambda p: np.abs(p[s] - y[s]).mean()
            ok2 = lambda p: ((p[s] >= 2) == (y[s] >= 2)).mean()
            print(f"{lab} {hm//60}시간 뒤 — 평균 오차 유지 {mk(b):.2f} / 지금 {mk(res['지금'][0]):.2f} / "
                  f"일요일처리 {mk(res['공휴일=일요일'][0]):.2f}   "
                  f"2대 적중 {ok2(b):.1%} / {ok2(res['지금'][0]):.1%} / {ok2(res['공휴일=일요일'][0]):.1%}")


if __name__ == "__main__":
    main()
