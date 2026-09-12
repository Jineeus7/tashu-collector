"""모델이 '그대로 유지'보다 적중률이 낮은 이유는, 아무 일도 없는 대부분의
순간에도 조금씩 움직이기 때문이다. 확신이 설 때만 움직이게 하면 어떨까.

  둔감폭(τ): |예측 변화량| 이 τ 미만이면 그냥 지금 값을 그대로 쓴다.

τ 를 올리면 적중률은 '그대로 유지' 쪽으로 수렴하고, 큰 변화 감지율은
떨어진다. 그 교환비를 본다.

실행: python3 experiments2.py <스냅샷 폴더>
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import (HORIZONS, PARAMS, build_grid, hour_profile,  # noqa: E402
                         load_snapshots, make_featfn, make_rows, near_matrix)
from datetime import timedelta  # noqa: E402


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
    near = near_matrix(coord)
    fn = make_featfn(prof_mean, prof_std, prof_hour, near,
                     use_hour=False, use_near=False, use_lag=False)

    Xtr, ytr, btr, _ = make_rows(grid, kst, 0, split, fn, 120,
                                 np.random.default_rng(42))
    Xva, yva, bva, hva = make_rows(grid, kst, split, n, fn, 400,
                                   np.random.default_rng(7))

    from xgboost import XGBRegressor
    m = XGBRegressor(**PARAMS).fit(Xtr, ytr - btr)
    delta = m.predict(Xva)

    print(f"{'둔감폭':>6}{'10분':>7}{'30분':>7}{'1시간':>7}{'2시간':>7}"
          f"{'큰변화':>9}{'움직인비율':>11}")

    for tau in (0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 99.0):
        d = np.where(np.abs(delta) < tau, 0.0, delta)
        pred = np.round(np.clip(bva + d, 0, None))
        line = [f"{tau:6.1f}" if tau < 99 else "  유지"]
        for h in (10, 30, 60, 120):
            s = hva == h
            line.append(f"{((pred[s] >= 1) == (yva[s] >= 1)).mean():7.1%}")
        chg = (hva == 60) & (np.abs(yva - bva) >= 3)
        det = ((np.sign(pred[chg] - bva[chg]) == np.sign(yva[chg] - bva[chg]))
               & (np.abs(pred[chg] - bva[chg]) >= 1)).mean()
        moved = (pred != bva).mean()
        line.append(f"{det:9.0%}{moved:11.0%}")
        print("".join(line))

    print("\n움직인비율 = 예측이 '지금 값'과 달라진 건의 비율")


if __name__ == "__main__":
    main()
