"""지금까지 모은 데이터로 낼 수 있는 가장 좋은 모델을 찾는다.

검증을 한 번에 끝내지 않는다. 기간을 넷으로 나눠
  - 튜닝 기간(9/20~9/23)에서만 피처·하이퍼파라미터·둔감폭을 고르고
  - 채점 기간 셋(9/23~9/26 추석, 9/26~9/29, 9/29~)에서 최종 성적을 잰다.
각 기간은 그 이전 데이터로만 학습한다(시간이 앞으로만 흐르게).
대여소 평균·변동도 기간마다 그 이전 데이터로만 다시 계산한다.

새로 보는 피처:
  어제_뒤변화   어제 같은 시각부터 h분 동안 실제로 얼마나 변했나 (트럭 포함)
  그제_뒤변화   그제 같은 시각 기준 같은 값
  지난주_뒤변화 7일 전 같은 시각 기준 같은 값
  지난주_대비   지금 재고 − 7일 전 같은 시각 재고
  3시간전차이   지금 − 3시간 전
  어제_목표차   어제 목표 시각 재고 − 지금 재고

실행: python3 experiments12.py ../data flow_profile.npz
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402
import train as T  # noqa: E402

DAY, WEEK = 144, 1008
TUNE = (datetime(2026, 9, 20), datetime(2026, 9, 23))
EVAL = [(datetime(2026, 9, 23), datetime(2026, 9, 26), "9/23~26 (추석)"),
        (datetime(2026, 9, 26), datetime(2026, 9, 29), "9/26~29"),
        (datetime(2026, 9, 29), datetime(2026, 10, 3), "9/29~10/2")]
BASE_PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.06, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=50, tree_method="hist",
                   n_jobs=-1, random_state=42)

RAW = ["cur", "d1", "d3", "d6", "d144", "hsin", "hcos", "dow", "wkend", "h"] + \
      [f"flow{k}" for k in range(5)] + \
      ["yday_fut", "yday2_fut", "wk_fut", "wk_lvl", "d18", "ytgt"]
BASE = RAW[:15]                                   # 지금 배포 모델과 같은 입력 (+ 대여소 평균·변동)
SETS = {
    "배포 중":            BASE,
    "+어제·그제 뒤변화":  BASE + ["yday_fut", "yday2_fut"],
    "+지난주":            BASE + ["yday_fut", "yday2_fut", "wk_fut", "wk_lvl"],
    "+전부":              BASE + ["yday_fut", "yday2_fut", "wk_fut", "wk_lvl", "d18", "ytgt"],
}


def log(*a):
    print(*a, flush=True)


def build(grid, kst, lo, hi, out, inn, cap, rng):
    n = grid.shape[0]
    cols = {k: [] for k in RAW}
    meta = {k: [] for k in ("st", "t", "hz", "cur", "nxt")}

    def at(idx, si):
        return grid[idx][si] if idx >= 0 else np.full(si.size, np.nan, np.float32)

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
            c = cur[si]
            hour = kst[t].hour + kst[t].minute / 60
            dw = flow.weekday(kst[t])
            vals = {
                "cur": c, "d1": c - grid[t - 1][si], "d3": c - grid[t - 3][si],
                "d6": c - grid[t - 6][si], "d144": c - grid[t - 144][si],
                "hsin": np.full(si.size, np.sin(2 * np.pi * hour / 24)),
                "hcos": np.full(si.size, np.cos(2 * np.pi * hour / 24)),
                "dow": np.full(si.size, float(dw)), "wkend": np.full(si.size, float(dw >= 5)),
                "h": np.full(si.size, h * 10.0),
                "yday_fut": at(t - DAY + h, si) - at(t - DAY, si),
                "yday2_fut": at(t - 2 * DAY + h, si) - at(t - 2 * DAY, si),
                "wk_fut": at(t - WEEK + h, si) - at(t - WEEK, si),
                "wk_lvl": c - at(t - WEEK, si),
                "d18": c - at(t - 18, si),
                "ytgt": at(t - DAY + h, si) - c,
            }
            for k, v in zip(range(5), flow.cols(kst[t], h, si, out, inn)):
                vals[f"flow{k}"] = v
            for k in RAW:
                cols[k].append(np.asarray(vals[k], np.float32))
            meta["st"].append(si); meta["t"].append(np.full(si.size, t, np.int32))
            meta["hz"].append(np.full(si.size, h * 10, np.int16))
            meta["cur"].append(c); meta["nxt"].append(nxt[si])
    R = {k: np.concatenate(v) for k, v in cols.items()}
    R.update({k: np.concatenate(v) for k, v in meta.items()})
    return R


def matrix(R, mask, names, pm, ps):
    X = [R[k][mask] for k in names] + [pm[R["st"][mask]], ps[R["st"][mask]]]
    return np.column_stack(X).astype(np.float32)


def metrics(pred, y, b, hz):
    out = {}
    for hm in (60, 120):
        s = hz == hm
        p, yy, bb = pred[s], y[s], b[s]
        out[hm] = {
            "1대": float(((p >= 1) == (yy >= 1)).mean()), "2대": float(((p >= 2) == (yy >= 2)).mean()),
            "±1대": float((np.abs(p - yy) <= 1).mean()), "오차": float(np.abs(p - yy).mean()),
            "유지_1대": float(((bb >= 1) == (yy >= 1)).mean()), "유지_2대": float(((bb >= 2) == (yy >= 2)).mean()),
            "유지_±1대": float((np.abs(bb - yy) <= 1).mean()), "유지_오차": float(np.abs(bb - yy).mean()),
        }
    return out


def to_count(delta, b, deadband):
    d = np.where(np.abs(delta) < deadband, 0.0, delta)
    return np.round(np.clip(b + d, 0, None))


def main():
    data_dir, prof = sys.argv[1], sys.argv[2]
    t_all = time.time()
    snaps = T.load_snapshots(data_dir)
    grid, sids, t0 = T.build_grid(snaps)
    n = grid.shape[0]
    kst = [t0 + timedelta(minutes=10 * k, hours=9) for k in range(n)]
    idx_of = lambda dt: next((k for k in range(n) if kst[k] >= dt), n)
    out, inn = flow.load(prof, sids)
    log(f"스냅샷 {len(snaps)}개 · {kst[0]:%m/%d %H시} ~ {kst[-1]:%m/%d %H시} · 대여소 {len(sids)}곳")

    tune_lo, tune_hi = idx_of(TUNE[0]), idx_of(TUNE[1])
    A = build(grid, kst, 0, n, out, inn, 120, np.random.default_rng(42))      # 학습용
    B = build(grid, kst, tune_lo, n, out, inn, 400, np.random.default_rng(7))  # 채점용
    log(f"학습용 {len(A['cur']):,}행 · 채점용 {len(B['cur']):,}행 ({time.time()-t_all:.0f}초)")

    from xgboost import XGBRegressor, XGBClassifier

    def fold(lo_dt, hi_dt):
        lo, hi = idx_of(lo_dt), idx_of(hi_dt)
        pm = np.nan_to_num(np.nanmean(grid[:lo], 0)); ps = np.nan_to_num(np.nanstd(grid[:lo], 0))
        tr = (A["t"] + A["hz"] // 10) < lo
        te = (B["t"] >= lo) & (B["t"] < hi)
        return pm, ps, tr, te

    def run_reg(names, params, lo_dt, hi_dt, deadbands=(0.8,)):
        pm, ps, tr, te = fold(lo_dt, hi_dt)
        m = XGBRegressor(**params).fit(matrix(A, tr, names, pm, ps), (A["nxt"] - A["cur"])[tr])
        delta = m.predict(matrix(B, te, names, pm, ps))
        y, b, hz = B["nxt"][te], B["cur"][te], B["hz"][te]
        return {db: metrics(to_count(delta, b, db), y, b, hz) for db in deadbands}, (delta, y, b, hz, B["t"][te])

    res = {"tune": {}, "eval": {}}

    # 1) 피처 묶음 — 튜닝 기간에서 고른다
    log("\n== 1. 피처 묶음 (튜닝 기간) ==")
    for name, names in SETS.items():
        r, _ = run_reg(names, BASE_PARAMS, *TUNE)
        m = r[0.8][120]; res["tune"][name] = r[0.8]
        log(f"  {name:16} 2시간: 2대 {m['2대']:.1%} · 1대 {m['1대']:.1%} · ±1 {m['±1대']:.1%} · 오차 {m['오차']:.3f}"
            f"   (유지 2대 {m['유지_2대']:.1%}, 오차 {m['유지_오차']:.3f})")
    best_set = min(SETS, key=lambda k: res["tune"][k][120]["오차"])
    log(f"  → 고른 피처: {best_set}")

    # 2) 하이퍼파라미터 — 튜닝 기간
    log("\n== 2. 하이퍼파라미터 (튜닝 기간) ==")
    grid_p = {
        "기본 (깊이6·400그루)": BASE_PARAMS,
        "깊이8·600": {**BASE_PARAMS, "max_depth": 8, "n_estimators": 600, "learning_rate": 0.05},
        "깊이10·800": {**BASE_PARAMS, "max_depth": 10, "n_estimators": 800, "learning_rate": 0.04, "min_child_weight": 100},
        "깊이8·800·잎200": {**BASE_PARAMS, "max_depth": 8, "n_estimators": 800, "learning_rate": 0.05, "min_child_weight": 200},
        "깊이6·800·느리게": {**BASE_PARAMS, "n_estimators": 800, "learning_rate": 0.03},
        "기본+절대오차 목표": {**BASE_PARAMS, "objective": "reg:absoluteerror"},
    }
    tune_p = {}
    for name, p in grid_p.items():
        r, _ = run_reg(SETS[best_set], p, *TUNE, deadbands=(0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2))
        tune_p[name] = r
        m = r[0.8][120]
        log(f"  {name:18} 2시간: 2대 {m['2대']:.1%} · ±1 {m['±1대']:.1%} · 오차 {m['오차']:.3f}")
    best_p = min(grid_p, key=lambda k: tune_p[k][0.8][120]["오차"])
    log(f"  → 고른 설정: {best_p}")

    # 3) 둔감폭 — 튜닝 기간 (2시간 오차 기준, 2대 적중률 같이 표시)
    log("\n== 3. 둔감폭 (튜닝 기간, 고른 설정) ==")
    for db, r in tune_p[best_p].items():
        m = r[120]; log(f"  {db:>4}  2시간: 2대 {m['2대']:.1%} · ±1 {m['±1대']:.1%} · 오차 {m['오차']:.3f}")
    best_db = min(tune_p[best_p], key=lambda d: tune_p[best_p][d][120]["오차"])
    log(f"  → 고른 둔감폭: {best_db}")

    # 4) 최종 채점 — 채점 기간 셋. 튜닝에서 고른 것만 가져온다
    log("\n== 4. 최종 채점 (튜닝에 안 쓴 기간) ==")
    pooled = {"배포 중": [], "새 모델": [], "분류(2대)": [], "분류(1대)": []}
    for lo_dt, hi_dt, lab in EVAL:
        r0, raw0 = run_reg(SETS["배포 중"], BASE_PARAMS, lo_dt, hi_dt)
        r1, raw1 = run_reg(SETS[best_set], grid_p[best_p], lo_dt, hi_dt, deadbands=(best_db,))
        pm, ps, tr, te = fold(lo_dt, hi_dt)
        Xtr, Xte = matrix(A, tr, SETS[best_set], pm, ps), matrix(B, te, SETS[best_set], pm, ps)
        cls = {}
        for k in (1, 2):
            pc = {kk: v for kk, v in grid_p[best_p].items() if kk != "objective"}
            c = XGBClassifier(**pc).fit(Xtr, (A["nxt"][tr] >= k).astype(int))
            cls[k] = c.predict_proba(Xte)[:, 1] >= 0.5
        a, b_ = r0[0.8], r1[best_db]
        res["eval"][lab] = {"배포 중": a, "새 모델": b_}
        for hm in (60, 120):
            s = raw1[3] == hm
            y = raw1[1][s]
            c1 = (cls[1][s] == (y >= 1)).mean(); c2 = (cls[2][s] == (y >= 2)).mean()
            res["eval"][lab][f"분류_{hm}"] = {"1대": float(c1), "2대": float(c2)}
        m0, m1 = a[120], b_[120]
        log(f"  [{lab}] 2시간 뒤")
        log(f"     그대로 유지  2대 {m0['유지_2대']:.1%} · 1대 {m0['유지_1대']:.1%} · ±1 {m0['유지_±1대']:.1%} · 오차 {m0['유지_오차']:.3f}")
        log(f"     배포 중      2대 {m0['2대']:.1%} · 1대 {m0['1대']:.1%} · ±1 {m0['±1대']:.1%} · 오차 {m0['오차']:.3f}")
        log(f"     새 모델      2대 {m1['2대']:.1%} · 1대 {m1['1대']:.1%} · ±1 {m1['±1대']:.1%} · 오차 {m1['오차']:.3f}")
        log(f"     분류 모델    2대 {res['eval'][lab]['분류_120']['2대']:.1%} · 1대 {res['eval'][lab]['분류_120']['1대']:.1%}")
        # 블록 재추출용으로 행 단위 결과를 모은다
        d0, y0, bb0, hz0, t0_ = raw0; d1, y1, bb1, hz1, t1_ = raw1
        pooled["배포 중"].append((to_count(d0, bb0, 0.8), y0, bb0, hz0, t0_))
        pooled["새 모델"].append((to_count(d1, bb1, best_db), y1, bb1, hz1, t1_))
        pooled["분류(2대)"].append((cls[2], y1, bb1, hz1, t1_))
        pooled["분류(1대)"].append((cls[1], y1, bb1, hz1, t1_))

    # 5) 세 기간을 합쳐 6시간 블록 재추출로 차이의 95% 구간
    log("\n== 5. 세 기간 합산 · 새 모델 − 배포 중 (95% 구간, 6시간 블록) ==")
    cat = lambda L, i: np.concatenate([x[i] for x in L])
    p0, y, bb, hz, tt = (cat(pooled["배포 중"], i) for i in range(5))
    p1 = cat(pooled["새 모델"], 0)
    c2 = cat(pooled["분류(2대)"], 0); c1 = cat(pooled["분류(1대)"], 0)
    blocks = (tt - tt.min()) // 36
    rng = np.random.default_rng(1)
    summary = {}
    for hm in (60, 120):
        s = hz == hm
        def boot(a, b):
            ids = np.unique(blocks[s]); per = {k: (a[s][blocks[s] == k], b[s][blocks[s] == k]) for k in ids}
            g = []
            for _ in range(2000):
                pk = rng.choice(ids, ids.size)
                g.append(np.concatenate([per[k][0] for k in pk]).mean() - np.concatenate([per[k][1] for k in pk]).mean())
            return np.percentile(g, [2.5, 97.5])
        rows = {
            "2대": (((p1 >= 2) == (y >= 2)), ((p0 >= 2) == (y >= 2)), ((bb >= 2) == (y >= 2)), (c2 == (y >= 2))),
            "1대": (((p1 >= 1) == (y >= 1)), ((p0 >= 1) == (y >= 1)), ((bb >= 1) == (y >= 1)), (c1 == (y >= 1))),
            "±1대": ((np.abs(p1 - y) <= 1), (np.abs(p0 - y) <= 1), (np.abs(bb - y) <= 1), None),
            "오차": (-np.abs(p1 - y), -np.abs(p0 - y), -np.abs(bb - y), None),
        }
        log(f"  {hm//60}시간 뒤")
        for k, (new, old, keep, clf) in rows.items():
            lo_, hi_ = boot(new.astype(float), old.astype(float))
            f = (lambda v: f"{-v:.3f}대") if k == "오차" else (lambda v: f"{v:.1%}")
            extra = f" · 분류 {clf[s].mean():.1%}" if clf is not None else ""
            log(f"    {k:4} 유지 {f(keep[s].mean())} · 배포 중 {f(old[s].mean())} · 새 모델 {f(new[s].mean())}{extra}"
                f"   차이 [{lo_*100:+.1f} ~ {hi_*100:+.1f}]" + ("%p" if k != "오차" else " (×0.01대, +면 개선)"))
            summary[f"{hm}_{k}"] = {"유지": float(keep[s].mean()), "배포": float(old[s].mean()),
                                    "새": float(new[s].mean()), "ci": [float(lo_), float(hi_)],
                                    **({"분류": float(clf[s].mean())} if clf is not None else {})}
    res.update(best_set=best_set, best_params=best_p, best_deadband=float(best_db), summary=summary)
    json.dump(res, open("experiments12_result.json", "w"), ensure_ascii=False, indent=1, default=str)
    log(f"\n끝 ({(time.time()-t_all)/60:.0f}분)")


if __name__ == "__main__":
    main()
