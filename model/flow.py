"""대여 이력에서 뽑은 '평소 흐름' 피처. 학습과 예측이 같은 코드를 쓴다.

재고 숫자만으로는 0~2대짜리 대여소가 2시간 뒤 어떻게 될지 알 수 없다.
누가 빌려 가고 반납하느냐로 정해지는데 그 정보가 재고에 없기 때문이다.
대여 이력 20개월에서 "이 대여소는 평일 18시에 보통 몇 대 빠지고 들어온다"를
뽑아 두면, 모델이 맞히려는 변화량과 같은 단위의 사전 지식이 된다.

학습(train.py)과 예측(predict.py)이 시각을 따로 계산하다 9시간이 어긋난
적이 있다. 계산을 이 파일 한 곳에 두어 두 쪽이 갈라지지 않게 한다.

프로필(flow_profile.npz)은 flow_profile.py 로 만든다.
"""

from datetime import date, timedelta

import numpy as np

NAMES = ["평소유출", "평소유입", "평소순증", "지금유출속도", "지금유입속도"]


def load(path, sids):
    """대여소 순서를 sids 에 맞춘다. 이력에 없는 곳은 NaN — 흐름이 0이라고
    말하면 거짓이고, XGBoost 는 결측을 알아서 다룬다."""
    z = np.load(path, allow_pickle=True)
    idx = {s: i for i, s in enumerate(z["sids"].tolist())}
    take = np.array([idx.get(s, -1) for s in sids])
    ok = take >= 0

    def pick(a):
        b = np.full((a.shape[0], 24, len(sids)), np.nan, dtype=np.float32)
        b[:, :, ok] = a[:, :, take[ok]]
        return b

    return pick(z["out"]), pick(z["inn"])


# 공휴일은 일요일로 본다. 모델에 공휴일 개념이 없어 추석을 평소 목·금으로
# 보고 출퇴근 흐름을 예측했고, 추석 당일 2시간 뒤 오차가 그대로 유지보다
# 35% 컸다. 일요일로 바꿔 넣자 연휴 손해의 약 4분의 3이 사라졌고, 평일·
# 주말 성능은 그대로였다. 명절 당일은 일요일보다도 조용해 여전히 그대로
# 유지보다 조금 못하다.
# 해마다 새 공휴일(대체공휴일 포함)을 여기에 더해야 한다.
HOLIDAYS = {
    date(2026, 1, 1), date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18),
    date(2026, 3, 2), date(2026, 5, 5), date(2026, 5, 25), date(2026, 6, 3),
    date(2026, 8, 17), date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 26),
    date(2026, 10, 5), date(2026, 10, 9), date(2026, 12, 25),
}


def weekday(d):
    """요일(월=0 … 일=6). 공휴일이면 일요일."""
    return 6 if d.date() in HOLIDAYS else d.weekday()


def _day(d, n_day):
    w = weekday(d)
    return w if n_day == 7 else int(w >= 5)


def cols(now, h_slots, si, out, inn):
    """now(한국 시각)부터 h_slots 칸(1칸=10분) 뒤까지 평소 유출·유입 합.
    시간당 요율을 10분 칸마다 1/6 씩 더한다. 자정·요일 경계도 칸마다 따진다."""
    nd = out.shape[0]
    o = np.zeros(len(si), dtype=np.float32)
    i_ = np.zeros(len(si), dtype=np.float32)
    for k in range(1, h_slots + 1):
        d = now + timedelta(minutes=10 * k)
        w = _day(d, nd)
        o += out[w, d.hour][si] / 6
        i_ += inn[w, d.hour][si] / 6
    w0 = _day(now, nd)
    return [o, i_, i_ - o, out[w0, now.hour][si], inn[w0, now.hour][si]]


# 어제·그제·지난주 같은 시각부터 실제로 얼마나 변했나. 대여 이력의 '평소 흐름'과
# 달리 트럭 움직임까지 들어 있다. 학습과 예측이 이 함수 하나를 같이 쓴다.
# 넣자 2시간 뒤 평균 오차가 0.806 → 0.789대, 2대 이상 적중률이 85.1 → 85.5%
# (튜닝에 안 쓴 9/23~10/2 세 기간 합산, 6시간 블록 재추출 95% 구간 +0.2~+0.6%p).
DAY, WEEK = 144, 1008                    # 칸 수 (1칸 = 10분)
HIST_NAMES = ["어제_뒤변화", "그제_뒤변화", "지난주_뒤변화", "지난주_대비", "3시간전차이", "어제_목표차"]


def history_cols(grid, t, h_slots, si):
    """grid: (시각, 대여소) 재고 격자, t: 지금 칸. 과거 칸이 없거나 비면 NaN."""
    def at(k):
        return grid[k][si] if k >= 0 else np.full(len(si), np.nan, dtype=np.float32)
    c = grid[t][si]
    return [at(t - DAY + h_slots) - at(t - DAY),
            at(t - 2 * DAY + h_slots) - at(t - 2 * DAY),
            at(t - WEEK + h_slots) - at(t - WEEK),
            c - at(t - WEEK),
            c - at(t - 18),
            at(t - DAY + h_slots) - c]
