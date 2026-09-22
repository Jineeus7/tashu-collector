"""대여 이력에서 뽑은 '평소 흐름' 피처. 학습과 예측이 같은 코드를 쓴다.

재고 숫자만으로는 0~2대짜리 대여소가 2시간 뒤 어떻게 될지 알 수 없다.
누가 빌려 가고 반납하느냐로 정해지는데 그 정보가 재고에 없기 때문이다.
대여 이력 20개월에서 "이 대여소는 평일 18시에 보통 몇 대 빠지고 들어온다"를
뽑아 두면, 모델이 맞히려는 변화량과 같은 단위의 사전 지식이 된다.

학습(train.py)과 예측(predict.py)이 시각을 따로 계산하다 9시간이 어긋난
적이 있다. 계산을 이 파일 한 곳에 두어 두 쪽이 갈라지지 않게 한다.

프로필(flow_profile.npz)은 flow_profile.py 로 만든다.
"""

from datetime import timedelta

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


def _day(d, n_day):
    return d.weekday() if n_day == 7 else int(d.weekday() >= 5)


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
