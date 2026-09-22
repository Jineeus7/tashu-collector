"""대여 이력에서 대여소별 '평소 흐름'을 뽑아 둔다.

재고 숫자만으로는 0~2대짜리 대여소가 2시간 뒤 어떻게 될지 알 수 없다.
누가 빌려 가고 반납하느냐로 정해지는데 그 정보가 입력에 없기 때문이다.
대여 이력은 실시간이 아니지만, 출퇴근·등하교 흐름은 해마다 반복되므로
"이 대여소는 평일 18시에 보통 몇 대 빠지고 몇 대 들어온다"는 쓸 수 있다.

배포 중인 프로필은 20개월 전체·요일 7칸(all 7)으로 만든다. 9~10월치만 쓴
것과 9월 검증 적중률이 같았고(78.8% vs 78.9%), 사계절이 들어 있어 겨울에도
쓸 수 있다. 계절을 섞어 절대량이 낮게 나오는 건 모델이 학습 구간에서 배율로
흡수한다 — 계절이 바뀌면 그 계절 재고로 다시 학습해야 하는 이유다.

결과: npz — (요일 칸, 시각 24, 대여소) 두 장.
      out = 시간당 대여 건수, inn = 시간당 반납 건수.

실행: python3 flow_profile.py <대여이력 폴더> <출력 npz> <season|all> <2|7>
"""

import csv
import os
import sys
from collections import defaultdict
from datetime import date

import numpy as np

SEASON = ("24년09월", "24년10월", "25년09월", "25년10월")   # 지금과 같은 계절


def encoding_of(path):
    """월마다 인코딩이 다르다. 어떤 달은 utf-8, 어떤 달은 cp949 로 온다."""
    with open(path, "rb") as fh:
        first = fh.readline()
    for enc in ("utf-8-sig", "cp949"):
        try:
            if "자전거번호" in first.decode(enc):
                return enc
        except UnicodeDecodeError:
            pass
    raise SystemExit(f"{path} 인코딩을 못 알아냈다")


def parse(ts):
    """'2026-03-01 00:00:10' → (date, hour). 형식이 어긋나면 None."""
    try:
        return date(int(ts[0:4]), int(ts[5:7]), int(ts[8:10])), int(ts[11:13])
    except (ValueError, IndexError):
        return None


def build(raw_dir, months, n_day):
    """n_day: 2 면 평일/주말, 7 이면 요일별로 나눈다."""
    files = [os.path.join(raw_dir, f) for f in sorted(os.listdir(raw_dir))
             if f.endswith(".csv") and (months is None or any(m in f for m in months))]
    if not files:
        raise SystemExit(f"{raw_dir} 에서 {months} 파일을 못 찾았다")

    def bucket(d):
        return d.weekday() if n_day == 7 else int(d.weekday() >= 5)

    out = defaultdict(int)          # (요일칸, 시각, 대여소) → 대여 건수
    inn = defaultdict(int)          # 반납 건수
    days = [set() for _ in range(n_day)]

    for path in files:
        print(f"  읽는 중 {os.path.basename(path)}", flush=True)
        with open(path, encoding=encoding_of(path), errors="replace", newline="") as fh:
            r = csv.reader(fh)
            head = next(r)
            i_rt, i_rs = head.index("대여일시"), head.index("대여_대여소ID")
            i_bt, i_bs = head.index("반납일시"), head.index("반납_대여소ID")
            for row in r:
                if len(row) <= i_bs:
                    continue
                p = parse(row[i_rt])
                if p:
                    d, h = p
                    w = bucket(d)
                    days[w].add(d)
                    out[(w, h, row[i_rs])] += 1
                p = parse(row[i_bt])
                if p:
                    d, h = p
                    inn[(bucket(d), h, row[i_bs])] += 1

    sids = sorted({s for _, _, s in out} | {s for _, _, s in inn})
    idx = {s: i for i, s in enumerate(sids)}
    n_days = [max(len(d), 1) for d in days]
    print(f"  칸별 일수 {n_days} / 대여소 {len(sids)}곳")

    a_out = np.zeros((n_day, 24, len(sids)), dtype=np.float32)
    a_in = np.zeros((n_day, 24, len(sids)), dtype=np.float32)
    for (w, h, s), c in out.items():
        a_out[w, h, idx[s]] = c / n_days[w]
    for (w, h, s), c in inn.items():
        a_in[w, h, idx[s]] = c / n_days[w]
    return a_out, a_in, sids


def main():
    raw_dir, dest = sys.argv[1], sys.argv[2]
    months = None if sys.argv[3] == "all" else SEASON
    n_day = int(sys.argv[4])
    a_out, a_in, sids = build(raw_dir, months, n_day)
    np.savez_compressed(dest, out=a_out, inn=a_in, sids=np.array(sids))
    busy = np.argsort(-(a_out.sum(axis=(0, 1))))[:3]
    print(f"\n{dest} 저장")
    print("가장 붐비는 대여소의 첫 칸 시간당 대여 건수 (0~23시):")
    for i in busy[:3]:
        line = " ".join(f"{v:.0f}" for v in a_out[0, :, i])
        print(f"  {sids[i]}  {line}")


if __name__ == "__main__":
    main()
