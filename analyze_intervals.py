"""수집 간격이 얼마나 규칙적인지 확인한다."""

import datetime
import glob
import json
import statistics

paths = sorted(glob.glob("data/*/*.json"))
if len(paths) < 2:
    raise SystemExit(f"수집 파일이 {len(paths)}개뿐입니다. 최소 2개는 있어야 간격을 잴 수 있어요.")

times = []
for path in paths:
    with open(path, encoding="utf-8") as f:
        times.append(datetime.datetime.fromisoformat(json.load(f)["collected_at"]))
times.sort()

gaps = [(b - a).total_seconds() / 60 for a, b in zip(times, times[1:])]

print(f"수집 건수: {len(times)}개")
print(f"기간: {times[0]:%Y-%m-%d %H:%M} ~ {times[-1]:%Y-%m-%d %H:%M} (KST)")
print()
print(f"간격 중앙값: {statistics.median(gaps):.1f}분")
print(f"간격 평균:   {statistics.mean(gaps):.1f}분")
print(f"간격 최소:   {min(gaps):.1f}분")
print(f"간격 최대:   {max(gaps):.1f}분")
print()

buckets = [
    ("10분 이내", lambda g: g <= 10),
    ("10~15분", lambda g: 10 < g <= 15),
    ("15~30분", lambda g: 15 < g <= 30),
    ("30분 초과", lambda g: g > 30),
]
print("간격 분포")
for label, match in buckets:
    n = sum(1 for g in gaps if match(g))
    print(f"  {label:>9}: {n:4d}건 ({n / len(gaps) * 100:5.1f}%)")

worst = sorted(zip(times[1:], gaps), key=lambda pair: -pair[1])[:5]
print()
print("가장 큰 구멍 5건")
for t, gap in worst:
    print(f"  {t:%m-%d %H:%M} 직전에 {gap:.1f}분 공백")
