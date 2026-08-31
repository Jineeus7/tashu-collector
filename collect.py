"""타슈 대여소의 자전거 거치 현황을 수집한다.

이름·좌표·주소는 사실상 고정값이라 stations.json에 따로 두고,
10분마다 쌓이는 스냅샷에는 실제로 변하는 parking_count만 담는다.
(전체를 매번 저장하면 1건당 408KB, 하루 59MB가 쌓인다.)
"""

import datetime
import json
import os
import urllib.request

API_URL = "https://bikeapp.tashu.or.kr:50041/v1/openapi/station"
API_TOKEN = os.environ["TASHU_API_TOKEN"]
KST = datetime.timezone(datetime.timedelta(hours=9))
STATIONS_PATH = "stations.json"

req = urllib.request.Request(API_URL, headers={"api-token": API_TOKEN})
with urllib.request.urlopen(req, timeout=30) as resp:
    stations = json.load(resp)["results"]

now = datetime.datetime.now(KST)

out_dir = os.path.join("data", f"{now:%Y-%m-%d}")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{now:%Y-%m-%d_%H-%M-%S}.json")
snapshot = {
    "collected_at": now.isoformat(),
    "counts": {s["id"]: s["parking_count"] for s in stations},
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(snapshot, f, ensure_ascii=False)
print(f"Saved {len(snapshot['counts'])} counts to {out_path}")

# 대여소 목록은 신설·폐지가 있을 때만 다시 쓴다.
meta = {
    s["id"]: {k: v for k, v in s.items() if k not in ("id", "parking_count")}
    for s in stations
}
old_meta = None
if os.path.exists(STATIONS_PATH):
    with open(STATIONS_PATH, encoding="utf-8") as f:
        old_meta = json.load(f)
if meta != old_meta:
    with open(STATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"Updated {STATIONS_PATH} ({len(meta)} stations)")
