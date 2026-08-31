"""타슈 대여소의 자전거 거치 현황을 원본 응답 그대로 수집한다.

나중에 어떤 필드가 필요해질지 미리 알 수 없으므로 응답을 가공하지 않는다.
다만 원본을 그대로 두면 1건당 386KB, 2주면 760MB가 쌓여 매 실행마다
저장소를 통째로 내려받는 워크플로우가 느려진다. gzip으로 56KB까지
줄이면 2주에 110MB로, 정보를 하나도 버리지 않고 감당 가능한 크기가 된다.
"""

import datetime
import gzip
import json
import os
import urllib.request

API_URL = "https://bikeapp.tashu.or.kr:50041/v1/openapi/station"
API_TOKEN = os.environ["TASHU_API_TOKEN"]
KST = datetime.timezone(datetime.timedelta(hours=9))

req = urllib.request.Request(API_URL, headers={"api-token": API_TOKEN})
with urllib.request.urlopen(req, timeout=30) as resp:
    payload = json.load(resp)

# 응답에는 수집 시각이 없으므로 파일명과 별개로 안에도 남겨둔다.
now = datetime.datetime.now(KST)
out_dir = os.path.join("data", f"{now:%Y-%m-%d}")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{now:%Y-%m-%d_%H-%M-%S}.json.gz")

with gzip.open(out_path, "wt", encoding="utf-8") as f:
    json.dump({"collected_at": now.isoformat(), "data": payload}, f, ensure_ascii=False)

print(f"Saved {payload['count']} stations to {out_path}")
