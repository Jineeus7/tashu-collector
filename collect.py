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
import time
import urllib.error
import urllib.request

API_URL = "https://bikeapp.tashu.or.kr:50041/v1/openapi/station"
API_TOKEN = os.environ["TASHU_API_TOKEN"]
KST = datetime.timezone(datetime.timedelta(hours=9))

# 실행이 10분에 한 번이므로 재시도에 쓸 수 있는 시간은 넉넉하다.
ATTEMPTS = 3
BACKOFF_SECONDS = 5


def fetch():
    """일시적인 네트워크 장애로 슬롯을 통째로 놓치지 않도록 몇 번 다시 부른다.

    지금까지 실패한 실행은 전부 러너 쪽 이름 해석 실패(Errno -3)였고 바로
    다음 실행은 멀쩡했다. 몇 초만 기다렸다 다시 부르면 넘어가는 종류의
    장애라, 재시도가 없으면 10분치 스냅샷 하나가 그냥 사라진다.

    반면 토큰이 틀렸거나 하는 4xx는 다시 불러도 결과가 같으므로 즉시
    포기해서, 진짜 고쳐야 할 문제가 재시도 뒤에 가려지지 않게 한다.
    """
    for attempt in range(1, ATTEMPTS + 1):
        try:
            req = urllib.request.Request(API_URL, headers={"api-token": API_TOKEN})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.URLError as err:
            if isinstance(err, urllib.error.HTTPError) and 400 <= err.code < 500:
                raise
            if attempt == ATTEMPTS:
                raise
            wait = BACKOFF_SECONDS * attempt
            print(f"수집 실패 ({attempt}/{ATTEMPTS}): {err} — {wait}초 후 재시도")
            time.sleep(wait)


payload = fetch()

# 응답에는 수집 시각이 없으므로 파일명과 별개로 안에도 남겨둔다.
now = datetime.datetime.now(KST)
out_dir = os.path.join("data", f"{now:%Y-%m-%d}")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{now:%Y-%m-%d_%H-%M-%S}.json.gz")

with gzip.open(out_path, "wt", encoding="utf-8") as f:
    json.dump({"collected_at": now.isoformat(), "data": payload}, f, ensure_ascii=False)

print(f"Saved {payload['count']} stations to {out_path}")
