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
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

API_URL = "https://bikeapp.tashu.or.kr:50041/v1/openapi/station"
API_TOKEN = os.environ["TASHU_API_TOKEN"]
KST = datetime.timezone(datetime.timedelta(hours=9))

# 실행이 10분에 한 번이므로 재시도에 쓸 수 있는 시간은 넉넉하다.
#
# 15초로 잡으면 대기만 15+30=45초, 여기에 이름 해석이 실패를 돌려주는 데
# 걸리는 시간이 시도당 10초쯤 더 붙어 실제로는 약 75초를 버틴다. 실패한
# 실행들이 이미 40~50초를 버티고도 죽었기 때문에 그보다는 길어야 의미가
# 있고, 10분 주기에 비하면 여전히 다음 슬롯을 침범할 여지가 없다.
ATTEMPTS = 3
BACKOFF_SECONDS = 15

API_HOST = urllib.parse.urlsplit(API_URL).hostname
# 실패한 순간에 러너의 이름 해석이 통째로 죽었는지 견줘볼 대조군.
CONTROL_HOST = "github.com"


def diagnose():
    """이름 해석이 왜 실패했는지 가를 단서를 실패한 그 순간에 남긴다.

    Errno -3 한 줄만으로는 러너의 리졸버가 죽은 것인지 tashu 쪽 네임서버가
    죽은 것인지 구분할 수 없는데, 둘은 처방이 정반대다. 전자면 재시도를
    늘리는 게 맞고 후자면 아무리 기다려도 소용이 없다. 대조군을 하나 같이
    조회해 두면 다음 실패 때 로그만 보고 둘을 가를 수 있다.

    조회가 또 막히면 호스트당 10초쯤 잡아먹지만, 이미 이번 슬롯을 포기하기로
    한 뒤에만 부르므로 스냅샷이 더 손해를 보지는 않는다.
    """
    for host in (API_HOST, CONTROL_HOST):
        try:
            socket.getaddrinfo(host, None)
            print(f"  진단: {host} 이름 해석 정상", flush=True)
        except OSError as err:
            print(f"  진단: {host} 이름 해석 실패 ({err})", flush=True)


def fetch():
    """일시적인 네트워크 장애로 슬롯을 통째로 놓치지 않도록 몇 번 다시 부른다.

    지금까지 실패한 실행은 전부 러너 쪽 이름 해석 실패(Errno -3)였고 바로
    다음 실행은 멀쩡했다. 다만 실패한 실행의 소요 시간을 재보면 40~50초를
    버티고도 세 번 모두 실패했으므로, 몇 초짜리 깜빡임이 아니라 최소 그
    정도는 지속되는 장애다. 재시도가 없으면 10분치 스냅샷 하나가 사라진다.

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
                diagnose()
                raise
            wait = BACKOFF_SECONDS * attempt
            # flush 하지 않으면 파이썬이 출력을 모아뒀다 끝에 한꺼번에 내보내
            # Actions 로그의 시각이 전부 같은 순간으로 찍힌다. 어느 시도가
            # 언제 실패했는지가 곧 진단 근거이므로 즉시 내보낸다.
            print(f"수집 실패 ({attempt}/{ATTEMPTS}): {err} — {wait}초 후 재시도", flush=True)
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
