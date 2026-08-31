"""수집한 스냅샷을 모델 학습에 쓰기 좋은 형태로 읽어들인다.

    import load
    rows = load.rows()        # [(수집시각, 대여소ID, 거치수), ...]
    meta = load.stations()    # {대여소ID: {name, address, x_pos, ...}}

pandas를 쓴다면:

    import pandas as pd, load
    df = pd.DataFrame(load.rows(), columns=["time", "station", "count"])

저장소의 data/ 대신 로컬 보관함을 읽으려면 TASHU_DATA 환경변수를 지정한다.
"""

import datetime
import glob
import gzip
import json
import os

DATA_DIR = os.environ.get("TASHU_DATA", "data")


def paths():
    """스냅샷 파일 경로를 시각 순으로 돌려준다."""
    return sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.json.gz")))


def snapshots():
    """(수집시각, 원본 응답)을 시각 순으로 넘겨준다."""
    for path in paths():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            snap = json.load(f)
        yield datetime.datetime.fromisoformat(snap["collected_at"]), snap["data"]


def rows():
    """(수집시각, 대여소ID, 거치수) 목록."""
    return [
        (t, s["id"], s["parking_count"])
        for t, data in snapshots()
        for s in data["results"]
    ]


def stations():
    """가장 최근 스냅샷 기준 대여소 정보. 시간에 따라 변하는 거치수는 뺀다."""
    latest = paths()
    if not latest:
        return {}
    with gzip.open(latest[-1], "rt", encoding="utf-8") as f:
        data = json.load(f)["data"]
    return {
        s["id"]: {k: v for k, v in s.items() if k not in ("id", "parking_count")}
        for s in data["results"]
    }
