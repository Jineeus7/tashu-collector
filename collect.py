import datetime
import json
import os
import urllib.request

API_URL = "https://bikeapp.tashu.or.kr:50041/v1/openapi/station"
API_TOKEN = os.environ["TASHU_API_TOKEN"]

req = urllib.request.Request(API_URL, headers={"api-token": API_TOKEN})
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.load(resp)

KST = datetime.timezone(datetime.timedelta(hours=9))
now = datetime.datetime.now(KST)
record = {"collected_at": now.isoformat(), "data": data}

out_dir = os.path.join("data", f"{now:%Y-%m-%d}")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{now:%Y-%m-%d_%H-%M-%S}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False)

print(f"Saved to {out_path}")
