import datetime
import json
import os
import urllib.request

API_URL = "https://bikeapp.tashu.or.kr:50041/v1/openapi/station"
API_TOKEN = os.environ["TASHU_API_TOKEN"]

req = urllib.request.Request(API_URL, headers={"api-token": API_TOKEN})
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.load(resp)

now = datetime.datetime.now(datetime.timezone.utc)
record = {"collected_at": now.isoformat(), "data": data}

out_dir = "data"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{now:%Y-%m-%d}.jsonl")
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved to {out_path}")
