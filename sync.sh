#!/usr/bin/env bash
# 저장소에 쌓인 스냅샷을 로컬 보관함으로 복사한다. 하루 한 번 실행하면 된다.
#
#   ./sync.sh                                 # ~/tashu-archive 로 복사
#   TASHU_ARCHIVE=/Volumes/USB/tashu ./sync.sh  # 다른 위치로 복사
#
# 이미 복사한 파일은 건드리지 않으므로 몇 번을 실행해도 안전하다.

set -euo pipefail

ARCHIVE="${TASHU_ARCHIVE:-$HOME/tashu-archive}"
cd "$(dirname "$0")"

git pull --quiet
mkdir -p "$ARCHIVE"
rsync -a --ignore-existing data/ "$ARCHIVE/"

echo "보관함: $ARCHIVE"
echo "  파일 수: $(find "$ARCHIVE" -name '*.json.gz' | wc -l | tr -d ' ')개"
echo "  크기:    $(du -sh "$ARCHIVE" | cut -f1)"
