#!/usr/bin/env bash
# 세션 시작·/clear·압축 후: 현재 상태와 최신 인수인계 노트를 Claude 컨텍스트에 넣는다
cd "$CLAUDE_PROJECT_DIR/vault" || exit 0
echo "=== vault/00_NOW.md ==="
cat 00_NOW.md
latest=$(ls -1 50_세션/*.md 2>/dev/null | sort -V | tail -n 1)
if [ -n "$latest" ]; then
  echo; echo "=== vault/$latest ==="; cat "$latest"
fi
