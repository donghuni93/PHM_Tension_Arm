#!/usr/bin/env bash
# 세션 시작·/clear·압축 후: 현재 상태를 Claude 컨텍스트에 넣는다
cd "$CLAUDE_PROJECT_DIR/notes" || exit 0
echo "=== notes/00_NOW.md ==="
cat 00_NOW.md
