#!/bin/bash
cd /kaggle/working || exit 9
gitc() { git -c credential.helper='!f(){ echo username=x; echo "password=$GHT"; };f' "$@"; }
echo "=== fetch ==="
gitc fetch origin main 2>&1 | tail -3 || { echo "FETCH_FAIL"; exit 2; }
A=$(git rev-list --count origin/main..HEAD)
B=$(git rev-list --count HEAD..origin/main)
echo "ahead=$A behind=$B"
if [ "$B" = "0" ]; then
  echo "=== no divergence, plain push ==="
  gitc push origin main 2>&1 | tail -4 && echo "PUSH_DONE"
  exit $?
fi
git diff --name-only "origin/main...HEAD" | sort > /tmp/mine.txt
git diff --name-only "HEAD...origin/main" | sort > /tmp/theirs.txt
OVERLAP=$(comm -12 /tmp/mine.txt /tmp/theirs.txt)
if [ -n "$OVERLAP" ]; then
  echo "CONFLICT_RISK (files changed on BOTH sides):"; echo "$OVERLAP" | head -20; exit 3
fi
echo "=== no file overlap; rebase + push ==="
gitc pull --rebase origin main 2>&1 | tail -5 || { echo "REBASE_FAIL"; git rebase --abort 2>/dev/null; exit 4; }
gitc push origin main 2>&1 | tail -4 && echo "PUSH_DONE"
