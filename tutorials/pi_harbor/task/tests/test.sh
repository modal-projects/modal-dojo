#!/bin/bash
answer=$(tr -d '[:space:]' < /app/answer.txt 2>/dev/null)
if [ "$answer" = "$(cat /tests/answer.txt)" ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
