#!/usr/bin/env bash

python augment_main.py \
    --case-study-name "climate-monitoring" \
    --tracer-id "test" \
    --job-id "1" \
    --kp-auth-token "test123" --kp-host "localhost" --kp-port "8000" --kp-scheme "http" \
     --log-level "INFO"
