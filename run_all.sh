#!/usr/bin/env bash
# Reproduce everything end-to-end (~35 min on 8 CPU cores, most of it DistilBERT inference; download ~1.1 GB)
set -euo pipefail
cd "$(dirname "$0")/src"
python 01_download_and_prepare.py   | tee ../outputs/01_prepare_log.txt
python 02_clean_and_eda.py          | tee ../outputs/02_eda_log.txt
python 03_classify.py               | tee ../outputs/03_classify_log.txt
python 04_sentiment.py              | tee ../outputs/04_sentiment_log.txt
python 05_topics.py                 | tee ../outputs/05_topics_log.txt
python 06_analysis.py               | tee ../outputs/06_analysis_log.txt
