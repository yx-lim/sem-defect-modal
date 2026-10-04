#!/bin/bash
# usage: drive_image.sh <image_id> <src.tif> <out_v001_dir> [ref_src ref_id]
# pass1 -> image_review -> pass2 -> diff image_review -> final -> validate
set -eo pipefail
ID=$1; SRC=$2; OUT=$3; REFS=$4; REFID=$5
PY=/home/ubuntu/sem_b3/.venv/bin/python
cd /home/ubuntu/sem_b3/pipeline
REF=(); RS=(); if [ -n "$REFS" ]; then REF=(--reference-src "$REFS" --reference-id "$REFID"); RS=(--reference-src "$REFS"); fi
t() { echo "[$(date -u +%FT%TZ) $(date +%s)] $*"; }
mkdir -p "$OUT"
t start $ID
[ -f "$OUT/pass1/annotation.json" ] || $PY -m sem84.cli run --image-id $ID --src $SRC --out $OUT --stage pass1 --threads 4 "${REF[@]}"
t pass1_done
$PY -m sem84.image_review --stage-dir $OUT/pass1 --src $SRC --work-dir $OUT/.verify/pass1 --out $OUT/review_pass2.json --reviewer auto-verifier-pass1 "${RS[@]}"
t review1_done
$PY -m sem84.cli run --image-id $ID --src $SRC --out $OUT --stage pass2 --threads 4 --force --review $OUT/review_pass2.json "${REF[@]}"
t pass2_done
$PY -m sem84.image_review --stage-dir $OUT/pass2 --src $SRC --work-dir $OUT/.verify/pass2 --out $OUT/review_final.json --reviewer auto-verifier-pass2 --diff $OUT/pass2/pass_diff.json --prev-policy $OUT/.verify/pass1/policy.json "${RS[@]}"
t review2_done
$PY -m sem84.cli run --image-id $ID --src $SRC --out $OUT --stage final --threads 4 --force --review $OUT/review_final.json "${REF[@]}"
t final_done
$PY -m sem84.cli validate $OUT/final > $OUT/validate.json
t validated $(python3 -c "import json;print(json.load(open('$OUT/validate.json'))['ok'])")
