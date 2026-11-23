#!/usr/bin/env bash
# Runs the whole GPU training job on a RunPod pod, unattended, then deletes the pod.
# Everything lives on the network volume (/workspace), so results survive the pod and
# a rerun resumes: finished stages are skipped and training resumes from checkpoints.
#
#   nohup bash ml/cloud/pod_run.sh > /workspace/logs/run.log 2>&1 &
#
# Inputs on the volume: /workspace/n5k/ (Nutrition5k crops and their manifest rows,
# uploaded from the Mac; optional). Outputs: /workspace/results/.
# MAX_HOURS caps the run; when it ends (or every stage is done) the pod removes itself.
set -uo pipefail
W=/workspace
REPO=$W/repo
BRANCH=${BRANCH:-feat/gpu-training}
MAX_HOURS=${MAX_HOURS:-10}
KEEP_POD=${KEEP_POD:-0}
mkdir -p $W/logs $W/results $W/done
STATUS=$W/logs/status.txt
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $STATUS; }

finish() {
  log "finishing: $1"
  cp -f $REPO/ml/data/foodseg/*-metrics.json $REPO/ml/data/foodclass/*-metrics.json $W/results/ 2>/dev/null
  cp -f $REPO/ml/data/foodseg/*.pt $W/results/ 2>/dev/null
  for f in checkpoint.pt effv2.pt convnext.pt; do
    [ -f $REPO/ml/data/foodclass/$f ] && cp -f $REPO/ml/data/foodclass/$f $W/results/foodclass-$f
  done
  sync
  if [ "$KEEP_POD" = 1 ]; then log "KEEP_POD=1, leaving the pod running"; return; fi
  log "removing pod ${RUNPOD_POD_ID:-?}"
  runpodctl remove pod "$RUNPOD_POD_ID" >> $STATUS 2>&1 || runpodctl pod delete "$RUNPOD_POD_ID" >> $STATUS 2>&1 \
    || runpodctl stop pod "$RUNPOD_POD_ID" >> $STATUS 2>&1
}

# Hard cap on spend: whatever happens, the pod goes away after MAX_HOURS.
( sleep $((MAX_HOURS * 3600)); finish "time limit of ${MAX_HOURS}h reached" ) &
WATCHDOG=$!

stage() {  # stage <name> <command...>: runs once; a failure is logged and the job moves on.
  local name=$1; shift
  if [ -f $W/done/$name ]; then log "skip $name (done)"; return 0; fi
  log "start $name"
  local began=$SECONDS
  if "$@" > $W/logs/$name.log 2>&1; then
    touch $W/done/$name; log "done $name in $(( (SECONDS - began) / 60 )) min"
  else
    log "FAILED $name after $(( (SECONDS - began) / 60 )) min (see logs/$name.log)"; return 1
  fi
}

setup() {
  if [ -d $REPO/.git ]; then git -C $REPO fetch -q origin $BRANCH && git -C $REPO checkout -q $BRANCH && git -C $REPO reset -q --hard origin/$BRANCH
  else git clone -q --branch $BRANCH https://github.com/misoverfitting/3d_camera_visualization.git $REPO; fi
  pip install -q pyarrow scipy pillow huggingface_hub hf_transfer
  python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))"
}

download() {
  cd $REPO/ml
  HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download as d
d('ethz/food101', repo_type='dataset', local_dir='data/food101')
d('EduardoPacheco/FoodSeg103', repo_type='dataset', local_dir='data/FoodSeg103')"
}

prepare() {
  cd $REPO/ml
  python -m foodclass prepare --workers 12
  if [ -f $W/n5k/n5k_rows.csv ]; then  # Nutrition5k pseudo-labeled crops from the Mac.
    mkdir -p data/foodclass/images
    tar -xzf $W/n5k/nutrition5k.tgz -C data/foodclass/images
    tail -n +2 $W/n5k/n5k_rows.csv >> data/foodclass/manifest.csv
    echo "appended $(($(wc -l < $W/n5k/n5k_rows.csv) - 1)) Nutrition5k rows"
  fi
  wc -l data/foodclass/manifest.csv data/foodclass/regions.csv
}

WORKERS=$(( $(nproc) > 4 ? $(nproc) - 2 : 2 ))
seg() { cd $REPO/ml && python -m foodseg train --workers $WORKERS "$@"; }
cls() { cd $REPO/ml && python -m foodclass train --workers $WORKERS "$@"; }

log "run started on ${RUNPOD_POD_ID:-?} ($(nproc) cpus, max ${MAX_HOURS}h)"
stage setup setup || { finish "setup failed"; exit 1; }
stage download download || { finish "download failed"; exit 1; }
stage prepare prepare || { finish "prepare failed"; exit 1; }
# Most important first, so a time limit still leaves the main results.
stage seg-lraspp seg --arch lraspp --epochs ${SEG_EPOCHS:-80} --batch 32
stage cls-mobilenet cls --arch mobilenet_v3_large --epochs ${CLS_EPOCHS:-30} --batch 256 --lr 1e-3
stage seg-deeplabv3 seg --arch deeplabv3 --epochs ${SEG_EPOCHS:-80} --batch 32 --out dl.pt
stage cls-effv2 cls --arch efficientnet_v2_s --epochs ${CLS_EPOCHS:-30} --batch 128 --lr 5e-4 --out effv2.pt
kill $WATCHDOG 2>/dev/null
finish "all stages ran"
