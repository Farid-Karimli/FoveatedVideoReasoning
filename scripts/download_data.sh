#!/bin/bash
# Download the project's video data on the SCC. Submit as a batch job, never on
# the login node:
#   qsub -P proactiveai -N fvr_download -pe omp 1 -l h_rt=12:00:00 -j y \
#        -o /projectnb/proactiveai/data/download.log scripts/download_data.sh <repo_dir>
# Re-running is safe: finished files are skipped, partial files are restarted.
set -u
REPO=${1:-$PWD}
DATA=/projectnb/proactiveai/data

# CaptainCook4D: 29 extra recordings (360p for gaze prediction, 4K for frames).
cd $DATA/captaincook4d/gopro
while read r u4 u3; do
  [ -s 360p/${r}_360p.mp4 ] || { curl -sL --retry 5 -o 360p/${r}_360p.mp4.part "$u3" && mv 360p/${r}_360p.mp4.part 360p/${r}_360p.mp4; }
  [ -s 4k/${r}_4k.mp4 ] || { curl -sL --retry 5 -o 4k/${r}_4k.mp4.part "$u4" && mv 4k/${r}_4k.mp4.part 4k/${r}_4k.mp4; }
  echo "cc4d $r done"
done < $REPO/configs/data/captaincook4d_extra_links.txt

# HD-EPIC: all MP4 videos, frame-time CSVs and MPS gaze/hand files (CC BY-NC 4.0).
B=https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47
mkdir -p $DATA/hd-epic && cd $DATA/hd-epic
[ -d annotations ] || git clone -q --depth 1 https://github.com/hd-epic/hd-epic-annotations annotations
while read f; do
  [ -s "$f" ] && continue
  mkdir -p "$(dirname "$f")"
  curl -sL --retry 5 -o "$f.part" "$B/$f" && mv "$f.part" "$f"
done < $REPO/configs/data/hdepic_files.txt
echo "hd-epic done: $(ls Videos/*/*.mp4 | wc -l) videos"
echo ALLDONE
