#!/usr/bin/env bash
# Fetch the public raw files of the four data sets and the embedding model (no login; paper Section VI-A, Datasets).
# See data/README.md for the sources.
#   data/download.sh RAW [amazon] [arxiv] [model] [yfcc] [youtube8m-plan]        (default: all)
# The Hugging Face files are fetched at the revision named by HF_REV_* (default: main); data/README.md lists the revisions used.
set -u
R=${1:?usage: download.sh RAW_DIR [what ...]}; shift
WHAT=${*:-amazon arxiv model yfcc youtube8m-plan}
FAILED=0
HF=https://huggingface.co
dl() { # url out
  if [ -s "$2" ]; then echo "exists $2"; return; fi
  mkdir -p "$(dirname "$2")"
  if curl -fsSL --retry 5 --retry-delay 10 -o "$2.part" "$1"; then mv "$2.part" "$2"; echo "ok $2"
  else rm -f "$2.part"; echo "FAIL $1"; FAILED=1; fi
}
for w in $WHAT; do
  case $w in
    amazon)   # Amazon Reviews 2023, item metadata of the category Sports and Outdoors
      dl $HF/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/${HF_REV_AMAZON:-main}/raw/meta_categories/meta_Sports_and_Outdoors.jsonl $R/amazon/meta_Sports_and_Outdoors.jsonl ;;
    arxiv)    # arXiv metadata snapshot, 11 parquet shards
      for i in $(seq -w 0 10); do
        dl $HF/datasets/librarian-bots/arxiv-metadata-snapshot/resolve/${HF_REV_ARXIV:-main}/data/train-000$i-of-00011.parquet $R/arxiv/train-000$i.parquet
      done ;;
    model)    # sentence-embedding model
      for f in config.json tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json model.safetensors modules.json sentence_bert_config.json 1_Pooling/config.json; do
        dl $HF/sentence-transformers/all-MiniLM-L6-v2/resolve/${HF_REV_MODEL:-main}/$f $R/models/all-MiniLM-L6-v2/$f
      done ;;
    yfcc)     # YFCC-10M, filtered-search track of the NeurIPS'23 Big-ANN competition: base, tags, the two official query sets
      for f in base.10M.u8bin base.metadata.10M.spmat query.public.100K.u8bin query.private.2727415019.100K.u8bin; do
        dl https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/$f $R/yfcc/$f
      done ;;
    youtube8m-plan)   # YouTube-8M video-level features: the download plan of the training shards (data/prepare_youtube8m.py fetch reads it)
      dl http://data.yt8m.org/2/download_plans/video_train.json $R/video_train.json ;;
    *) echo "unknown item $w"; FAILED=1 ;;
  esac
done
if [ "$FAILED" = 0 ]; then echo DONE; else echo "some downloads failed"; exit 1; fi
