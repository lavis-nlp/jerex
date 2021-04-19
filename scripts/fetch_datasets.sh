#!/usr/bin/env bash
curr_dir=$(pwd)

mkdir -p data
mkdir -p data/datasets

wget -r -nH --cut-dirs=100 --reject "index.html*" --no-parent http://lavis.cs.hs-rm.de/storage/jerex/public/datasets/docred_joint/ -P ${curr_dir}/data/datasets/docred_joint