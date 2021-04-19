#!/usr/bin/env bash
curr_dir=$(pwd)

mkdir -p data
mkdir -p data/models

wget -r -nH --cut-dirs=100 --reject "index.html*" --no-parent http://lavis.cs.hs-rm.de/storage/jerex/public/models/docred/rel_classify_multi_instance -P ${curr_dir}/data/models/docred/rel_classify_multi_instance
wget -r -nH --cut-dirs=100 --reject "index.html*" --no-parent http://lavis.cs.hs-rm.de/storage/jerex/public/models/docred_joint/joint_multi_instance -P ${curr_dir}/data/models/docred_joint/joint_multi_instance