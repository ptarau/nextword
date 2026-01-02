python arrow.py train \
--data "data/$1_sents.txt" \
--out_dir "ckpts/$1/" \
--resume "ckpts/$1/ckpt_latest.pt" \
--resume_optim


python arrow.py train --data "data/$1_sents.txt" --out_dir "ckpts/$1/"