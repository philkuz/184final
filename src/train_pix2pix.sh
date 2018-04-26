
python pix2pix.py \
  --mode train \
  --output_dir ../logs/pix2pix-geometry-v1/ \
  --max_epochs 200 \
  --input_dir ../data/geometry-v1/pix2pix/train \
  --which_direction BtoA
