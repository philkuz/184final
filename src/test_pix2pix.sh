
python pix2pix.py \
  --mode test \
  --output_dir ../logs/pix2pix-geometry-v3/ \
  --checkpoint ../logs/pix2pix-geometry-v3/ \
  --max_epochs 200 \
  --input_dir ../data/geometry-v3/pix2pix/train \
  --which_direction BtoA \
  --use_cue \
  --img_masks \
  --input_mask_dir ../data/geometry-v3/masks \
