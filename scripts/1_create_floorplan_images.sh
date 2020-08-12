# run from root directory
# bash scripts/1_create_floorplan_images.sh

input_directory='./data/geodata'
output_directory='./data/images/floorplan'
module='floor_plan'
step='generate_images'

python ./main/main_core.py \
  --module=$module \
  --step=$step \
  --input_directory=$input_directory \
  --output_directory=$output_directory
