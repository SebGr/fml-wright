# run from root directory
# bash scripts/9_create_complete_floorplan_images.sh

input_directory='./data/geodata'
output_directory='./data/images/complete_floorplan'
module='complete_floorplan'
step='generate_images'

python ./main/main_core.py \
  --module=$module \
  --step=$step \
  --input_directory=$input_directory \
  --output_directory=$output_directory
