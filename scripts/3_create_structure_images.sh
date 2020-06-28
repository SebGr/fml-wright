# run from root directory
# bash scripts/3_create_structure_images.sh

input_directory='./data/all_representation_prediction'
output_directory='./data/images/structure'
module='structure_plan'
step='generate_images'

python ./main/main_core.py \
  --module=$module \
  --step=$step \
  --input_directory=$input_directory \
  --output_directory=$output_directory
