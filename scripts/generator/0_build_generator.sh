# run from root directory
# bash scripts/generator/0_build_generator.sh

# Due to the nature of the keras models, the predictor can't be stored unfortunately.
# This is an example of how to load the model using the main entrypoint.

config='./config/generator/complete_floorplan.yaml'

python ./main/generator.py \
  --config=$config
