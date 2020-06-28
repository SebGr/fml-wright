# run from root directory
# bash scripts/5_train_floorplan_nn.sh

config='./config/config_nn_floor_plan.yaml'

python ./main/main_neural_network.py \
  --config=$config
