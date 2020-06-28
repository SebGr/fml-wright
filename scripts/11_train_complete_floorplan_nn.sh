# run from root directory
# bash scripts/11_train_complete_floorplan_nn.sh

config='./config/config_nn_complete_floorplan.yaml'

python ./main/main_neural_network.py \
  --config=$config
