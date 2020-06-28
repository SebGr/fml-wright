# run from root directory
# bash scripts/6_train_structure_nn.sh

config='./config/config_nn_structure_plan.yaml'

python ./main/main_neural_network.py \
  --config=$config
