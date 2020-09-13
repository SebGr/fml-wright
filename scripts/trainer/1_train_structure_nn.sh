# run from root directory
# bash scripts/trainer/1_train_structure_nn.sh

config='./config/trainer/config_nn_structure_plan.yaml'

python ./main/trainer.py \
  --config=$config
