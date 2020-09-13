# run from root directory
# bash scripts/trainer/0_train_floorplan_nn.sh

config='./config/trainer/config_nn_floor_plan.yaml'

python ./main/trainer.py \
  --config=$config
