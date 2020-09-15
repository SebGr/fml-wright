# run from root directory
# bash scripts/trainer/4_train_complete_floorplan_nn.sh

config='./config/trainer/config_nn_complete_floorplan.yaml'

python ./main/trainer.py \
  --config=$config
