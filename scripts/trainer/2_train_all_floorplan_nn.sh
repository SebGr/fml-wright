# run from root directory
# bash scripts/trainer/2_train_all_floorplan_nn.sh

config='./config/trainer/config_nn_floor_plan.yaml'

CATEGORIES=(
    "single_bedroom"
    "double_bedroom"
    "multiple_bedroom"
    "single_bathroom"
    "double_bathroom"
    "multiple_bathroom"
    "single_floor"
    "double_floor"
    "multiple_floor"
    "living_room"
    "balcony"
    "washing_room"
    )

for _cat in ${CATEGORIES[*]}; do
    python ./main/trainer.py \
    --config=$config \
    --category=$_cat
done
