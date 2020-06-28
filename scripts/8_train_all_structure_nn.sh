# run from root directory
# bash scripts/8_train_all_structure_nn.sh

config='./config/config_nn_structure_plan.yaml'

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
    python ./main/main_neural_network.py \
    --config=$config \
    --category=$_cat
done
