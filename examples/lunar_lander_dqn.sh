DATE=$(date +"%Y%m%d%H%M%S")
MODEL_DIR="models/lunar_lander_dqn/$DATE"

python examples/lunar_lander_dqn.py train \
    --model_dir "$MODEL_DIR" \
    --visualize_eval