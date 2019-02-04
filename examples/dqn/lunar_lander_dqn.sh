DATE=$(date +"%Y%m%d%H%M%S")
MODEL_DIR="models/lunar_lander_dqn/$DATE"
SEED=64

python examples/dqn/lunar_lander_dqn.py train \
    --model_dir "$MODEL_DIR" \
    --seed "$SEED" \
    --visualize_eval