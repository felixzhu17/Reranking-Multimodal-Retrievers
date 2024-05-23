python src/main.py --config configs/Rerank/evqa_experiments/evqa_base_rerank.jsonnet \
    --mode train \
    --experiment_name "EVQA_FLMRQueryEncoder(query+doc)_BERT(1Layer)_SingleHead_BCE_ckpt1" \
    --opts train.load_model_path="experiments/EVQA_FLMRQueryEncoder(query+doc)_BERT(1Layer)_SingleHead_BCE/train/saved_models/model_step_2250.ckpt" 