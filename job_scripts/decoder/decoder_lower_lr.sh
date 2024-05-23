python src/main.py --config configs/Rerank/decoder_experiments/decoder_rerank.jsonnet \
    --mode train --reset --override \
    --experiment_name "OKVQA_Decoder_Reranker_low_lr" \
    --opts train.optimizer_config.optimizer_params.lr=1e-4