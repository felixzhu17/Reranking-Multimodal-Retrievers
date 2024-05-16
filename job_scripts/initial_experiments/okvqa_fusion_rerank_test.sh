# python src/main.py --config configs/Rerank/initial_experiments/okvqa_fusion_rerank.jsonnet --mode test --test_suffix model_step_500 --opts train.load_model_path="experiments/OKVQA_Fusion_Reranker/train/saved_models/model_step_500.ckpt"
python src/main.py --config configs/Rerank/initial_experiments/okvqa_fusion_rerank.jsonnet --mode test --experiment_name OKVQA_Fusion_Reranker_5_test --test_suffix model_step_1000 --opts model_config.fusion_multiplier=5 train.load_model_path="experiments/OKVQA_Fusion_Reranker_5/train/saved_models/model_step_1000.ckpt"