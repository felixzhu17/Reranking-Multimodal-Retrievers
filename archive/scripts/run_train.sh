#! Full path to application executable:
application="python src/main.py"

#! Run options for the application:
options="--experiment_name \"OKVQA_RAG_VisualColBERT_BLIP2_T5xl_with_pretrained_ViT(WIT)_with_text_based_vision_K=5_LR1e-4\" \
    --config \"configs/rag/okvqa/RAG_colbert_BLIP2_with_vision.jsonnet\" \
    --accelerator auto --devices auto --strategy ddp \
    --reset --override \
    --num_sanity_val_steps 2 \
    --precision bf16 \
    --mode train \
    --opts train.trainer_paras.max_epochs=1000 \
             train.batch_size=2 \
             train.trainer_paras.val_check_interval=500 \
             valid.batch_size=4 \
             train.trainer_paras.accumulate_grad_batches=16 \
             train.early_stopping_callback_paras.patience=5 \
             train.optimizer_config.optimizer_params.lr=0.0001 \
             train.optimizer_config.retriever_lr=0.0001 \
             train.optimizer_config.scheduler=none \
             train.model_checkpoint_callback_paras.save_top_k=1 \
             model_config.num_beams=2 \
             model_config.num_ROIs=0 \
             model_config.num_knowledge_passages=5"

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

echo $CMD
eval $CMD
