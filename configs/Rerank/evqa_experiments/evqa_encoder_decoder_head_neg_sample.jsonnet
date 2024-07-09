local meta = import '../../meta_configs/hpc_meta_config.libsonnet';
local data = import '../../data/evqa_data.libsonnet';
local merge_data = data.merge_data_pipeline;

local pretrained_ckpt_path = "LinWeizheDragon/PreFLMR_ViT-B";
local reranker_pretrained_ckpt_path = "LinWeizheDragon/PreFLMR_ViT-B";
local image_processor_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k";

local tokenizer_config = {
  "tokenizer": {
    "TokenizerClass": "FLMRQueryEncoderTokenizer",
    "TokenizerModelVersion": pretrained_ckpt_path,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
  },
  "decoder_tokenizer": {
    "TokenizerClass": "FLMRContextEncoderTokenizer",
    "TokenizerModelVersion": pretrained_ckpt_path,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
  },
};
local feature_extractor_config = {
};
local image_processor_config = {
  "vit_image_processor": {
    "ImageProcessorClass": "AutoImageProcessor",
    "ImageProcessorModelVersion": image_processor_name,
  },
};

local index_files = {
  "index_path": "",
  "embedding_path": "",
  "static_results": [
    "/home/fz288/rds/hpc-work/PreFLMR/search_index/EVQA/PreFLMR-B/_test_EVQADatasetForDPR.test_predictions_rank_0.pkl"
  ],
};

local data_loader = {
  transforms: {
    'output:PrepareDataloaders': {
      input_node: [
        'process:ConcatenatePassageDatasets',
        'process:WrapOutputIntoKeys',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: true,
      setup_kwargs: {
        extra_columns: {
          "passages": "train_passages",
          "valid_passages": "valid_passages",
        },
        pass_columns: {
          "train_passages": "train_passages",
          "valid_passages": "valid_passages",
          "test_passages": "test_passages",
          "vqa_data_with_dpr_output": "evqa_data",
        },
        datasets_config: {
          train: [
            {
              dataset_type: 'EVQADatasetForDPR',
              split: 'train',
              use_column: 'evqa_data',
            },
          ],
          valid: [
            {
              dataset_type: 'EVQADatasetForDPR',
              split: 'test',
              use_column: 'evqa_data',
            },
          ],
          test: [
            {
              dataset_type: 'EVQADatasetForDPR',
              split: 'test',
              use_column: 'evqa_data',
            },
          ],
        },
        tokenizer_config: tokenizer_config,
        feature_extractor_config: feature_extractor_config,
        image_processor_config: image_processor_config,
      },
    },
  },
};

local validation_indexing_source = ["evqa_passages"];

local data_pipeline = std.mergePatch(merge_data, data_loader);

{
    experiment_name: 'EVQA_Decoder_Reranker',
    test_suffix: 'default_test',
    meta: meta.default_meta,
    data_pipeline: data_pipeline,
    model_config: {
        "retriever_config":{
          "base_model": "FLMR",
          "ConfigClass": "FLMRConfig",
          "ModelClass": "FLMRModelForRetrieval",
          "ModelVersion": pretrained_ckpt_path,
        },
        "reranker_config":{
          "RerankerClass":"DecoderHeadRerankModel",
          "GeneratorModelClass": "Blip2ForConditionalGeneration", 
          "GeneratorConfigClass": "Blip2Config",
          "GeneratorModelVersion": "Salesforce/blip2-flan-t5-xl",
          "max_query_length": 32,
          "max_decoder_source_length": 512,
          "loss_fn": "negative_sampling"
        },
        "Ks": [5, 10, 20, 50, 100],
        "num_negative_samples": 4,
        "max_source_length": 32,
        "max_decoder_source_length": 512,
        "fusion_multiplier": 1,
        "pretrained": 1,
        "modules": [
            "separate_query_and_item_encoders",
            // "full_validation",
            "decoder_reranker",
            "split_testing_batch"
        ],
        "index_files": index_files,
        "nbits": 8,
        "docs_to_rerank": 100,
        "prepend_tokens": {
            "query_encoder": "",
            "item_encoder": "",
        },
        "input_modules": {
            "module_list":[
                {"type": "VisionInput",  "option": "from_file"},
                {"type": "InstructionInput",  "option": "default",
                "separation_tokens": {'start': '', 'end': ''}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessVisionInputProcessing", "option": "default"},
                {"type": "PostProcessFLMRQuestionInputTokenization", "option": "default"},
            ],
        },
        "decoder_input_modules": {
            "module_list":[
                {"type": "KnowledgeInput",  "option": "default",
                        "separation_tokens": {'start': '', 'end': ''}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessFLMRItemInputTokenization", "option": "default"},
            ],
        },
        "output_modules": {
            "module_list":[
                {"type": "SimilarityOutput", "option": "default"},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessConcatenateLabels", "option": "default"},
            ],
        },
    },
    executor: {
        ExecutorClass: 'RerankerBaseExecutor',
        init_kwargs: {
            "use_data_node": "output:PrepareDataloaders",
            "index_splits": ['train', 'valid', 'test'],
            "validation_indexing_source": validation_indexing_source,
        },
    },
    train: {
        batch_size: 2,
        num_dataloader_workers: 4,
        trainer_paras: {
            accelerator: 'auto',
            devices: 'auto',
            strategy: 'ddp_find_unused_parameters_true',
            precision: 'bf16',
            max_epochs: -1,
            accumulate_grad_batches: 16,
            check_val_every_n_epoch: null,
            val_check_interval: 1000,
            log_every_n_steps: 10,
            // limit_train_batches: 2,
            limit_val_batches: 3,
        },
        model_checkpoint_callback_paras: {
            monitor: 'valid/EVQADatasetForDPR.test/loss',
            save_top_k: 5,
            mode: "min",
            filename: 'model_step_{step}',
            save_last: true,
            verbose: true,
            auto_insert_metric_name: false,
            save_on_train_epoch_end: false,
        },
        early_stopping_callback_paras: {
            patience: 3,
            verbose: true,
            mode: "min",
        },
        optimizer_config: {
            optimizer_name: "AdamW",
            optimizer_params: {
                lr: 1e-4,
                eps: 1e-08,
            },
            scheduler: "none",
            scheduler_params: {
                num_warmup_steps: 0,
            },
        },
    },
    valid: {
        batch_size: 16,
        num_dataloader_workers: 0,
    },
    test: {
        checkpoint_name: "",
        load_model_path: "",
        load_best_model: false,
        trainer_paras: {
            accelerator: 'auto',
            devices: 'auto',
            strategy: 'ddp_find_unused_parameters_true',
            precision: 'bf16',
            limit_test_batches: 65,
        },
        batch_size: 16,
        num_dataloader_workers: 0,

    },
    eval: {
        'eval_op_name': 'Your eval op name'
    },
    "metrics": [
        {'name': 'compute_rerank_DPR_scores'},
        {'name': 'compute_rerank_DPR_scores_with_pos_ids'},
    ],
}
