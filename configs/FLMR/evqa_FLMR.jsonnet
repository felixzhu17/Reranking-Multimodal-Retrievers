local meta = import '../meta_configs/hpc_meta_config.libsonnet';
local data = import '../data/evqa_data.libsonnet';
local merge_data = data.merge_data_pipeline;

local pretrained_ckpt_path = "LinWeizheDragon/PreFLMR_ViT-G";
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
    experiment_name: 'default_DPR',
    test_suffix: 'default_test',
    meta: meta.default_meta,
    data_pipeline: data_pipeline,
    model_config: {
        "base_model": "FLMR",
        "ConfigClass": "FLMRConfig",
        "ModelClass": "FLMRModelForRetrieval",
        "ModelVersion": pretrained_ckpt_path,
        "pretrained": 1,
        "modules": [
            "separate_query_and_item_encoders",
        ],
        "Ks": [5, 10, 20, 50, 80, 100, 500],
        "nbits": 8,
        "num_negative_samples": 4,
        "max_source_length": 32,
        "max_decoder_source_length": 512,
        "query_embedding_size": 4096,
        "adapter_embedding_size": 128,
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
        ExecutorClass: 'FLMRBaseExecutor',
        init_kwargs: {
            "use_data_node": "output:PrepareDataloaders",
            "index_splits": ['valid', 'test'],
            "validation_indexing_source": validation_indexing_source,
        },
    },
    train: {
        batch_size: 8,
        num_dataloader_workers: 4,
        trainer_paras: {
            accelerator: 'auto',
            devices: 'auto',
            strategy: 'ddp_find_unused_parameters_true',
            precision: 'bf16',
            max_epochs: -1,
            accumulate_grad_batches: 8,
            check_val_every_n_epoch: null,
            val_check_interval: 250,
            log_every_n_steps: 10,
        },
        model_checkpoint_callback_paras: {
            monitor: 'valid/EVQADatasetForDPR.test/recall_at_5',
            save_top_k: 3,
            mode: "max",
            filename: 'model_step_{step}',
            save_last: true,
            verbose: true,
            auto_insert_metric_name: false,
            save_on_train_epoch_end: false,
        },
        early_stopping_callback_paras: {
            patience: 3,
            verbose: true,
            mode: "max",
        },
        optimizer_config: {
            optimizer_name: "AdamW",
            optimizer_params: {
                lr: 0.00001,
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
        },
        batch_size: 16,
        num_dataloader_workers: 0,

    },
    eval: {
        'eval_op_name': 'Your eval op name'
    },
    "metrics": [
        {'name': 'compute_DPR_scores'},
        {'name': 'compute_DPR_scores_with_pos_ids'},
    ],
}
