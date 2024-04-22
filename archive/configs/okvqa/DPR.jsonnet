local meta = import '../meta_configs/hpc_meta_config.libsonnet';
local data = import 'okvqa_data_config.libsonnet';
local okvqa_data = data.okvqa_data_pipeline;

local tokenizer_config = {
  "tokenizer": {
    "TokenizerClass": "DPRQuestionEncoderTokenizer",
    "TokenizerModelVersion": "facebook/dpr-question_encoder-single-nq-base",
    "SPECIAL_TOKENS":{
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
  },
  "decoder_tokenizer": {
    "TokenizerClass": "DPRContextEncoderTokenizer",
    "TokenizerModelVersion": "facebook/dpr-ctx_encoder-single-nq-base",
    "SPECIAL_TOKENS":{
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
  },
};
local feature_extractor_config = {
};

local data_loader = {
  transforms: {
    'output:PrepareDataloaders': {
      input_node: [
        'process:LoadOKVQAData',
        'input:LoadGoogleSearchPassageData',
        'input:LoadGoogleSearchAnnotations',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: false,
      setup_kwargs: {
        extra_columns: ["passages"],
        pass_columns: {
          "passages": "passages",
          "vqa_data_with_dpr_output": "okvqa_data_with_dpr_output",
          // "vqa_data": "okvqa_data",
        },
        datasets_config: {
          train: [
            {
              dataset_type: 'OKVQADatasetForDPR',
              split: 'train',
              use_column: 'okvqa_data_with_dpr_output',
            },
          ],
          valid: [
            {
              dataset_type: 'OKVQADatasetForDPR',
              split: 'test',
              use_column: 'okvqa_data_with_dpr_output',
            },
          ],
          test: [
            {
              dataset_type: 'OKVQADatasetForDPR',
              split: 'test',
              use_column: 'okvqa_data_with_dpr_output',
            },
          ],
        },
        tokenizer_config: tokenizer_config,
        feature_extractor_config: feature_extractor_config,
      },
    },
  },
};

local okvqa_data_pipeline = std.mergePatch(okvqa_data, data_loader);

{
    experiment_name: 'default_DPR',
    test_suffix: 'default_test',
    meta: meta.default_meta,
    data_pipeline: okvqa_data_pipeline,
    model_config: {
        "base_model": "DPR",
        "ModelClass": "RetrieverDPR",
        "QueryEncoderModelClass": "DPRQuestionEncoder",
        "QueryEncoderConfigClass": "DPRConfig",
        "QueryEncoderModelVersion": "facebook/dpr-question_encoder-single-nq-base",
        "ItemEncoderModelClass": "DPRContextEncoder",
        "ItemEncoderConfigClass": "DPRConfig",
        "ItemEncoderModelVersion": "facebook/dpr-ctx_encoder-single-nq-base",
        "pretrained": 1,
        "modules": [
            "separate_query_and_item_encoders",
        ],
        "Ks": [1, 5, 10, 20, 50, 80, 100],
        "num_negative_samples": 1,
        "max_source_length":512,
        "max_decoder_source_length": 512,
        "full_corpus_in_training": true,
        "full_corpus_in_testing": true,
        "prepend_tokens": {
            "query_encoder": "",
            "item_encoder": "",
        },
        "input_modules": {
            "module_list":[
                {"type": "QuestionInput",  "option": "default", 
                        "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},
                {"type": "TextBasedVisionInput",  "option": "caption",
                        "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
                {"type": "TextBasedVisionInput",  "option": "object", 
                        "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
                        "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessInputTokenization", "option": "default"},
            ],
        },
        "decoder_input_modules": {
            "module_list":[
                {"type": "KnowledgeInput",  "option": "default",
                        "separation_tokens": {'start': '<BOK>', 'end': '<EOK>'}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessDecoderInputTokenization", "option": "default"},
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
        ExecutorClass: 'DPRExecutor',
        init_kwargs: {
            "use_data_node": "output:PrepareDataloaders",
        },
    },
    train: {
        batch_size: 8,
        num_dataloader_workers: 0,
        trainer_paras: {
            max_epochs: 100,
            accumulate_grad_batches: 4,
            check_val_every_n_epoch: null,
            val_check_interval: 10,
            log_every_n_steps: 10,
        },
        model_checkpoint_callback_paras: {
            monitor: 'valid/OKVQADatasetForDPR.test/recall_at_5',
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
        batch_size: 64,
        num_dataloader_workers: 0,
    },
    test: {
        checkpoint_name: "",
        load_model_path: "",
        load_best_model: false,
        trainer_paras: {},
        batch_size: 64,
        num_dataloader_workers: 0,
    },
    eval: {
        'eval_op_name': 'Your eval op name'
    },
    "metrics": [
        {'name': 'compute_DPR_scores'},
    ],
}
