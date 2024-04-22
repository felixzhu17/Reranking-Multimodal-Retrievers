local meta = import '../../meta_configs/hpc_meta_config.libsonnet';
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
    "TokenizerClass": "Blip2Processor",
    "TokenizerModelVersion": "Salesforce/blip2-flan-t5-xl",
    // "TokenizerModelVersion": "Salesforce/blip2-opt-2.7b",
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
      // "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
  },
};
local feature_extractor_config = {
};

local index_files = {
  "index_passages_path": "$OKVQA_DPR_with_text_based_vision/test/generate_index/step_6020/dataset/",
  "index_path": "$OKVQA_DPR_with_text_based_vision/test/generate_index/step_6020/dataset_hnsw_index.faiss",
};
local data_loader = {
  transforms: {
    'output:PrepareDataloaders': {
      input_node: [
        'process:LoadOKVQAData',
        'input:LoadGoogleSearchPassageData',
        // 'input:LoadGoogleSearchAnnotations',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: false,
      setup_kwargs: {
        extra_columns: ["passages"],
        pass_columns: {
          "passages": "passages",
          "vqa_data": "okvqa_data",
        },
        datasets_config: {
          train: [
            {
              dataset_type: 'OKVQADataset',
              split: 'train',
              use_column: 'okvqa_data',
            },
          ],
          valid: [
            {
              dataset_type: 'OKVQADataset',
              split: 'test',
              use_column: 'okvqa_data',
            },
          ],
          test: [
            {
              dataset_type: 'OKVQADataset',
              split: 'test',
              use_column: 'okvqa_data',
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
    experiment_name: 'default_RAG',
    test_suffix: 'default_test',
    meta: meta.default_meta,
    data_pipeline: okvqa_data_pipeline,
    model_config: {
        "base_model": "RAG",
        "ModelClass": "RagModelForBlip",
        "TokenizerClass": tokenizer_config.tokenizer.TokenizerClass,  // question encoder tokenizer
        "TokenizerModelVersion": tokenizer_config.tokenizer.TokenizerModelVersion, 
        "DecoderTokenizerClass": tokenizer_config.decoder_tokenizer.TokenizerClass,  // generator tokenizer
        "DecoderTokenizerModelVersion": tokenizer_config.decoder_tokenizer.TokenizerModelVersion, // generator tokenizer version
        
        "QueryEncoderModelClass": "DPRQuestionEncoder", // question encoder
        "QueryEncoderModelVersion": "$OKVQA_DPR_with_text_based_vision/train/saved_models/step_6020/query_encoder",
        "QueryEncoderConfigClass": "DPRConfig", // question encoder

        "GeneratorModelClass": "Blip2ForConditionalGeneration", // answer generator
        "GeneratorConfigClass": "Blip2Config",
        "GeneratorModelVersion": "Salesforce/blip2-flan-t5-xl",
        // "GeneratorModelVersion": "Salesforce/blip2-opt-2.7b",
        "pretrained": 1,
        "RAVQA_loss_type": "Approach6",
        "loss_ratio": {
            "nll_loss": 1,
            "rag_loss": 0,
            "additional_loss": 0,
        },
        "modules": [
            "freeze_question_encoder",
            "force_existence",
        ],
        "Ks": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 80, 100],
        "num_beams": 5,
        "max_source_length":512,
        "max_decoder_source_length": 512,
        'max_target_length':10,
        'num_knowledge_passages': 5,
        "index_files": index_files,
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
                {"type": "VisionInput", "option": "from_file"},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessInputTokenization", "option": "default"},
                {"type": "PostProcessBlip2VisionInputProcessing", "option": "default"},
            ],
        },
        "decoder_input_modules": {
            "module_list":[
                // {"type": "KnowledgeInput",  "option": "default",
                //         "separation_tokens": {'start': '<BOK>', 'end': '<EOK>'}},
            ],
            "postprocess_module_list": [
                // {"type": "PostProcessColBERTItemInputTokenization", "option": "default"},
            ],
        },
        "output_modules": {
            "module_list":[
                {"type": "GenerationOutput", "option": "default"},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessBlipOutputTokenization", "option": "default"},
            ],
        },
    },
    executor: {
        ExecutorClass: 'RagBlipExecutor',
        init_kwargs: {
            "use_data_node": "output:PrepareDataloaders",
        },
    },
    train: {
        batch_size: 8,
        num_dataloader_workers: 0,
        trainer_paras: {
            max_epochs: 10,
            accumulate_grad_batches: 4,
            check_val_every_n_epoch: null,
            val_check_interval: 10,
            log_every_n_steps: 10,
        },
        model_checkpoint_callback_paras: {
            monitor: 'valid/OKVQADataset.test/accuracy_overall',
            save_top_k: 1,
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
                lr: 0.0006,
                eps: 1e-08,
            },
            retriever_lr: 0.0001,
            scheduler: "linear",
            scheduler_params: {
                num_warmup_steps: 0,
            },
        },
        weight_decay: 0.05,
        label_smoothing_factor: 0.1,
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
        {'name': 'compute_exact_match'},
        {'name': 'compute_retrieval_metrics'},
        {'name': 'compute_okvqa_scores'},
    ],
}
