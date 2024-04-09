local meta = import '../meta_configs/hpc_meta_config.libsonnet';
local data = import 'visual_genome_config.libsonnet';
local vg_data = data.vg_data_pipeline;

local tokenizer_config = {
  "tokenizer": {
    "TokenizerClass": "QueryTokenizer",
    "TokenizerModelVersion": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/checkpoints/colbertv2.0",
    "SPECIAL_TOKENS":{
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
  },
  "decoder_tokenizer": {
    "TokenizerClass": "DocTokenizer",
    "TokenizerModelVersion": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/checkpoints/colbertv2.0",
    "SPECIAL_TOKENS":{
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
  },
};
local feature_extractor_config = {
};
local image_processor_config = {
  // "vit_image_processor": {
  //   "ImageProcessorClass": "AutoImageProcessor",
  //   "ImageProcessorModelVersion": "google/vit-base-patch16-224-in21k",
  // },
  "vit_image_processor": {
    "ImageProcessorClass": "AutoImageProcessor",
    "ImageProcessorModelVersion": "openai/clip-vit-base-patch32",
  },
  // "vit_image_processor": {
  //   "ImageProcessorClass": "AutoImageProcessor",
  //   "ImageProcessorModelVersion": "facebook/vit-mae-base",
  // },
};

local data_loader = {
  transforms: {
    'output:PrepareDataloaders': {
      input_node: [
        'process:PrepareVisualGenomeForRetrieval',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: false,
      setup_kwargs: {
        extra_columns: ["passages", "images"],
        pass_columns: {
          "passages": "passages",
          "vqa_data": "vg_data",
        },
        datasets_config: {
          train: [
            {
              dataset_type: 'VisualGenomeDatasetForDPR',
              split: 'train',
              use_column: 'vg_data',
            },
          ],
          valid: [
            {
              dataset_type: 'VisualGenomeDatasetForDPR',
              split: 'valid',
              use_column: 'vg_data',
            },
          ],
          test: [
            {
              dataset_type: 'VisualGenomeDatasetForDPR',
              split: 'valid',
              use_column: 'vg_data',
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

local data_pipeline = std.mergePatch(vg_data, data_loader);

{
    experiment_name: 'default_DPR',
    test_suffix: 'default_test',
    meta: meta.default_meta,
    data_pipeline: data_pipeline,
    model_config: {
        "base_model": "ColBERT",
        "ModelClass": "VisualColBERTForPretraining",
        "EncoderModelVersion": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/checkpoints/colbertv2.0",
        // "EncoderModelVersion": "/home/wl356/cvnlp_rds/wl356/projects/KBVQA/experiments/OKVQA_ColBERT_full_corpus/train/saved_models/validation_temp_model"，
        // "VisionModelConfigClass": "ViTConfig",
        // "VisionModelClass": "ViTModel",
        // "VisionModelVersion": "google/vit-base-patch16-224-in21k",
        "VisionModelConfigClass": "CLIPVisionConfig",
        "VisionModelClass": "CLIPVisionModel",
        "VisionModelVersion": "openai/clip-vit-base-patch32",
        // "VisionModelConfigClass": "ViTMAEConfig",
        // "VisionModelClass": "ViTMAEModel",
        // "VisionModelVersion": "facebook/vit-mae-base",
        "pretrained": 1,
        "modules": [
            "separate_query_and_item_encoders",
        ],
        "Ks": [100, 500, 1000, 5000],
        "nbits": 8,
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
                {"type": "VisionInput",  "option": "from_file"},
                {"type": "EmptyTextInput",  "option": "default"},
                // {"type": "QuestionInput",  "option": "default", 
                        // "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},
                // {"type": "TextBasedVisionInput",  "option": "caption",
                //         "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
                // {"type": "TextBasedVisionInput",  "option": "object", 
                //         "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
                //         "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessVisionInputProcessing", "option": "default"},
                {"type": "PostProcessColBERTQuestionInputTokenization", "option": "default"},
            ],
        },
        "decoder_input_modules": {
            "module_list":[
                {"type": "KnowledgeInput",  "option": "default",
                        "separation_tokens": {'start': '<BOK>', 'end': '<EOK>'}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessColBERTItemInputTokenization", "option": "default"},
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
        ExecutorClass: 'ColBERTVisionPretrainingExecutor',
        init_kwargs: {
            "use_data_node": "output:PrepareDataloaders",
            "use_index": "/home/wl356/cvnlp_rds/wl356/projects/KBVQA/experiments/VG_DPR_ViT_ColBERT/train/saved_models",
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
            monitor: 'valid/VisualGenomeDatasetForDPR.valid/recall_at_1000',
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
        {'name': 'compute_DPR_scores_with_pos_ids'},
        {'name': 'compute_BLEU_scores'},
    ],
}
