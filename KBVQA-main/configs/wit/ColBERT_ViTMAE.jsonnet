local base = import 'ColBERT_large_vision.jsonnet';
local data = import 'wit_data_config.libsonnet';
local wit_data = data.wit_data_pipeline;

// local pretrained_ckpt_path = "/additional_data/projects/KBVQA/data/checkpoints/colbert-large-200000/colbert-200000";
local pretrained_ckpt_path = "/home/ubuntu/additional_data/projects/ColBERT/checkpoints/colbertv2.0";
// local pretrained_ckpt_path = "/data/colbert_experiments/0902-v1-bert-base-msmarco-bs=128-full[cut]/none/2023-09/02/01.27.19/checkpoints/colbert-200000";

local tokenizer_config = {
  "tokenizer": {
    "TokenizerClass": "QueryTokenizer",
    "TokenizerModelVersion": pretrained_ckpt_path,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
  },
  "decoder_tokenizer": {
    "TokenizerClass": "DocTokenizer",
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
    "ImageProcessorModelVersion": "facebook/vit-mae-base",
  },
};

local data_loader = {
  transforms: {
    'process:WrapOutputIntoKeys': {
      transform_name: 'WrapOutputIntoKeys',
      input_node: [
        'process:PrepareWITDataForRetrieval',
      ],
      regenerate: true,
      cache: false,
      setup_kwargs: {
        output_keys: ["wit_data"],
      },
    },
    'output:PrepareDataloaders': {
      input_node: [
        'process:SplitWITPassagesForLargeScaleTraining',
        'process:PrepareImagesForWITData',
        // 'process:PrepareWITDataForRetrieval',
        'process:WrapOutputIntoKeys',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: false,
      setup_kwargs: {
        extra_columns: ["passages", "filtered_passages", "images"],
        pass_columns: {
          "passages": "passages",
          "filtered_passages": "filtered_passages",
          "vqa_data": "wit_data",
          "vqa_data_with_dpr_output": "wit_data",
        },
        datasets_config: {
          train: [
            {
              dataset_type: 'WITDatasetForDPR',
              split: 'train',
              use_column: 'wit_data',
            },
          ],
          valid: [
            {
              dataset_type: 'WITDatasetForDPR',
              split: 'valid',
              use_column: 'wit_data',
            },
          ],
          test: [
            {
              dataset_type: 'WITDatasetForDPR',
              split: 'valid',
              use_column: 'wit_data',
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

local data_pipeline = std.mergePatch(wit_data, data_loader);

local override = {
    data_pipeline: data_pipeline,
    model_config: {
        "base_model": "ColBERT",
        "ModelClass": "VisualColBERTForPretraining",
        "EncoderModelVersion": pretrained_ckpt_path,
        // "EncoderModelVersion": "/home/wl356/cvnlp_rds/wl356/projects/KBVQA/experiments/OKVQA_ColBERT_full_corpus/train/saved_models/validation_temp_model"，
        "VisionModelConfigClass": "ViTMAEConfig",
        "VisionModelClass": "ViTMAEModel",
        "VisionModelVersion": "facebook/vit-mae-base",
        "pretrained": 1,
        "modules": [
            "separate_query_and_item_encoders",
        ],
        "Ks": [1, 5, 10, 20, 50, 80, 100, 500],
        "nbits": 8,
        "num_negative_samples": 1,
        "max_source_length":512,
        "max_decoder_source_length": 512,
        "mapping_network_prefix_length": 50,
        "transformer_mapping_config_base": "t5-base",
        "vision_embedding_size": 768,
        "lm_embedding_size": 128,
        "prepend_tokens": {
            "query_encoder": "",
            "item_encoder": "",
        },
        "input_modules": {
            "module_list":[
                {"type": "VisionInput",  "option": "from_file"},
                {"type": "EmptyTextInput",  "option": "default"},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessVisionInputProcessing", "option": "default"},
                {"type": "PostProcessColBERTQuestionInputTokenization", "option": "default"},
            ],
        },
        "decoder_input_modules": {
            "module_list":[
                {"type": "KnowledgeInput",  "option": "default",
                        "separation_tokens": {'start': '', 'end': ''}},
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
    train: {
        optimizer_config: {
            mapping_network_lr: 0.0001,
        },
    },
};


std.mergePatch(base, override)
