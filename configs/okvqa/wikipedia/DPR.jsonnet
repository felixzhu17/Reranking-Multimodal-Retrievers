local base = import "../DPR.jsonnet";
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
local image_processor_config = {
  // "vit_image_processor": {
  //   "ImageProcessorClass": "AutoImageProcessor",
  //   "ImageProcessorModelVersion": "google/vit-base-patch16-224-in21k",
  // },
  "vit_image_processor": {
    "ImageProcessorClass": "AutoImageProcessor",
    "ImageProcessorModelVersion": "openai/clip-vit-base-patch32",
  },
};

local data_loader = {
  transforms: {
    'output:PrepareDataloaders': {
      input_node: [
        // 'process:LoadOKVQAData',
        // 'input:LoadWikipediaPassageData',
        'input:PrepareWikipediaPassageAnnotations',
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
              dataset_type: 'OKVQAWikipediaDatasetForDPR',
              split: 'train',
              use_column: 'okvqa_data_with_dpr_output',
            },
          ],
          valid: [
            {
              dataset_type: 'OKVQAWikipediaDatasetForDPR',
              split: 'test',
              use_column: 'okvqa_data_with_dpr_output',
            },
          ],
          test: [
            {
              dataset_type: 'OKVQAWikipediaDatasetForDPR',
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

local override = {
    data_pipeline: okvqa_data_pipeline,
    model_config: {
      full_corpus_in_training: false,
      full_corpus_in_testing: false,
    },
    train: {
        model_checkpoint_callback_paras: {
            monitor: 'valid/OKVQAWikipediaDatasetForDPR.test/recall_at_5',
            save_top_k: 3,
        },
    },
};

std.mergePatch(base, override)
