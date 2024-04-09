local base = import "../ColBERT.jsonnet";
local data = import 'okvqa_data_config.libsonnet';
local okvqa_data = data.okvqa_data_pipeline;

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
};


local data_loader = {
  transforms: {
    'output:PrepareDataloaders': {
      input_node: [
        "input:PrepareWITPassageAnnotations",
        'process:ReduceWITImagesSize',
        'process:ReduceWITPassagesSize_AfterImagePreparation',
        'process:ExtractImageFeaturesWithViT_WIT',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: false,
      setup_kwargs: {
        extra_columns: ["passages", "images", "image_dataset_with_embeddings", "imgId2path"],
        pass_columns: {
          "passages": "passages",
          "vqa_data_with_dpr_output": "okvqa_data_with_dpr_output",
          // "vqa_data": "okvqa_data",
        },
        datasets_config: {
          train: [
            {
              dataset_type: 'OKVQAWITDatasetForDPR',
              split: 'train',
              use_column: 'okvqa_data_with_dpr_output',
            },
          ],
          valid: [
            {
              dataset_type: 'OKVQAWITDatasetForDPR',
              split: 'test',
              use_column: 'okvqa_data_with_dpr_output',
            },
          ],
          test: [
            {
              dataset_type: 'OKVQAWITDatasetForDPR',
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
      "ModelClass": "ColBERTWithMultimodalDocs",
      "mapping_network_prefix_length": 32,
      "vision_embedding_size": 768,
      "lm_embedding_size": 128,
      "multimodal_docs": true,
      "decoder_input_modules": {
          "module_list":[
              {"type": "PassageVisionInput",  "option": "from_embeddings"},
              {"type": "KnowledgeInput",  "option": "default",
                      "separation_tokens": {'start': '<BOK>', 'end': '<EOK>'}},
          ],
          "postprocess_module_list": [
              {"type": "PostProcessItemVisionInputFromEmbeddings", "option": "default"},
              {"type": "PostProcessColBERTItemInputTokenization", "option": "default"},
          ],
      },
    },
    train: {
        model_checkpoint_callback_paras: {
            monitor: 'valid/OKVQAWITDatasetForDPR.test/recall_at_5',
            save_top_k: 1,
        },
    },
};

std.mergePatch(base, override)
