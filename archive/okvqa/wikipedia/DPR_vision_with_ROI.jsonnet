local base = import 'DPR.jsonnet';
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
    'process:LoadOKVQAData': {
      regenerate: false,
      setup_kwargs: {
        add_images: false,
      },
    },
    'process:CropRegionOfInterestImages': {
      input_node: [
        'process:LoadOKVQAData',
      ],
      transform_name: 'CropRegionOfInterestImages',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        max_objects: 9,
      },
    },
    'process:ExtractImageFeaturesWithViT': {
      input_node: [
        'process:CropRegionOfInterestImages',
      ],
      transform_name: 'ExtractImageFeaturesWithViT',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        input_column: "images",
        image_processor_config: image_processor_config,
        vit_model_config: {
          "VisionModelConfigClass": "CLIPVisionConfig",
          "VisionModelClass": "CLIPVisionModel",
          "VisionModelVersion": "openai/clip-vit-base-patch32",
        },
        batch_size: 32,
        max_objects: 9,
      },
    },
    'input:PrepareWikipediaPassageAnnotations_withCropROIs': {
      transform_name: 'PrepareWikipediaPassageAnnotations',
      input_node: [
        'process:IndexPassagesWithElasticSearch',
        'process:CropRegionOfInterestImages',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        index_name: "wikipedia",
      },
    },
    'process:ReduceWikipediaPassagesSize_withCropROIs': {
      transform_name: 'ReduceWikipediaPassagesSize',
      input_node: [
        'input:PrepareWikipediaPassageAnnotations_withCropROIs',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        index_name: "wikipedia",
        include_concepts: [
          "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wikipedia/sg_labels.json",
        ],
      },
    },
    'output:PrepareDataloaders': {
      input_node: [
        'process:ReduceWikipediaPassagesSize_withCropROIs',
        'process:CropRegionOfInterestImages',
        'process:ExtractImageFeaturesWithViT',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: false,
      setup_kwargs: {
        extra_columns: ["passages", "images", "image_dataset_with_embeddings"],
        pass_columns: {
          "passages": "passages",
          "vqa_data_with_dpr_output": "okvqa_data_with_dpr_output",
          "vqa_data": "okvqa_data",
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
        image_processor_config: image_processor_config,
      },
    },
  },
};

local okvqa_data_pipeline = std.mergePatch(okvqa_data, data_loader);

local override = {
    data_pipeline: okvqa_data_pipeline,
    model_config: {
        "base_model": "DPR",
        "ModelClass": "VisualDPRForRetrieval",
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
        "num_ROIs": 0,
        "max_source_length":512,
        "max_decoder_source_length": 512,
        "mapping_network_prefix_length": 6,
        "vision_embedding_size": 768,
        "lm_embedding_size": 768,
        "full_corpus_in_training": false,
        "full_corpus_in_testing": false,
        "prepend_tokens": {
            "query_encoder": "",
            "item_encoder": "",
        },
        "input_modules": {
            "module_list":[
                {"type": "VisionInput",  "option": "from_embeddings", "use_ROI": true},
                {"type": "QuestionInput",  "option": "default", 
                        "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},
                {"type": "TextBasedVisionInput",  "option": "caption",
                        "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
                {"type": "TextBasedVisionInput",  "option": "object", 
                        "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
                        "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessVisionInputFromEmbeddings", "option": "default"},
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
};

std.mergePatch(base, override)