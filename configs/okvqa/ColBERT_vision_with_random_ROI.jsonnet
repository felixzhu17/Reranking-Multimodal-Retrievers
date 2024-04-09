local meta = import '../meta_configs/hpc_meta_config.libsonnet';
local data = import 'okvqa_data_config.libsonnet';
local base = import 'ColBERT_vision_preload_features.jsonnet';

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
    'process:LoadOKVQAData': {
      regenerate: false,
      setup_kwargs: {
        add_images: false,
      },
    },
    'process:CropRandomRegionOfInterestImages': {
      input_node: [
        'process:LoadOKVQAData',
      ],
      transform_name: 'CropRandomRegionOfInterestImages',
      regenerate: true,
      cache: true,
      setup_kwargs: {
        max_objects: 4,
        even_split: true,
      },
    },
    'process:ExtractImageFeaturesWithViT_Cropped': {
      input_node: [
        'process:CropRandomRegionOfInterestImages',
      ],
      transform_name: 'ExtractImageFeaturesWithViT',
      regenerate: true,
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
      },
    },
    'input:LoadGoogleSearchAnnotations': {
      input_node: [
        'input:LoadGoogleSearchPassageData',
        'process:CropRandomRegionOfInterestImages',
      ],
      regenerate: false,
    },
    'output:PrepareDataloaders': {
      input_node: [
        // 'process:LoadOKVQAData',
        'input:LoadGoogleSearchPassageData',
        'input:LoadGoogleSearchAnnotations',
        'process:CropRandomRegionOfInterestImages',
        'process:ExtractImageFeaturesWithViT_Cropped',
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
        image_processor_config: image_processor_config,
      },
    },
  },
};

local okvqa_data_pipeline = std.mergePatch(okvqa_data, data_loader);

local override = {
    data_pipeline: okvqa_data_pipeline,
    model_config: {
        "input_modules": {
            "module_list":[
                {"type": "VisionInput",  "option": "from_embeddings", "num_ROIs": 4},
                // {"type": "EmptyTextInput",  "option": "default"},
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
};

std.mergePatch(base, override)
