local meta = import '../meta_configs/hpc_meta_config.libsonnet';
local data = import 'infoseek_data_config.libsonnet';
local base = import 'ColBERT_vision_preload_features.jsonnet';
local infoseek_data_pipeline = data.infoseek_data_pipeline;

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
    'process:CropRegionOfInterestImages_infoseek': {
      input_node: [
        'process:LoadInfoSeekData',
      ],
      transform_name: 'CropRegionOfInterestImagesForInfoSeek',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        max_objects: 0,
      },
    },
    // 'process:ExtractImageFeaturesWithViT_infoseek': {
    //   input_node: [
    //     'process:CropRegionOfInterestImages_infoseek',
    //   ],
    //   transform_name: 'ExtractImageFeaturesWithViTv2',
    //   regenerate: false,
    //   cache: true,
    //   setup_kwargs: {
    //     input_column: "images",
    //     cache_folder: "/home/wl356/big_picture_rds/wl356/infoseek/extracted_image_features_safetensors",
    //     image_processor_config: image_processor_config,
    //     vit_model_config: {
    //       "VisionModelConfigClass": "CLIPVisionConfig",
    //       "VisionModelClass": "CLIPVisionModel",
    //       "VisionModelVersion": "openai/clip-vit-base-patch32",
    //     },
    //     batch_size: 32,
    //   },
    // },
    'process:ExtractImageFeaturesWithViT_infoseek': {
      input_node: [
        'process:CropRegionOfInterestImages_infoseek',
      ],
      transform_name: 'ExtractImageFeaturesWithViTv3',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        input_column: "images",
        index_name: "encoded_image_features",
        image_processor_config: image_processor_config,
        vit_model_config: {
          "VisionModelConfigClass": "CLIPVisionConfig",
          "VisionModelClass": "CLIPVisionModel",
          "VisionModelVersion": "openai/clip-vit-base-patch32",
        },
        batch_size: 32,
      },
    },
    'process:CaptionInfoSeekDataset': {
      input_node: [
        'process:CropRegionOfInterestImages_infoseek',
      ],
      transform_name: 'CaptionImageWithBLIP2v2',
      setup_kwargs: {
        pretrained_model_name: 'Salesforce/blip2-flan-t5-xl',
        save_to_disk_column: "image_path",
        splits_to_process: ["train", "val"],
        index_name: "image_captions",
      },
      cache: true,
      regenerate: false,
    },
    'process:MergeDataColumns': {
      transform_name: 'MergeDataColumns',
      input_node: [
        'process:CaptionInfoSeekDataset',
        'process:PrepareWikipediaPassageAnnotationsForInfoSeek',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        merge_on: 1,
        merge_from: 0,
        splits_to_process: ["train", "val"],
      },
    },
    'process:WrapOutputIntoKeys': {
      transform_name: 'WrapOutputIntoKeys',
      input_node: [
        'process:MergeDataColumns',
      ],
      regenerate: true,
      cache: false,
      setup_kwargs: {
        output_keys: ["infoseek_data_with_dpr_output", ],
      },
    },
    'output:PrepareDataloaders': {
      input_node: [
        'process:CropRegionOfInterestImages_infoseek',
        'input:LoadWikipediaPassageData',
        'process:ExtractImageFeaturesWithViT_infoseek',
        'process:WrapOutputIntoKeys',
        'process:ReduceWikipediaPassagesSizeForInfoSeek',
        // 'process:CaptionInfoSeekDataset',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: false,
      setup_kwargs: {
        extra_columns: ["passages", "filtered_passages", "images", "image_dataset_with_embeddings"],
        pass_columns: {
          "passages": "passages",
          "vqa_data_with_dpr_output": "infoseek_data_with_dpr_output",
          "vqa_data": "infoseek_data_with_dpr_output",
        },
        datasets_config: {
          train: [
            {
              dataset_type: 'InfoseekDatasetForDPR',
              split: 'train',
              use_column: 'infoseek_data_with_dpr_output',
            },
          ],
          valid: [
            {
              dataset_type: 'InfoseekDatasetForDPR',
              split: 'val',
              use_column: 'infoseek_data_with_dpr_output',
            },
          ],
          test: [
            {
              dataset_type: 'InfoseekDatasetForDPR',
              split: 'val',
              use_column: 'infoseek_data_with_dpr_output',
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

local infoseek_data_pipeline_merged = std.mergePatch(infoseek_data_pipeline, data_loader);

local override = {
    data_pipeline: infoseek_data_pipeline_merged,
    model_config: {
        "num_ROIs": 9,
        "input_modules": {
            "module_list":[
                {"type": "VisionInput",  "option": "from_embeddings", "use_ROI": true},
                // {"type": "EmptyTextInput",  "option": "default"},
                {"type": "QuestionInput",  "option": "default", 
                        "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},
                {"type": "TextBasedVisionInput",  "option": "caption",
                        "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
                {"type": "TextBasedVisionInput",  "option": "object", 
                        "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 0,
                        "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessVisionInputFromEmbeddings", "option": "default"},
                {"type": "PostProcessColBERTQuestionInputTokenization", "option": "default"},
            ],
        },
    },
};

std.mergePatch(base, override)
