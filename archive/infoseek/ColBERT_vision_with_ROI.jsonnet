local meta = import '../meta_configs/hpc_meta_config.libsonnet';
local data = import 'infoseek_data_config.libsonnet';
local base = import 'ColBERT_base.jsonnet';
local infoseek_data_pipeline = data.infoseek_data_pipeline;
local pretrained_ckpt_path = "/home/ubuntu/additional_data/projects/ColBERT/checkpoints/colbertv2.0";

local tokenizer_config = {
  "tokenizer": {
    "TokenizerClass": "QueryTokenizer",
    "TokenizerModelVersion": pretrained_ckpt_path,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
      // "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
  },
  "decoder_tokenizer": {
    "TokenizerClass": "DocTokenizer",
    "TokenizerModelVersion": pretrained_ckpt_path,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
      // "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
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
    'process:CaptionInfoSeekDataset': {
      input_node: [
        'process:CropRegionOfInterestImages_infoseek',
      ],
      transform_name: 'CaptionImageWithBLIP2v3',
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
    'process:ShuffleTrainValidSplit': {
      transform_name: 'ShuffleData',
      input_node: 'process:MergeDataColumns',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        splits_to_process: ["train", "val"],
      },
    },
    'process:WrapOutputIntoKeys': {
      transform_name: 'WrapOutputIntoKeys',
      input_node: [
        'process:ShuffleTrainValidSplit',
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
        'process:WrapOutputIntoKeys',
        'process:ReduceWikipediaPassagesSizeForInfoSeek',
        // 'process:CaptionInfoSeekDataset',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: false,
      setup_kwargs: {
        extra_columns: ["passages", "filtered_passages", "images"],
        pass_columns: {
          "passages": "passages",
          "filtered_passages": "filtered_passages",
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
            // {
            //   dataset_type: 'InfoseekDatasetForDPR',
            //   split: 'train',
            //   use_column: 'infoseek_data_with_dpr_output',
            //   cutoff: 5000,
            // },
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
        "num_ROIs": 0,
        "input_modules": {
            "module_list":[
                {"type": "VisionInput",  "option": "from_file", "use_ROI": true},
                // {"type": "EmptyTextInput",  "option": "default"},
                // {"type": "QuestionInput",  "option": "default", 
                //         "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},
                // {"type": "TextBasedVisionInput",  "option": "caption",
                //         "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
                // {"type": "TextBasedVisionInput",  "option": "object", 
                //         "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 0,
                //         "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
                {"type": "QuestionInput",  "option": "default", 
                        "separation_tokens": {'start': 'Identify the document that is connected to answering this question:', 'end': ''}},
                {"type": "TextBasedVisionInput",  "option": "caption",
                        "separation_tokens": {'start': 'Caption:', 'end': ''}},
                {"type": "TextBasedVisionInput",  "option": "object", 
                        "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 0,
                        "separation_tokens": {'start': 'Objects:', 'sep': ',', 'end': ''}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessVisionInputProcessing", "option": "default"},
                {"type": "PostProcessColBERTQuestionInputTokenization", "option": "default"},
            ],
        },
        "decoder_input_modules": {
            "module_list":[
                // {"type": "KnowledgeInput",  "option": "default",
                //         "separation_tokens": {'start': '<BOK>', 'end': '<EOK>'}},
                {"type": "KnowledgeInput",  "option": "default",
                        "separation_tokens": {'start': '', 'end': ''}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessColBERTItemInputTokenization", "option": "default"},
            ],
        },
    },
};

std.mergePatch(base, override)
