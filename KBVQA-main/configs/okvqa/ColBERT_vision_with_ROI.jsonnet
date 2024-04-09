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
    'input:LoadGoogleSearchAnnotations': {
      input_node: [
        'input:LoadGoogleSearchPassageData',
        'process:CropRegionOfInterestImages',
      ],
      regenerate: false,
    },
    'output:PrepareDataloaders': {
      input_node: [
        // 'process:LoadOKVQAData',
        'input:LoadGoogleSearchPassageData',
        'input:LoadGoogleSearchAnnotations',
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
              split: 'train',
              use_column: 'okvqa_data_with_dpr_output',
            },
            // {
            //   dataset_type: 'OKVQADatasetForDPR',
            //   split: 'test',
            //   use_column: 'okvqa_data_with_dpr_output',
            // },
          ],
        },
        tokenizer_config: tokenizer_config,
        feature_extractor_config: feature_extractor_config,
        image_processor_config: image_processor_config,
      },
    },
  },
};


local VAE_patch = {
  transforms: {
    'process:ExtractImageFeaturesWithVAE': {
      input_node: [
        'process:CropRegionOfInterestImages',
      ],
      transform_name: 'ExtractImageFeaturesWithVAE',
      regenerate: true,
      cache: true,
      setup_kwargs: {
        input_column: "images",
        vae_model_config: {
          "VisionModelClass": "AutoencoderKL",
          "VisionModelVersion": "stabilityai/sd-vae-ft-mse",
        },
        batch_size: 32,
        max_objects: 9,
      },
    },
    'output:PrepareDataloaders': {
      input_node: [
        'input:LoadGoogleSearchPassageData',
        'input:LoadGoogleSearchAnnotations',
        'process:CropRegionOfInterestImages',
        'process:ExtractImageFeaturesWithVAE',
      ],
    },
  },
};

local okvqa_data_pipeline = std.mergePatch(okvqa_data, data_loader);
local okvqa_data_pipeline_with_VAE_features = std.mergePatch(okvqa_data_pipeline, VAE_patch); 




local override = {
    data_pipeline: okvqa_data_pipeline_with_VAE_features,
    model_config: {
      "vision_embedding_size": 16384, #768,
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
