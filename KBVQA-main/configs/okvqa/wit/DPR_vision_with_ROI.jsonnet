local base = import 'DPR.jsonnet';
local data = import 'okvqa_data_config.libsonnet';

local wit_data_paths = {
  "image_data_path": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/images",
  "train": [
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00000-of-00010.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00001-of-00010.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00002-of-00010.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00003-of-00010.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00004-of-00010.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00005-of-00010.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00006-of-00010.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00007-of-00010.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00008-of-00010.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.train.all-00009-of-00010.tsv",
  ],
  "valid": [
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.val.all-00000-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.val.all-00001-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.val.all-00002-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.val.all-00003-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.val.all-00004-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.test.all-00000-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.test.all-00001-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.test.all-00002-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.test.all-00003-of-00005.tsv",
    "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/wit_v1.test.all-00004-of-00005.tsv",
  ],
};

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
    'input:PrepareWITPassageAnnotations_withCropROIs': {
      transform_name: 'PrepareWITPassageAnnotations',
      input_node: [
        'process:IndexWITPassagesWithElasticSearch',
        'process:CropRegionOfInterestImages',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        index_name: "wit",
      },
    },
    'process:ReduceWITPassagesSize_withCropROIs': {
      transform_name: 'ReduceWITPassagesSize',
      input_node: [
        'input:PrepareWITPassageAnnotations_withCropROIs',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        index_name: "wit",
        include_concepts: [
          "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wikipedia/sg_labels.json",
        ],
      },
    },
    'process:PrepareImagesForWITDataFromPassages_full_withCropROIs': {
      input_node: "process:ReduceWITPassagesSize_withCropROIs",
      transform_name: 'PrepareImagesForWITDataFromPassages',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_paths: wit_data_paths,
        fetch_images: false,
      },
    },
    'process:ReduceWITPassagesSize_AfterImagePreparation_withCropROIs': {
      transform_name: 'ReduceWITPassagesSize',
      input_node: [
        'process:PrepareImagesForWITDataFromPassages_full_withCropROIs',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        index_name: "wit",
        include_concepts: [
          "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wikipedia/sg_labels.json",
        ],
      },
    },
    'process:ReduceWITImagesSize_withCropROIs': {
      transform_name: 'ReduceWITImagesSize',
      input_node: [
        'process:ReduceWITPassagesSize_AfterImagePreparation_withCropROIs',
        'process:PrepareImagesForWITDataFromPassages_full_withCropROIs',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
      },
    },
    'process:ExtractImageFeaturesWithViT_WIT_withCropROIs': {
      input_node: [
        'process:ReduceWITImagesSize_withCropROIs',
      ],
      transform_name: 'ExtractImageFeaturesWithViT',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        input_column: "images",
        image_processor_config: {
          "vit_image_processor": {
            "ImageProcessorClass": "AutoImageProcessor",
            "ImageProcessorModelVersion": "openai/clip-vit-base-patch32",
          }
        },
        vit_model_config: {
          "VisionModelConfigClass": "CLIPVisionConfig",
          "VisionModelClass": "CLIPVisionModel",
          "VisionModelVersion": "openai/clip-vit-base-patch32",
        },
        batch_size: 32,
      },
    },
    'process:ConcatenateImageCorpus': {
      transform_name: 'ConcatenateImageCorpus',
      input_node: [
        'process:CropRegionOfInterestImages',
        'process:ExtractImageFeaturesWithViT_WIT_withCropROIs',
        'process:ReduceWITImagesSize_withCropROIs',
        'process:ExtractImageFeaturesWithViT',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
      },
    },
    'output:PrepareDataloaders': {
      input_node: [
        "input:PrepareWITPassageAnnotations_withCropROIs",
        'process:ReduceWITPassagesSize_AfterImagePreparation_withCropROIs',
        'process:ReduceWITImagesSize_withCropROIs',
        // 'process:ExtractImageFeaturesWithViT_WIT_withCropROIs',
        // 'process:ReduceWITPassagesSize_withCropROIs',
        'process:ConcatenateImageCorpus',
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
        image_processor_config: image_processor_config,
      },
    },
  },
};

local okvqa_data_pipeline = std.mergePatch(okvqa_data, data_loader);

local override = {
    data_pipeline: okvqa_data_pipeline,
    model_config: {
        "ModelClass": "VisualDPRWithMultiModalDocs",
      "mapping_network_prefix_length": 6,
      "vision_embedding_size": 768,
      "lm_embedding_size": 768,
      "multimodal_docs": true,
      full_corpus_in_training: false,
      full_corpus_in_testing: false,
      "num_ROIs": 5,
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
              {"type": "PassageVisionInput",  "option": "from_embeddings"},
              {"type": "KnowledgeInput",  "option": "default",
                      "separation_tokens": {'start': '<BOK>', 'end': '<EOK>'}},
          ],
          "postprocess_module_list": [
              {"type": "PostProcessItemVisionInputFromEmbeddings", "option": "default"},
              {"type": "PostProcessDecoderInputTokenization", "option": "default"},
          ],
      },
    },
};

std.mergePatch(base, override)