local vqa_data_path = {
  question_files: {
    train: '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/OpenEnded_mscoco_train2014_questions.json',
    test: '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/OpenEnded_mscoco_val2014_questions.json',
  },
  annotation_files: {
    train: '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/mscoco_train2014_annotations.json',
    test: '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/mscoco_val2014_annotations.json',
  },
};
local image_data_path = {
  train: '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/train2014',
  test: '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/val2014',
};
local caption_features = {
  train: '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/pre-extracted_features/captions/train_predictions.json',
  valid: '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/pre-extracted_features/captions/valid_predictions.json',
  test: '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/pre-extracted_features/captions/test_predictions.json',
};
local VinVL_features = {
  train: '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_trainset_full/inference/vinvl_vg_x152c4/predictions.tsv',
  test: '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_testset_full/inference/vinvl_vg_x152c4/predictions.tsv',
};
local ocr_features = {
  "train": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/pre-extracted_features/OCR/train",
  "test": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/pre-extracted_features/OCR/valid",
  "combine_with_vinvl": true,
};
local passage_data = {
  "full": "/rds/project/rds-hirYTW1FQIw/wl356/wiki_dataset",
};
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

local okvqa_data_pipeline = {
  name: 'OKVQADataPipeline',
  regenerate: false,
  do_inspect: true,
  transforms: {
    'input:LoadVinVLFeatures': {
      transform_name: 'LoadVinVLFeatures',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        VinVL_features: VinVL_features,
      },
    },
    'input:LoadOscarCaptionFeatures': {
      transform_name: 'LoadOscarCaptionFeatures',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        caption_features: caption_features,
      },
    },
    'input:LoadGoogleOCRFeatures': {
      transform_name: 'LoadGoogleOCRFeatures',
      input_node: 'input:LoadVinVLFeatures',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        ocr_features: ocr_features,
      }
    },
    'process:LoadOKVQAData': {
      input_node: [
        'input:LoadVinVLFeatures',
        'input:LoadOscarCaptionFeatures',
        'input:LoadGoogleOCRFeatures',
      ],
      transform_name: 'LoadOKVQAData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        vqa_data_path: vqa_data_path,
        image_data_path: image_data_path,
        add_images: false,
        add_caption_features: true,
        add_OCR_features: true,
        add_VinVL_features: true,
      },
    },
    'input:LoadWITData_full': {
      transform_name: 'LoadWITData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_paths: wit_data_paths,
      },
    },
    // 'process:PrepareImagesForWITData_full': {
    //   input_node: "input:LoadWITData_full",
    //   transform_name: 'PrepareImagesForWITData',
    //   regenerate: false,
    //   cache: true,
    //   setup_kwargs: {
    //     data_paths: wit_data_paths,
    //     fetch_images: false,
    //   },
    // },
    'process:LoadWITPassages_full': {
      input_node: "input:LoadWITData_full",
      transform_name: 'LoadWITPassages',
      regenerate: false,
      cache: true,
      setup_kwargs: {
      },
    },
    'process:TruncateWITPassages_full': {
      input_node: "process:LoadWITPassages_full",
      transform_name: 'TruncateWITPassages',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        truncation_length: 500,
      },
    },
    'process:IndexWITPassagesWithElasticSearch': {
      transform_name: 'IndexWITPassagesWithElasticSearch',
      input_node: [
        'process:TruncateWITPassages_full',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        index_name: "wit",
      },
    },
    'input:PrepareWITPassageAnnotations': {
      transform_name: 'PrepareWITPassageAnnotations',
      input_node: [
        'process:IndexWITPassagesWithElasticSearch',
        'process:LoadOKVQAData',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        index_name: "wit",
      },
    },
    'process:ReduceWITPassagesSize': {
      transform_name: 'ReduceWITPassagesSize',
      input_node: [
        'input:PrepareWITPassageAnnotations',
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
    'process:PrepareImagesForWITDataFromPassages_full': {
      input_node: "process:ReduceWITPassagesSize",
      transform_name: 'PrepareImagesForWITDataFromPassages',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_paths: wit_data_paths,
        fetch_images: false,
      },
    },
    'process:ReduceWITPassagesSize_AfterImagePreparation': {
      transform_name: 'ReduceWITPassagesSize',
      input_node: [
        'process:PrepareImagesForWITDataFromPassages_full',
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
    'process:ReduceWITImagesSize': {
      transform_name: 'ReduceWITImagesSize',
      input_node: [
        'process:ReduceWITPassagesSize_AfterImagePreparation',
        'process:PrepareImagesForWITDataFromPassages_full',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
      },
    },
    'process:ExtractImageFeaturesWithViT_WIT': {
      input_node: [
        'process:ReduceWITImagesSize',
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
  },
};

{
  okvqa_data_pipeline: okvqa_data_pipeline,
}
