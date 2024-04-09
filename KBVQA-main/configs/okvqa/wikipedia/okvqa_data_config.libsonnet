local vqa_data_path = {
  question_files: {
    train: '/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/OpenEnded_mscoco_train2014_questions.json',
    test: '/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/OpenEnded_mscoco_val2014_questions.json',
  },
  annotation_files: {
    train: '/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/mscoco_train2014_annotations.json',
    test: '/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/mscoco_val2014_annotations.json',
  },
};
local image_data_path = {
  train: '/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/train2014',
  test: '/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/val2014',
};
local caption_features = {
  train: '/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/pre-extracted_features/captions/train_predictions.json',
  valid: '/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/pre-extracted_features/captions/valid_predictions.json',
  test: '/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/pre-extracted_features/captions/test_predictions.json',
};
local VinVL_features = {
  train: '/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_trainset_full/inference/vinvl_vg_x152c4/predictions.tsv',
  test: '/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_testset_full/inference/vinvl_vg_x152c4/predictions.tsv',
};
local ocr_features = {
  "train": "/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/pre-extracted_features/OCR/train",
  "test": "/home/ubuntu/additional_data/projects/KBVQA/data/ok-vqa/pre-extracted_features/OCR/valid",
  "combine_with_vinvl": true,
};
local wikipedia_dataset_path = {
  "full": "/data/wiki_dataset",
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
    // 'input:LoadFullWikipediaPassageData': {
    //   transform_name: 'LoadFullWikipediaPassageData',
    //   regenerate: false,
    //   cache: true,
    //   setup_kwargs: {
    //     dataset_name: "olm/wikipedia",
    //     dataset_date: "20230301",
    //     truncation_length: 500,
    //   },
    // },
    'input:LoadWikipediaPassageData': {
      transform_name: 'LoadWikipediaPassageData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        passage_data_path: wikipedia_dataset_path,
        add_title: true,
      },
    },
    'process:IndexPassagesWithElasticSearch': {
      transform_name: 'IndexPassagesWithElasticSearch',
      input_node: [
        'input:LoadWikipediaPassageData',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        index_name: "wikipedia",
        _run_index: false,
      },
    },
    'process:PrepareWikipediaPassageAnnotationsForOKVQA': {
      transform_name: 'PrepareWikipediaPassageAnnotations',
      input_node: [
        'process:IndexPassagesWithElasticSearch',
        'process:LoadOKVQAData',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        index_name: "wikipedia",
      },
    },
    'process:ReduceWikipediaPassagesSizeForOKVQA': {
      transform_name: 'ReduceWikipediaPassagesSize',
      input_node: [
        'input:LoadWikipediaPassageData',
        'process:PrepareWikipediaPassageAnnotationsForOKVQA',
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
  },
};

{
  okvqa_data_pipeline: okvqa_data_pipeline,
}
