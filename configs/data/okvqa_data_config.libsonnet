local vqa_data_path = {
  question_files: {
    train: '/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/OpenEnded_mscoco_train2014_questions.json',
    test: '/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/OpenEnded_mscoco_val2014_questions.json',
  },
  annotation_files: {
    train: '/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/mscoco_train2014_annotations.json',
    test: '/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/mscoco_val2014_annotations.json',
  },
};
local image_data_path = {
  train: '/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/train2014',
  test: '/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/val2014',
};
local passage_data = {
  "train": "/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/pre-extracted_features/passages/okvqa_train_corpus.csv",
  "full": "/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/pre-extracted_features/passages/okvqa_full_corpus.csv",
};
local okvqa_data_pipeline = {
  name: 'OKVQADataPipeline',
  regenerate: false,
  do_inspect: true,
  transforms: {
    'process:LoadOKVQAData': {
      transform_name: 'LoadOKVQAData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        vqa_data_path: vqa_data_path,
        image_data_path: image_data_path,
        add_images: false,
        add_caption_features: false,
        add_OCR_features: false,
        add_VinVL_features: false,
      },
    },
    'input:LoadGoogleSearchPassageData': {
      transform_name: 'LoadGoogleSearchPassageData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        passage_data_path: passage_data,
        use_full_split: true,
      },
    },
    'process:PrepareGoogleSearchPassages':{
      transform_name: 'PrepareGoogleSearchPassages',
      input_node: [
        'input:LoadGoogleSearchPassageData',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {},
    },
  },
};

{
  okvqa_data_pipeline: okvqa_data_pipeline,
}
