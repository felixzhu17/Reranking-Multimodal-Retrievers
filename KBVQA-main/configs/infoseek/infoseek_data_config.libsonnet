
local VinVL_features = {
  full: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_10k_2k_2k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_0k_20k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_0k-20k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_20k_40k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_20k-40k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_40k_60k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_40k-60k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_60k_100k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_60k-100k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_100k_150k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_100k-150k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_150k_200k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_150k-200k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_200k_250k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_200k-250k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_250k_300k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_250k-300k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_300k_350k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_300k-350k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_350k_400k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_350k-400k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_400k_450k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_400k-450k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_450k_500k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_450k-500k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_500k_550k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_500k-550k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_550k_600k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_550k-600k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_600k_650k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_600k-650k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_650k_700k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_650k-700k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_700k_750k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_700k-750k/inference/vinvl_vg_x152c4/predictions.tsv',
  train_750k_770k: '/data/infoseek/infoseek/preprocessed_features/vinvl_features/infoseek_train_750k-770k/inference/vinvl_vg_x152c4/predictions.tsv',
};
local image_data_path = {
  full: '/data/infoseek/infoseek/infoseek_images/images',
};
local vqa_data_path = {
  "train": "/data/infoseek/infoseek/infoseek_data/infoseek_train.jsonl",
  "val": "/data/infoseek/infoseek/infoseek_data/infoseek_val.jsonl",
  // "test": "/data/infoseek/infoseek/infoseek_data/infoseek_test.jsonl",
  // "human": "/data/infoseek/infoseek/infoseek_data/infoseek_human.jsonl"
};
local passage_annotation_file_path = {
  "train": "/data/infoseek/infoseek/infoseek_data/infoseek_train_withkb.jsonl",
  "val": "/data/infoseek/infoseek/infoseek_data/infoseek_val_withkb.jsonl",
};
local passage_data_path = {
  "full": "/data/infoseek/infoseek/Wiki6M_ver_1_0_title_only.jsonl",
};
local wikipedia_dataset_path = {
  "full": "/data/wiki_dataset",
};


local infoseek_data_pipeline = {
  name: 'InfoSeekDataPipeline',
  regenerate: false,
  do_inspect: true,
  transforms: {
    'input:LoadInfoSeekVinVLFeatures': {
      transform_name: 'LoadVinVLFeatures',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        add_images: false,
        VinVL_features: VinVL_features,
      },
    },
    'process:LoadInfoSeekData': {
      input_node: [
        'input:LoadInfoSeekVinVLFeatures',
      ],
      transform_name: 'LoadInfoSeekData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        vqa_data_path: vqa_data_path,
        image_data_path: image_data_path,
        passage_annotation_file_path: passage_annotation_file_path,
        add_VinVL_features: true,
        num_train: -1,
        num_valid: -1,
      },
    },
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
    'process:PrepareWikipediaPassageAnnotationsForInfoSeek': {
      transform_name: 'PrepareWikipediaPassageAnnotationsForInfoSeek',
      input_node: [
        'process:IndexPassagesWithElasticSearch',
        'process:LoadInfoSeekData',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        index_name: "wikipedia",
        supervision_type: "ground-truth",
        num_train: -1,
        num_valid: 5120,
      },
    },
    'process:ReduceWikipediaPassagesSizeForInfoSeek': {
      transform_name: 'ReduceWikipediaPassagesSizeForInfoSeek',
      input_node: [
        'input:LoadWikipediaPassageData',
        'process:PrepareWikipediaPassageAnnotationsForInfoSeek',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        num_random_passages: 30000,
      },
    },
  },
};

{
  infoseek_data_pipeline: infoseek_data_pipeline,
}
