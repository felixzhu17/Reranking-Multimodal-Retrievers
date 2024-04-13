local num_negatives = 4;

local merge_data_pipeline = {
  name: 'MergeDataPipeline',
  regenerate: false,
  do_inspect: true,
  transforms: {
    'process:LoadEVQAData': {
      transform_name: 'LoadPreprocessedData_v2',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///EVQA_data",
        passage_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///EVQA_passages",
        image_root_folder: "/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/EVQA/images",
        add_instruction: [
          "Using the provided image, obtain documents that address the subsequent question: ",
          "Retrieve documents that provide an answer to the question alongside the image: ",
          "Extract documents linked to the question provided in conjunction with the image: ",
          "Utilizing the given image, obtain documents that respond to the following question: ",
          "Using the given image, access documents that provide insights into the following question: ",
          "Obtain documents that correspond to the inquiry alongside the provided image: ",
          "With the provided image, gather documents that offer a solution to the question: ",
          "Utilizing the given image, obtain documents that respond to the following question: ",
        ],
      },
    },
  'process:WrapOutputIntoKeys': {
    transform_name: 'WrapOutputIntoKeys',
    input_node: [
      'process:LoadEVQAData',
    ],
    regenerate: true,
    cache: true,
    setup_kwargs: {
      output_keys: ['evqa_data'],
    },
  },
  'process:ConcatenatePassageDatasets': {
    transform_name: 'ConcatenatePassageDatasets',
    input_node: [
      'process:LoadEVQAData',
    ],
    regenerate: false,
    cache: true,
    setup_kwargs: {
      names: ['evqa_passages'],
      concat_splits: {
        'train_passages': [true],
        'valid_passages': [true],
        'test_passages': [true],
      },
    },
  },
  },
};

{
  merge_data_pipeline: merge_data_pipeline,
}
