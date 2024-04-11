local num_negatives = 4;

local merge_data_pipeline = {
  name: 'MergeDataPipeline',
  regenerate: false,
  do_inspect: true,
  transforms: {
    'process:LoadCCData': {
      transform_name: 'LoadPreprocessedData_v2',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///CC_data",
        passage_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///CC_passages",
        image_root_folder: "/home/fz288/rds/rds-bigpicture-iS0FZqj9lmg/shared_space/datasets/CC3M-LLAVA-595K",
      },
    },
    'process:LoadLLaVaData': {
      transform_name: 'LoadPreprocessedData_v2',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///LLaVA_data",
        passage_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///LLaVA_passages",
        image_root_folder: "/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/train2014",
        add_instruction: [
          "Provide a brief description of the image along with the following question:",
          "Provide a concise explanation of the image along with the following question:",
        ],
      },
    },
    'process:LoadMSMARCOData': {
      transform_name: 'LoadPreprocessedData_v2',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///MSMARCO_data",
        passage_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///MSMARCO_passages",
      },
    },
    'process:LoadOvenData': {
      transform_name: 'LoadPreprocessedData_v2',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///OVEN_data",
        passage_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///OVEN_passages",
        image_root_folder: "/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/OVEN/Oven_images",
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
    'process:LoadKVQAData': {
      transform_name: 'LoadPreprocessedData_v2',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///KVQA_data",
        passage_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///KVQA_passages",
        image_root_folder: "/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/KVQA/KVQAimgs",
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
    'process:LoadOKVQAData': {
      transform_name: 'LoadPreprocessedData_v2',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///OKVQA_data",
        passage_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///OKVQA_passages",
        image_root_folder: "/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa",
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
    'process:LoadWITData': {
      transform_name: 'LoadPreprocessedData_v2',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///WIT_data",
        passage_path: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR///WIT_passages",
        image_root_folder: "/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/wit/images",
        add_instruction: [
          "Identify the document that is connected to this image.:",
          "Provide information about the document linked to this image.:",
          "Please describe the document that corresponds to this image.:",
          "What is the document that this image is related to?:",
          "Could you elucidate the document associated with this image?:",
          "Describe the document that accompanies this image.:",
          "Please give information on the document that goes with this image.:",
          "What document is represented by this image?:",
          "Identify the document that this image pertains to.:",
        ],
      },
    },
    // 'process:AddTextBasedVisionToInfoseek': {
    //   transform_name: 'AddTextBasedVision',
    //   input_node: 'process:LoadInfoSeekData',
    //   regenerate: false,
    //   cache: true,
    //   setup_kwargs: {
    //     splits_to_process: ["train", "valid"],
    //     caption_config: {"separation_tokens": {'start': 'Caption:', 'end': ''}},
    //     object_config: {
    //       "object_max": 40, 
    //       "attribute_max": 3, 
    //       "attribute_thres":0.05, 
    //       "ocr": 0,
    //       "separation_tokens": {'start': 'Objects:', 'sep': ',', 'end': ''}},
    //   },
    // },
    // 'process:AddInstructionToOKVQA': {
    //   transform_name: 'AddInstruction',
    //   input_node: 'process:PrepareWikipediaPassageAnnotationsForOKVQA',
    //   regenerate: false,
    //   cache: true,
    //   setup_kwargs: {
    //     splits_to_process: ["train", "valid", "test"],
    //     add_instruction: [
    //       "Using the provided image, obtain documents that address the subsequent question: ",
    //       "Retrieve documents that provide an answer to the question alongside the image: ",
    //       "Extract documents linked to the question provided in conjunction with the image: ",
    //       "Utilizing the given image, obtain documents that respond to the following question: ",
    //       "Using the given image, access documents that provide insights into the following question: ",
    //       "Obtain documents that correspond to the inquiry alongside the provided image: ",
    //       "With the provided image, gather documents that offer a solution to the question: ",
    //       "Utilizing the given image, obtain documents that respond to the following question: ",
    //     ],
    //   },
    // },
    // 'process:AddTextBasedVisionToOKVQA': {
    //   transform_name: 'AddTextBasedVision',
    //   input_node: 'process:AddInstructionToOKVQA',
    //   regenerate: false,
    //   cache: true,
    //   setup_kwargs: {
    //     splits_to_process: ["train", "valid", "test"],
    //     caption_config: {"separation_tokens": {'start': 'Caption:', 'end': ''}},
    //     object_config: {
    //       "object_max": 40, 
    //       "attribute_max": 3, 
    //       "attribute_thres":0.05, 
    //       "ocr": 1,
    //       "separation_tokens": {'start': 'Objects:', 'sep': ',', 'end': ''}},
    //   },
    // },
    // 'process:ConcatenateDatasets': {
    //   transform_name: 'ConcatenateDatasets',
    //   input_node: [
    //     'process:LoadWITData',
    //     'process:LoadCCData',
    //     'process:LoadLLaVaData',
    //     'process:LoadMSMARCOData',
    //     'process:LoadOvenData',
    //     'process:LoadKVQAData',
    //     'process:LoadOKVQAData',
    //     'process:LoadEVQAData',
    //   ],
    //   regenerate: false,
    //   cache: true,
    //   setup_kwargs: {
    //     negative_names: ["wit_passages", "cc_passages", "llava_passages", "msmarco_passages", "oven_passages", "kvqa_passages", 'okvqa_passages', 'evqa_passages'],
    //     concat_splits: {
    //       'train': [true, true, true, true, true, true, 10, true],
    //       'valid': [true, false, true, true, false, false, true, true],
    //       'test': [true, false, true, true, false, false, true, true],
    //     },
    //     splits_to_process: ["train", "valid", "test"],
    //   },
    // },
    'process:ConcatenateDatasets': {
      transform_name: 'ConcatenateDatasets',
      input_node: [
        'process:LoadKVQAData',
        'process:LoadOKVQAData',
        'process:LoadEVQAData'
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        negative_names: ["kvqa_passages", "okvqa_passages", "evqa_passages"],
        concat_splits: {
          'train': [true, true, true],
          'valid': [true, true, true],
          'test': [true, true, true]
        },
        splits_to_process: ["train", "valid", "test"],
      },
    },
    'process:WrapOutputIntoKeys': {
      transform_name: 'WrapOutputIntoKeys',
      input_node: [
        'process:LoadWITData',
        'process:LoadCCData',
        'process:LoadLLaVaData',
        'process:LoadMSMARCOData',
        'process:LoadOvenData',
        'process:LoadKVQAData',
        'process:LoadOKVQAData',
        'process:LoadEVQAData',
        'process:ConcatenateDatasets',
      ],
      regenerate: true,
      cache: false,
      setup_kwargs: {
        output_keys: ["wit_data", "cc_data", "llava_data", "msmarco_data", "oven_data", "kvqa_data", "okvqa_data", 'evqa_data', "combined_data"],
      },
    },
    'process:ConcatenatePassageDatasets': {
      transform_name: 'ConcatenatePassageDatasets',
      input_node: [
        'process:LoadWITData',
        'process:LoadCCData',
        'process:LoadLLaVaData',
        'process:LoadMSMARCOData',
        'process:LoadOvenData',
        'process:LoadKVQAData',
        'process:LoadOKVQAData',
        'process:LoadEVQAData',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        names: ["wit_passages", "cc_passages", "llava_passages", "msmarco_passages", "oven_passages", "kvqa_passages", 'okvqa_passages', 'evqa_passages'],
        concat_splits: {
          'train_passages': [true, true, true, true, true, true, true, true],
          'valid_passages': [true, false, true, true, false, false, true, true],
          'test_passages': [true, false, true, true, false, false, true, true],
        },
      },
    },
  },
};

{
  merge_data_pipeline: merge_data_pipeline,
}
