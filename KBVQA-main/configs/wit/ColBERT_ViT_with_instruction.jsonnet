local base = import 'ColBERT_large_vision.jsonnet';
local data = import 'wit_data_config.libsonnet';
local wit_data = data.wit_data_pipeline;

// local pretrained_ckpt_path = "/additional_data/projects/KBVQA/data/checkpoints/colbert-large-200000/colbert-200000";
// local pretrained_ckpt_path = "/home/ubuntu/additional_data/projects/ColBERT/checkpoints/colbertv2.0";
// local pretrained_ckpt_path = "/home/ubuntu/data/colbert_experiments/0915-v1-bert-base[orig_loss]-msmarco[all_n=20]-bs=128-full[cut]+shuffled-ib=True/none/2023-09/15/00.55.52/checkpoints/colbert-62254";
local pretrained_ckpt_path = "/home/ubuntu/data/colbert_experiments/1029-v1-bert-large[orig_loss]-msmarco[all_n=20]-bs=128-accum=2-full[cut]+shuffled-ib=True/none/2023-10/29/15.48.07/checkpoints/colbert-62254";


local train_with_all = {
    pipeline: {
      transforms: {
        'process:ConcatenateDatasets': {
          transform_name: 'ConcatenateDatasets',
          input_node: [
            'process:PrepareWITDataForRetrieval',
            'process:LoadCCData',
            'process:LoadLLaVaData',
            'process:LoadMSMARCOData',
            'process:LoadInfoSeekData',
            'process:LoadOvenData',
            'process:LoadKVQAData',
          ],
          regenerate: false,
          cache: true,
          setup_kwargs: {
            negative_names: ["wit_passages", "cc_passages", "llava_passages", "msmarco_passages", "infoseek_passages", "oven_passages", "kvqa_passages"],
            concat_splits: {
              'train': [true, true, true, true, false, true, true],
              'valid': [true, false, true, true, true, false, false],
            },
            splits_to_process: ["train", "valid"],
          },
        },
        'process:WrapOutputIntoKeys': {
          transform_name: 'WrapOutputIntoKeys',
          input_node: [
            'process:PrepareWITDataForRetrieval',
            'process:LoadCCData',
            'process:LoadLLaVaData',
            'process:LoadMSMARCOData',
            'process:LoadInfoSeekData',
            'process:LoadOvenData',
            'process:LoadKVQAData',
            'process:ConcatenateDatasets',
          ],
          regenerate: true,
          cache: false,
          setup_kwargs: {
            output_keys: ["wit_data", "cc_data", "llava_data", "msmarco_data", "infoseek_data", "oven_data", "kvqa_data", "combined_data"],
          },
        },
        'process:ConcatenatePassageDatasets': {
          transform_name: 'ConcatenatePassageDatasets',
          input_node: [
            'process:SplitWITPassagesForLargeScaleTraining',
            'process:LoadCCData',
            'process:LoadLLaVaData',
            'process:LoadMSMARCOData',
            'process:LoadInfoSeekData',
            'process:LoadOvenData',
            'process:LoadKVQAData',
          ],
          regenerate: false,
          cache: true,
          setup_kwargs: {
            names: ["wit_passages", "cc_passages", "llava_passages", "msmarco_passages", "infoseek_passages", "oven_passages", "kvqa_passages"],
            concat_splits: {
              'passages': [true, true, true, true, true, true, true],
              'filtered_passages': [true, false, true, false, true, false, false],
            },
          },
        },
        'output:PrepareDataloaders': {
          setup_kwargs: {
            datasets_config: {
              valid: [
                {
                  dataset_type: 'WITDatasetForDPR',
                  split: 'valid',
                  use_column: 'wit_data',
                },
                {
                  dataset_type: 'llavaDatasetForDPR',
                  split: 'valid',
                  use_column: 'llava_data',
                },
                // {
                //   dataset_type: 'MSMARCODatasetForDPR',
                //   split: 'valid',
                //   use_column: 'msmarco_data',
                // },
                {
                  dataset_type: 'InfoseekDatasetForDPR',
                  split: 'valid',
                  use_column: 'infoseek_data',
                },
              ],
              test: [
                {
                  dataset_type: 'WITDatasetForDPR',
                  split: 'valid',
                  use_column: 'wit_data',
                },
              ],
            },
          },
        },
      },
    },
    validation_indexing_source: ["wit_passages", "llava_passages", "infoseek_passages"],
    monitor: 'valid/WITDatasetForDPR.valid/pos_item_ids_recall_at_10',
    // monitor: 'valid/llavaDatasetForDPR.valid/pos_item_ids_recall_at_1',
};


local traianing_config = train_with_all;


// local clip_config = {
//   "ModelVersion": "openai/clip-vit-base-patch32",
//   "embedding_size": 768,
// };
local clip_config = {
  "ModelVersion": "openai/clip-vit-large-patch14",
  "ProcessorVersion": "openai/clip-vit-large-patch14",
  "embedding_size": 1024,
};
// local clip_config = {
//   "ModelVersion": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
//   "ProcessorVersion": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
//   "embedding_size": 1664,
// };

local tokenizer_config = {
  "tokenizer": {
    "TokenizerClass": "QueryTokenizer",
    "TokenizerModelVersion": pretrained_ckpt_path,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
  },
  "decoder_tokenizer": {
    "TokenizerClass": "DocTokenizer",
    "TokenizerModelVersion": pretrained_ckpt_path,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
  },
};
local feature_extractor_config = {
};
local image_processor_config = {
  "vit_image_processor": {
    "ImageProcessorClass": "AutoImageProcessor",
    "ImageProcessorModelVersion": clip_config.ProcessorVersion,
  },
};



local data_loader_pipeline = {
  transforms: {
    'process:LoadCCData': {
      transform_name: 'LoadPreprocessedData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "/home/ubuntu/data/CC3M/LLaVA-CC3M-Pretrain-595K/PrepareCC595kPrefixSepDataForRetrieval.hf",
        passage_path: "/home/ubuntu/data/CC3M/LLaVA-CC3M-Pretrain-595K/SplitCC595kPrefixSepPassagesForLargeScaleTraining.hf",
      },
    },
    'process:LoadLLaVaData': {
      transform_name: 'LoadPreprocessedData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "/home/ubuntu/data/LLaVA-Instruct-150K/PrepareLLAVA150kPrefixSepDataForRetrieval.hf",
        passage_path: "/home/ubuntu/data/LLaVA-Instruct-150K/SplitLLAVA150kPrefixSepPassagesForLargeScaleTraining.hf",
      },
    },
    'process:LoadMSMARCOData': {
      transform_name: 'LoadPreprocessedData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "/home/ubuntu/data/MSMARCO-400K/MSMARCO_training_with_instruction_sep.hf",
        passage_path: "/home/ubuntu/data/MSMARCO-400K/MSMARCO_passages_with_instruction_sep.hf",
        num_passages: 2000000,
      },
    },
    'process:LoadOvenData': {
      transform_name: 'LoadPreprocessedData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        // data_path: "/home/ubuntu/data/OVEN/HF/PrepareOvenWiki6MFullEntity.hf",
        // passage_path: "/home/ubuntu/data/OVEN/HF/SplitOvenWiki6MFullEntity.hf",
        data_path: "/home/ubuntu/data/OVEN/HF/PrepareOvenWiki6MSummaryEntity.hf",
        passage_path: "/home/ubuntu/data/OVEN/HF/SplitOvenWiki6MSummaryEntity.hf",
        // data_path: "/home/ubuntu/data/OVEN/HF/PrepareOvenSplitWikiEntity.hf",
        // passage_path: "/home/ubuntu/data/OVEN/HF/SplitOvenSplitWikiEntity.hf",
        shuffle_splits: ['train', 'valid'],
        num_data: {'train': 1000000, 'valid': 10000},
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
    'process:LoadInfoSeekData': {
      transform_name: 'LoadPreprocessedData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "/home/ubuntu/additional_data/projects/KBVQA/cache/InfoSeekDataPipeline/process:MergeDataColumns-16198829ca9181f8d8443592a1d2c77f.hf",
        passage_path: "/home/ubuntu/additional_data/projects/KBVQA/cache/InfoSeekDataPipeline/process:ReduceWikipediaPassagesSizeForInfoSeek-45fd077e40519f6556958a1004ed7b0a.hf",
        use_filtered_passages: true,
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
      transform_name: 'LoadPreprocessedData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "/home/ubuntu/data/KVQA/PrepareKVQADataForRetrieval.hf",
        passage_path: "/home/ubuntu/data/KVQA/SplitKVQAPassagesForLargeScaleTraining.hf",
      },
    },
    'output:PrepareDataloaders': {
      input_node: [
        'process:ConcatenatePassageDatasets',
        'process:PrepareImagesForWITData',
        // 'process:PrepareWITDataForRetrieval',
        'process:WrapOutputIntoKeys',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: false,
      setup_kwargs: {
        extra_columns: ["passages", "filtered_passages", "images"],
        pass_columns: {
          "passages": "passages",
          "filtered_passages": "filtered_passages",
          "vqa_data": "combined_data",
          "vqa_data_with_dpr_output": "combined_data",
        },
        datasets_config: {
          train: [
            {
              dataset_type: 'WITDatasetForDPR',
              split: 'train',
              use_column: 'combined_data',
            },
          ],
          valid: [
            {
              dataset_type: 'WITDatasetForDPR',
              split: 'valid',
              use_column: 'wit_data',
            },
            {
              dataset_type: 'llavaDatasetForDPR',
              split: 'valid',
              use_column: 'llava_data',
            },
            {
              dataset_type: 'MSMARCODatasetForDPR',
              split: 'valid',
              use_column: 'msmarco_data',
            },
          ],
          test: [
            {
              dataset_type: 'WITDatasetForDPR',
              split: 'valid',
              use_column: 'wit_data',
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

local data_loader = std.mergePatch(data_loader_pipeline, traianing_config.pipeline);

local data_pipeline = std.mergePatch(wit_data, data_loader);



local override = {
    data_pipeline: data_pipeline,
    model_config: {
        "base_model": "ColBERT",
        "ModelClass": "VisualColBERTForPretrainingWithShallowTransformerMappingComposedWithCrossAttn",
        "EncoderModelVersion": pretrained_ckpt_path,
        // "EncoderModelVersion": "/home/wl356/cvnlp_rds/wl356/projects/KBVQA/experiments/OKVQA_ColBERT_full_corpus/train/saved_models/validation_temp_model"，
        "VisionModelConfigClass": "CLIPVisionConfig",
        "VisionModelClass": "CLIPVisionModel",
        "VisionModelVersion": clip_config.ModelVersion,
        "pretrained": 1,
        "modules": [
            "separate_query_and_item_encoders",
        ],
        "Ks": [1, 5, 10, 20, 50, 80, 100, 500],
        "nbits": 8,
        "num_negative_samples": 1,
        "max_source_length":32,
        "max_decoder_source_length": 512,
        "mapping_network_prefix_length": 32,
        "transformer_mapping_config_base": "bert-base-uncased",
        "vision_embedding_size": clip_config.embedding_size,
        "lm_embedding_size": 128,
        "prepend_tokens": {
            "query_encoder": "",
            "item_encoder": "",
        },
        "input_modules": {
            "module_list":[
                {"type": "VisionInput",  "option": "from_file"},
                // {"type": "EmptyTextInput",  "option": "default"},
                {"type": "InstructionInput",  "option": "default", 
                "prompts": [
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
                "separation_tokens": {'start': '', 'end': ''}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessVisionInputProcessing", "option": "default"},
                {"type": "PostProcessColBERTQuestionInputTokenization", "option": "default"},
            ],
        },
        "decoder_input_modules": {
            "module_list":[
                {"type": "KnowledgeInput",  "option": "default",
                        "separation_tokens": {'start': '', 'end': ''}},
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
    executor: {
        ExecutorClass: 'ColBERTVisionPretrainingExecutor',
        init_kwargs: {
            "validation_indexing_source": traianing_config.validation_indexing_source,
        },
    },
    train: {
        num_dataloader_workers: 16,
        optimizer_config: {
            mapping_network_lr: 0.0001,
        },
        model_checkpoint_callback_paras: {
            monitor: traianing_config.monitor,
        },
    },
    metrics: [
        {'name': 'compute_DPR_scores'},
        {'name': 'compute_DPR_scores_with_pos_ids'},
    ],
};


std.mergePatch(base, override)
