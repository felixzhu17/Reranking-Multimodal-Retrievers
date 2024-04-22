local meta = import '../../meta_configs/hpc_meta_config.libsonnet';
local okvqa_data_pipeline_config = import "../../okvqa/wikipedia/okvqa_data_config.libsonnet";
local okvqa_data_pipeline = okvqa_data_pipeline_config.okvqa_data_pipeline;

local pretrained_ckpt_path = "/home/ubuntu/additional_data/projects/ColBERT/checkpoints/colbertv2.0";

local tokenizer_config = {
  "tokenizer": {
    "TokenizerClass": "QueryTokenizer",
    "TokenizerModelVersion": pretrained_ckpt_path,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
  },
  "decoder_tokenizer": {
    "TokenizerClass": "Blip2Processor",
    "TokenizerModelVersion": "Salesforce/blip2-flan-t5-xl",
    // "TokenizerModelVersion": "Salesforce/blip2-opt-2.7b",
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
      // "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
  },
};
local clip_config = {
  "ModelVersion": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
  "embedding_size": 1664,
};
local feature_extractor_config = {
};
local image_processor_config = {
  "vit_image_processor": {
    "ImageProcessorClass": "AutoImageProcessor",
    "ImageProcessorModelVersion": clip_config.ModelVersion,
  },
};


local index_files = {
  "index_path": "$AWS_Finetune_Infoseek_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_index/test/generate_index/step_500/colbert_index",
  "embedding_path": "$AWS_Finetune_Infoseek_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_index/test/generate_index/step_500/item_embeddings.pkl",
  "static_results": [
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_EVQADatasetForDPR.train_predictions_rank_0.pkl",
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_EVQADatasetForDPR.train_predictions_rank_1.pkl",
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_EVQADatasetForDPR.train_predictions_rank_2.pkl",
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_EVQADatasetForDPR.train_predictions_rank_3.pkl",
    // "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_valid_index/test/generate_valid_index/generate_valid_index_test_EVQADatasetForDPR.valid_predictions_rank_0.pkl",
    // "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_valid_index/test/generate_valid_index/generate_valid_index_test_EVQADatasetForDPR.valid_predictions_rank_1.pkl",
    // "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_valid_index/test/generate_valid_index/generate_valid_index_test_EVQADatasetForDPR.valid_predictions_rank_2.pkl",
    // "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_valid_index/test/generate_valid_index/generate_valid_index_test_EVQADatasetForDPR.valid_predictions_rank_3.pkl",
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_EVQADatasetForDPR.valid_predictions_rank_0.pkl",
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_EVQADatasetForDPR.valid_predictions_rank_1.pkl",
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_EVQADatasetForDPR.valid_predictions_rank_2.pkl",
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_EVQADatasetForDPR.valid_predictions_rank_3.pkl",
  ],
};
// local index_files = {
//   "index_path": "$OKVQA_VisualColBERT_with_pretrained_ViT(WIT)_ColBERT_mapping_trainable_ViT_frozen_with_text_based_vision/test/generate_index/step_5016/colbert_index",
//   "embedding_path": "$OKVQA_VisualColBERT_with_pretrained_ViT(WIT)_ColBERT_mapping_trainable_ViT_frozen_with_text_based_vision/test/generate_index/step_5016/item_embeddings.pkl",
// };

local data_loader = {
  name: "WITDataPipeline",
  transforms: {
    'process:LoadInfoSeekData': {
      transform_name: 'LoadPreprocessedData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_path: "/home/ubuntu/additional_data/projects/KBVQA/cache/InfoSeekDataPipeline/process:MergeDataColumns-16198829ca9181f8d8443592a1d2c77f.hf",
        passage_path: "/home/ubuntu/additional_data/projects/KBVQA/cache/InfoSeekDataPipeline/process:ReduceWikipediaPassagesSizeForInfoSeek-45fd077e40519f6556958a1004ed7b0a.hf",
        use_filtered_passages: true,
        shuffle_splits: ['train'],
        num_data: {'train': 100000, 'valid': -1},
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
      transform_name: 'LoadPreprocessedData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        // shuffle_splits: ['train'],
        // /home/ubuntu/data/EVQA/HF/PrepareEVQADownsampleWikiWeb.hf 
        // /home/ubuntu/data/EVQA/HF/SplitEVQADownsampleWikiWeb.hf
        // data_path: "/home/ubuntu/data/EVQA/HF/PrepareEVQADownsampleWikiWeb.hf",
        // passage_path: "/home/ubuntu/data/EVQA/HF/SplitEVQADownsampleWikiWeb.hf",
        data_path: "/home/ubuntu/data/EVQA/HF/PrepareEVQASingleTestWikiWeb.hf",
        passage_path: "/home/ubuntu/data/EVQA/HF/SplitEVQASingleTestWikiWeb.hf",
        // data_path: "/home/ubuntu/data/EVQA/HF/PrepareEVQAWikiWeb.hf",
        // passage_path: "/home/ubuntu/data/EVQA/HF/SplitEVQAWikiWeb.hf",
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
    'process:AddTextBasedVisionToInfoseek': {
      transform_name: 'AddTextBasedVision',
      input_node: 'process:LoadInfoSeekData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        splits_to_process: ["train", "valid"],
        caption_config: {"separation_tokens": {'start': 'Caption:', 'end': ''}},
        object_config: {
          "object_max": 40, 
          "attribute_max": 3, 
          "attribute_thres":0.05, 
          "ocr": 0,
          "separation_tokens": {'start': 'Objects:', 'sep': ',', 'end': ''}},
      },
    },
    'process:AddInstructionToOKVQA': {
      transform_name: 'AddInstruction',
      input_node: 'process:PrepareWikipediaPassageAnnotationsForOKVQA',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        splits_to_process: ["train", "valid", "test"],
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
    'process:AddTextBasedVisionToOKVQA': {
      transform_name: 'AddTextBasedVision',
      input_node: 'process:AddInstructionToOKVQA',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        splits_to_process: ["train", "valid", "test"],
        caption_config: {"separation_tokens": {'start': 'Caption:', 'end': ''}},
        object_config: {
          "object_max": 40, 
          "attribute_max": 3, 
          "attribute_thres":0.05, 
          "ocr": 1,
          "separation_tokens": {'start': 'Objects:', 'sep': ',', 'end': ''}},
      },
    },
    'process:ConcatenateDatasets': {
      transform_name: 'ConcatenateDatasets',
      input_node: [
        'process:LoadInfoSeekData',
        // 'process:AddTextBasedVisionToInfoseek',
        // 'process:AddInstructionToOKVQA',
        'process:AddTextBasedVisionToOKVQA',
        'process:LoadEVQAData',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        negative_names: ["infoseek_passages", 'okvqa_passages', 'evqa_passages'],
        concat_splits: {
          'train': [false, false, true],
          'valid': [false, false, true],
        },
        splits_to_process: ["train", "valid"],
      },
    },
    'process:WrapOutputIntoKeys': {
      transform_name: 'WrapOutputIntoKeys',
      input_node: [
        'process:LoadInfoSeekData',
        // 'process:AddTextBasedVisionToInfoseek',
        // 'process:AddInstructionToOKVQA',
        'process:AddTextBasedVisionToOKVQA',
        'process:LoadEVQAData',
        'process:ConcatenateDatasets',
      ],
      regenerate: true,
      cache: false,
      setup_kwargs: {
        output_keys: ["infoseek_data", "okvqa_data", 'evqa_data', "combined_data"],
      },
    },
    'process:ConcatenatePassageDatasets': {
      transform_name: 'ConcatenatePassageDatasets',
      input_node: [
        'process:LoadInfoSeekData',
        'process:ReduceWikipediaPassagesSizeForOKVQA',
        'process:LoadEVQAData',
      ],
      regenerate: false,
      cache: true,
      setup_kwargs: {
        names: ["infoseek_passages", 'okvqa_passages', 'evqa_passages'],
        concat_splits: {
          'passages': [false, false, true],
          'filtered_passages': [false, false, true],
        },
      },
    },
    'output:PrepareDataloaders': {
      input_node: [
        'process:ConcatenatePassageDatasets',
        'process:WrapOutputIntoKeys',
      ],
      transform_name: 'PrepareDataloaders',
      regenerate: true,
      cache: false,
      setup_kwargs: {
        extra_columns: ["passages", "filtered_passages"],
        pass_columns: {
          "passages": "passages",
          "filtered_passages": "filtered_passages",
          "vqa_data": "combined_data",
        },
        datasets_config: {
          train: [
            {
              dataset_type: 'EVQADataset',
              split: 'train',
              use_column: 'combined_data',
            },
          ],
          valid: [
            {
              dataset_type: 'EVQADataset',
              split: 'valid',
              use_column: 'evqa_data',
            },
            // {
            //   dataset_type: 'OKVQADataset',
            //   split: 'valid',
            //   use_column: 'okvqa_data',
            // },
            // {
            //   dataset_type: 'EVQADataset',
            //   split: 'valid',
            //   use_column: 'evqa_data',
            // },
          ],
          test: [
            {
              dataset_type: 'EVQADataset',
              split: 'valid',
              use_column: 'evqa_data',
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

local data_pipeline = std.mergePatch(okvqa_data_pipeline, data_loader);

{
    experiment_name: 'default_RAG',
    test_suffix: 'default_test',
    meta: meta.default_meta,
    data_pipeline: data_pipeline,
    model_config: {
        "base_model": "RAG",
        "ModelClass": "RagModelForBlip",
        "TokenizerClass": "QueryTokenizer",  // question encoder tokenizer
        "DecoderTokenizerClass": tokenizer_config.decoder_tokenizer.TokenizerClass,  // generator tokenizer
        "DecoderTokenizerModelVersion": tokenizer_config.decoder_tokenizer.TokenizerModelVersion, // generator tokenizer version
        
        "VisionModelConfigClass": "CLIPVisionConfig",
        "VisionModelClass": "CLIPVisionModel",
        "VisionModelVersion": clip_config.ModelVersion,

        "QueryEncoderBaseModelVersion": pretrained_ckpt_path,
        "QueryEncoderModelClass": "VisualColBERTForRetrievalWithShallowTransformerMappingComposed", // question encoder
        "QueryEncoderModelVersion": "$AWS_Finetune_Infoseek_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)/train/saved_models/model_step_500.ckpt",
        
        "GeneratorModelClass": "Blip2ForConditionalGeneration", // answer generator
        "GeneratorConfigClass": "Blip2Config",
        "GeneratorModelVersion": "Salesforce/blip2-flan-t5-xl",
        // "GeneratorModelVersion": "Salesforce/blip2-opt-2.7b",
        "pretrained": 1,
        "RAVQA_loss_type": "Approach6",
        "loss_ratio": {
            "nll_loss": 1,
            "rag_loss": 0,
            "additional_loss": 0,
        },
        "modules": [
            "freeze_question_encoder",
            "force_existence",
        ],
        "Ks": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 80, 100],
        "num_beams": 5,
        "num_ROIs": 0,
        "max_source_length":512,
        "max_decoder_source_length": 512,
        'max_target_length':10,
        'num_knowledge_passages': 5,
        "mapping_network_prefix_length": 32,
        "transformer_mapping_config_base": "bert-base-uncased",
        "vision_embedding_size": clip_config.embedding_size,
        "lm_embedding_size": 128,
        "index_files": index_files,
        "prepend_tokens": {
            "query_encoder": "",
            "item_encoder": "",
        },
        "input_modules": {
            "module_list":[
                {"type": "QuestionInput",  "option": "default", 
                        "separation_tokens": {'start': '', 'end': ''}},
                // {"type": "TextBasedVisionInput",  "option": "caption",
                //         "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
                // {"type": "TextBasedVisionInput",  "option": "object", 
                //         "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
                //         "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
                // {"type": "VisionInput",  "option": "from_file", "use_ROI": false},
                {"type": "VisionInput", "option": "from_file"},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessColBERTQuestionInputTokenization", "option": "default"},
                {"type": "PostProcessBlip2VisionInputProcessing", "option": "default"},
                {"type": "PostProcessVisionInputProcessing", "option": "default"},
            ],
        },
        "decoder_input_modules": {
            "module_list":[
                // {"type": "KnowledgeInput",  "option": "default",
                //         "separation_tokens": {'start': '<BOK>', 'end': '<EOK>'}},
            ],
            "postprocess_module_list": [
                // {"type": "PostProcessColBERTItemInputTokenization", "option": "default"},
            ],
        },
        "output_modules": {
            "module_list":[
                {"type": "GenerationOutput", "option": "default"},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessBlipOutputTokenization", "option": "default"},
            ],
        },
    },
    executor: {
        ExecutorClass: 'RagBlipExecutor',
        init_kwargs: {
            "use_data_node": "output:PrepareDataloaders",
        },
    },
    train: {
        batch_size: 8,
        num_dataloader_workers: 0,
        trainer_paras: {
            max_epochs: 10,
            accumulate_grad_batches: 4,
            check_val_every_n_epoch: null,
            val_check_interval: 10,
            log_every_n_steps: 10,
        },
        trainer_fit_paras: {
            ckpt_path: null,
        },
        model_checkpoint_callback_paras: {
            monitor: 'valid/EVQADataset.valid/accuracy',
            save_top_k: 1,
            mode: "max",
            filename: 'model_step_{step}',
            save_last: true,
            verbose: true,
            auto_insert_metric_name: false,
            save_on_train_epoch_end: false,
        },
        early_stopping_callback_paras: {
            patience: 3,
            verbose: true,
            mode: "max",
        },
        optimizer_config: {
            optimizer_name: "AdamW",
            optimizer_params: {
                lr: 0.0006,
                eps: 1e-08,
            },
            retriever_lr: 0.0001,
            scheduler: "linear",
            scheduler_params: {
                num_warmup_steps: 0,
            },
        },
        weight_decay: 0.05,
        label_smoothing_factor: 0.1,
    },
    valid: {
        batch_size: 64,
        num_dataloader_workers: 0,
    },
    test: {
        checkpoint_name: "",
        load_model_path: "",
        load_best_model: false,
        trainer_paras: {},
        batch_size: 64,
        num_dataloader_workers: 0,
    },
    eval: {
        'eval_op_name': 'Your eval op name'
    },
    "metrics": [
        {'name': 'compute_exact_match_with_numeric_values'},
        {'name': 'compute_retrieval_metrics'},
    ],
}
