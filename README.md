# Retrieval Augmented Visual Question Answering
This is the research repository of the Retrieval Augmented Visual Question Answering (RAVQA) project.

The author is Weizhe Lin, PhD student of the Department of Engineering, University of Cambridge.

**This repository is not-for-distribution in its current form.**

<!-- TOC -->

- [Retrieval Augmented Visual Question Answering](#retrieval-augmented-visual-question-answering)
  - [Overview](#overview)
    - [Structure](#structure)
    - [Configs](#configs)
    - [ModuleParser](#moduleparser)
    - [MetricsProcessor](#metricsprocessor)
    - [NOTE](#note)
  - [Environments](#environments)
  - [Download Datasets](#download-datasets)
    - [COCO images](#coco-images)
    - [OKVQA Dataset](#okvqa-dataset)
    - [Google Search Corpus](#google-search-corpus)
  - [Feature Extraction](#feature-extraction)
    - [VinVL Features (object detection/attributes/relations)](#vinvl-features-object-detectionattributesrelations)
      - [Step 1: Install environments](#step-1-install-environments)
      - [Step 2: Generating OKVQA datasets](#step-2-generating-okvqa-datasets)
      - [Step 3: Download pre-trained models](#step-3-download-pre-trained-models)
      - [Step 4: Running models](#step-4-running-models)
      - [Step 5: Recommended Save Path](#step-5-recommended-save-path)
    - [Oscar+ Features (image captioning)](#oscar-features-image-captioning)
      - [Step 1: Download data](#step-1-download-data)
      - [Step 2: Download the pre-trained model](#step-2-download-the-pre-trained-model)
      - [Step 3: Running the inference](#step-3-running-the-inference)
      - [Step 4: Recommended Save Path](#step-4-recommended-save-path)
    - [Google OCR Features](#google-ocr-features)
  - [Dense Passage Retrieval](#dense-passage-retrieval)
    - [Training](#training)
    - [Generating Static Retrieval Results by Testing](#generating-static-retrieval-results-by-testing)
    - [Prepare FAISS index files for dynamic DPR retrieval](#prepare-faiss-index-files-for-dynamic-dpr-retrieval)
  - [Baseline models without DPR for retrieval](#baseline-models-without-dpr-for-retrieval)
    - [RA-VQA-NoDPR (T5 baseline)](#ra-vqa-nodpr-t5-baseline)
  - [Baseline models with DPR](#baseline-models-with-dpr)
    - [TRiG](#trig)
    - [T5 baseline with Knowledge](#t5-baseline-with-knowledge)
  - [RAVQA framework](#ravqa-framework)
    - [Static retrieval](#static-retrieval)
    - [RA-VQA-FrDPR](#ra-vqa-frdpr)
    - [RA-VQA-NoPR](#ra-vqa-nopr)
    - [RA-VQA](#ra-vqa)
    - [RA-VQA-NoCT](#ra-vqa-noct)
    - [RA-VQA on Wikipedia](#ra-vqa-on-wikipedia)

<!-- /TOC -->


## Overview
The training and testing are backboned by pytorch-lightning. The pre-trained Transformer models are from Huggingface-transformers. The training platform is Pytorch.

### Structure
The framework consists of:

1. **main.py**: the main program. It loads a config file and override some entries with command-line arguments. It initialises a data loader wrapper, a model trainer, and a pytorch-lightning trainer to execute training and testing.
2. **Data Loader Wrapper**: it loads the data according to `data_modules` defined in config files. `.set_dataloader()` is called after data loading is finished. `.train_dataloader` and `.test_dataloader` are loaded.
3. **Datasets**: they are automatically loaded by the data loader wrapper. `.collate_fn` is defined to collate the data. An decorator class `ModuleParser` is used to help generate the training inputs. This decorator class generates input dict according to configs (`config.model_config.input_modules/decorder_input_modules/output_modules`).
4. **Model Trainers**: a pytorch-lightning `LightningModule` instance. It defines training/testing behaviors (training steps, optimizers, schedulers, logging, checkpointing, and so on). It initialises the model being trained at `self.model`.
5. **Models**: pytorch `nn.Modules` models.

### Configs
The configuration is achieved with `jsonnet`. It enables inheritance of config files. For example, `RAVQA.jsonnet` override its configs to `RAVQA_base.jsonnet`, which again inherits from `base_env.jsonnet` where most of important paths are defined.

By including the corresponding key:value pair in the config file, overriding can be easily performed.

### ModuleParser
A decorator class that helps to parse data into features that are used by models.

An example is shown below:
```
"input_modules": {
    "module_list":[
    {"type": "QuestionInput",  "option": "default", 
                "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},  
    {"type": "TextBasedVisionInput",  "option": "caption",
                "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
    {"type": "TextBasedVisionInput",  "option": "object", 
                "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
                "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
    ],
    "postprocess_module_list": [
    {"type": "PostProcessInputTokenization", "option": "default"},
    ],
},
"decoder_input_modules": {
    "module_list":[],
    "postprocess_module_list": [],
},
"output_modules": {
    "module_list":[
    {"type": "GenerationOutput", "option": "default"},
    ],
    "postprocess_module_list": [
    {"type": "PostProcessOutputTokenization", "option": "default"},
    ],
},
```
which first generates text_sequences:
```
<BOQ> Question <EOQ> <BOC> Caption <EOC> <BOV> obj1 attr1 attr2 <SOV> obj2 ... [OCR results] <EOV>
```
in the order defined in `input_modules`, and then the postprocessing unit `PostProcessInputTokenization` is used to tokenize the input into `input_ids` and `input_attention_masks`.

By defining new functions in `ModuleParser`, e.g. `self.TextBasedVisionInput`, a new behavior can be easily introduced to transform modules into training features.

### MetricsProcessor
The following entries in config file `test.metrics` define the metrics to compute in validation and testing. Each module uploads `log_dict` with `metrics_name: metrics_value` which can be processed in trainers conveniently.
```
"metrics": [
    {'name': 'compute_exact_match'},
    {'name': 'compute_retrieval_metrics'},
    {'name': 'compute_okvqa_scores'},
],
```

### NOTE
This framework is designed for **research purpose**, with flexibility for extension. It is not a perfect framework for production, of course.

## Environments
Create virtualenv:
```
conda create -n VQA python=3.8
conda activate VQA
```
Install Pytorch:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Install other libraries:
```
pip install transformers==4.38.2
conda install -c pytorch faiss-gpu -y
pip install setuptools==59.5.0
pip install wandb pytorch-lightning==2.0.4 jsonnetbin easydict pandas scipy opencv-python fuzzywuzzy scikit-image matplotlib timm scikit-learn sentencepiece tensorboard datasets
pip install ujson evaluate GPUtil easydict peft==0.4.0
pip install bitarray spacy ujson gitpython
pip install ninja
pip install absl-py
pip install openai
pip install sacrebleu
pip install diffusers==0.20.1
pip install einops transformers_stream_generator tiktoken
cd third_party/ColBERT
pip install -e .
```


## Download Datasets
### COCO images
`data\ok-vqa\train2014`: [Train images](http://images.cocodataset.org/zips/train2014.zip)

`data\ok-vqa\val2014`: [Test images](http://images.cocodataset.org/zips/val2014.zip)

### OKVQA Dataset
`data\ok-vqa\mscoco_train2014_annotations.json`: [Training annotations](https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip)

`data\ok-vqa\mscoco_val2014_annotations.json`: [Testing annotations](https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip)

`data\ok-vqa\OpenEnded_mscoco_train2014_questions.json`: [Training questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip)

`data\ok-vqa\OpenEnded_mscoco_val2014_questions.json`: [Testing questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip)

### Google Search Corpus
[Official download link](https://drive.google.com/drive/folders/15uWx33RY5UmR_ZmLO6Ve1wyzbXsLxV6o?usp=sharing)

Data can be saved to `data\ok-vqa\pre-extracted_features\passages\okvqa_full_corpus.csv`.


## Feature Extraction
### VinVL Features (object detection/attributes/relations)
#### Step 1: Install environments
VinVL needs a separate env.

Refer to [Offical installation guide](https://github.com/microsoft/scene_graph_benchmark/blob/main/INSTALL.md)

Since HPC uses A-100, which requires a higher version of CUDA, the recommended environment with CUDA 10.1 does not work.

```
conda create --name sg_benchmark python=3.7 -y
conda activate sg_benchmark
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install ipython h5py nltk joblib jupyter pandas scipy -y
pip install ninja yacs>=0.1.8 cython matplotlib tqdm opencv-python numpy>=1.19.5 
python -m pip install cityscapesscripts
pip install pycocotools scikit-image timm einops
cd scene_graph_benchmark
python setup.py build develop
```

A verified environment on the Cambridge HPC is: `py3.7_cuda11.0.221_cudnn8.0.5_0`.
Its spec file can be loaded by:
```
conda create --name sg_benchmark --file materials/scene_graph_benchmark/hpc-spec-file.txt
```

#### Step 2: Generating OKVQA datasets
```
cd materials/scene_graph_benchmark
python tools/prepare_data_for_okvqa.py
```
This command generates trainset/testset of OKVQA datasets to `datasets/okvqa/`, which will be used in object detection.

#### Step 3: Download pre-trained models
```
mkdir models
mkdir models/vinvl
/path/to/azcopy copy https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth ./models/vinvl/
```

#### Step 4: Running models
`vinvl_vg_x152c4` is a pre-trained model with object and attribute detection:
For OKVQA dataset:
```
python tools/test_sg_net.py \
    --config-file sgg_configs/vgattr/vinvl_x152c4_okvqa_testset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```
```
python tools/test_sg_net.py  \
    --config-file sgg_configs/vgattr/vinvl_x152c4_okvqa_trainset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```
For FVQA dataset:
```
python tools/test_sg_net.py \
    --config-file sgg_configs/vgattr/vinvl_x152c4_fvqa_testset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```
```
python tools/test_sg_net.py  \
    --config-file sgg_configs/vgattr/vinvl_x152c4_fvqa_trainset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```

`vinvl_large` is a pre-trained model with **only** object detection. But it was pre-trained on more object detection datasets!
```
python tools/test_sg_net.py  \
    --config-file sgg_configs/vgattr/vinvl_large_okvqa_testset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_large.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```
```
python tools/test_sg_net.py  \
    --config-file sgg_configs/vgattr/vinvl_large_okvqa_trainset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_large.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```
Relation detection. Not used in this project but provided for reference.
```
python tools/test_sg_net.py  \
    --config-file sgg_configs/oi_vrd/R152FPN_reldn_oi_OKVQA_testset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/sgg_model_zoo/oi_R152_reldn.pth  \
    DATA_DIR "./datasets/"  \
    TEST.OUTPUT_FEATURE True
```
#### Step 5: Recommended Save Path
The object/attribute data can be saved to `data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_trainset_full/inference/vinvl_vg_x152c4/predictions.tsv`.

The relation data can be saved to `data\ok-vqa\pre-extracted_features\relation_features\train`.

### Oscar+ Features (image captioning)
#### Step 1: Download data
We can download COCO-caption data with azcopy:
```
cd materials/Oscar
path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/datasets/coco_caption' ./oscar_dataset --recursive
```
Reference: [offical download page](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md)

#### Step 2: Download the pre-trained model
We can download [COCO captioning large](https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/image_captioning/coco_captioning_large_scst.zip) here, or refer to the [official download page](https://github.com/microsoft/Oscar/blob/master/VinVL_MODEL_ZOO.md#Image-Captioning-on-COCO) for the model checkpoints.

Save the pre-trained model to `pretrained_models/coco_captioning_large_scst`.

#### Step 3: Running the inference
```
python oscar/run_captioning.py \
    --do_test \
    --do_eval \
    --test_yaml oscar_dataset/coco_caption/[train/val/test].yaml \
    --per_gpu_eval_batch_size 64 \
    --num_beams 5 \
    --max_gen_length 20 \
    --output_prediction_path './output/[train/val/test]_predictions.json' \
    --eval_model_dir pretrained_models/coco_captioning_large_scst/checkpoint-4-50000
```
For FVQA dataset
```
python oscar/run_captioning.py \
    --do_test \
    --do_eval \
    --test_yaml ../scene_graph_benchmark/datasets/fvqa_for_oscar/test.yaml \
    --per_gpu_eval_batch_size 16 \
    --num_beams 5 \
    --max_gen_length 20 \
    --output_prediction_path './output/test_predictions.json' \
    --eval_model_dir /mnt/e/projects/Oscar/pretrained_models/coco_captioning_large_scst/checkpoint-4-50000
```

Note that in the script, `transformer` is renamed to `transformer2` such that it won't conflict with existing `transformer` package in your environment.

#### Step 4: Recommended Save Path
The data can be saved to `data\ok-vqa\pre-extracted_features\captions\train_predictions.json`.


### Google OCR Features
First, enable Google OCR APIs; download the key file to `google_ocr_key.json`. This is **not** free! Ask me for the already generated features.
```
cd src
python ocr.py
```
The detected features will be saved to `data/ok-vqa/pre-extracted_features/OCR`.

## Dense Passage Retrieval
### Training
```
python main.py ../configs/DPR.jsonnet --mode train --experiment_name OKVQA_DPR_FullCorpus --accelerator auto --devices auto --opts train.epochs=10 train.batch_size=30 valid.step_size=1 valid.batch_size=32 train.additional.gradient_accumulation_steps=2 train.lr=0.00001
```
### Generating Static Retrieval Results by Testing
Training set:
```
python main.py ../configs/DPR.jsonnet --mode test --experiment_name OKVQA_DPR_FullCorpus --accelerator auto --devices auto --test_evaluation_name generate_test_set --opts train.batch_size=64 valid.batch_size=64 test.load_model_path=/home/wl356/rds/rds-cvnlp-hirYTW1FQIw/wl356/Experiments/OKVQA_DPR_FullCorpus/train/saved_model/model_05.ckpt
```
Testing set:
```
python main.py ../configs/DPR.jsonnet --mode test --experiment_name OKVQA_DPR_FullCorpus --accelerator auto --devices auto --test_evaluation_name generate_train_set --opts train.batch_size=64 valid.batch_size=64 test.load_model_path=/home/wl356/rds/rds-cvnlp-hirYTW1FQIw/wl356/Experiments/OKVQA_DPR_FullCorpus/train/saved_model/model_05.ckpt data_loader.use_dataset=train
```
### Prepare FAISS index files for dynamic DPR retrieval
```
python tools/prepare_faiss_index.py  \
    --csv_path ../data/ok-vqa/pre-extracted_features/passages/okvqa_full_corpus_title.csv \
    --output_dir  ../data/ok-vqa/pre-extracted_features/faiss/ok-vqa-passages-full-new-framework \
    --dpr_ctx_encoder_model_name /home/wl356/rds/rds-cvnlp-hirYTW1FQIw/wl356/Experiments/OKVQA_DPR_FullCorpus/train/saved_model/epoch6/item_encoder \
    --dpr_ctx_encoder_tokenizer_name /home/wl356/rds/rds-cvnlp-hirYTW1FQIw/wl356/Experiments/OKVQA_DPR_FullCorpus/train/saved_model/epoch6/item_encoder_tokenizer \
```

## Baseline models without DPR for retrieval
### RA-VQA-NoDPR (T5 baseline)
```
python main.py ../configs/baseline_T5.jsonnet \
    --mode train \
    --experiment_name OKVQA_RA-VQA-NoDPR  \
    --accelerator auto --devices auto  \
    --opts train.epochs=10  \
            train.batch_size=1  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=32  \
            train.lr=0.00006  \
            train.scheduler=linear
```

## Baseline models with DPR
For models using static DPR outputs, pre-trained DPR features (derived from "Generating Static Retrieval Results by Testing") should be configured at the config file.
Can override `data_loader.dataset_modules.module_dict.LoadPretrainedDPROutputForGoogleSearchPassage.config.pretrained_dpr_outputs` or simply change the path in `base_env.jsonnet`:
```
local pretrained_dpr_features = {
  "train": "/home/wl356/rds/rds-wjb31-nmt2020/wl356/Experiments/Knowledge_Retriever_DPR_dim_768_inbatch_negative_caption_FullCorpus_NewRun/test/test_evaluation/train_predictions.json",
  "test": "/home/wl356/rds/rds-wjb31-nmt2020/wl356/Experiments/Knowledge_Retriever_DPR_dim_768_inbatch_negative_caption_FullCorpus_NewRun/test/test_evaluation/test_predictions.json",
};
```
Then run the training script.

### TRiG
```
python main.py ../configs/TRiG.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_TRiG  \
    --accelerator auto --devices auto  \
    --opts train.epochs=10  \
            train.batch_size=4  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=8  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            data_loader.additional.num_knowledge_passages=5
```


### T5 baseline with Knowledge
Randomly pick a retrieved passage to append to the query
```
python main.py ../configs/baseline_T5_with_knowledge.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_RA-VQA-NoDPR_with_random_pick_passage  \
    --accelerator auto --devices auto  \
    --opts train.epochs=10  \
            train.batch_size=8  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=4  \
            train.lr=0.0001  \
            train.scheduler=linear  \
            data_loader.additional.num_knowledge_passages=5
```
With all passages in training:

The whole training set will be populated by `num_knowledge_passages` times. Each query is assoicated with `num_knowledge_passages` retrieved passages in training.
```
python main.py ../configs/baseline_T5_with_knowledge.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_RA-VQA-NoDPR_with_all_passages  \
    --accelerator auto --devices auto  \
    --opts train.epochs=10  \
            train.batch_size=8  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=4  \
            train.lr=0.0001  \
            train.scheduler=linear  \
            data_loader.additional.num_knowledge_passages=5  \
            data_loader.dataset_modules.module_dict.LoadPretrainedDPROutputForGoogleSearchPassage.option=default  \
            data_loader.dataset_type=OKVQADatasetWithAllPassages
```
To decide which output answer is selected, some methods can be chosen by --modules module_name:
```
if 'max_accumulated_scores' in self.config.model_config.modules:
    selected_answer = max(answer2score.items(), key=operator.itemgetter(1))[0]
elif 'max_average_scores' in self.config.model_config.modules:
    selected_answer = max(answer2score.items(), key=operator.itemgetter(1))[0]
elif 'max_average_scores_per_answer' in self.config.model_config.modules:
    for answer, score in answer2score.items():
        answer2score[answer] = score / answer_count[answer]
    selected_answer = max(answer2score.items(), key=operator.itemgetter(1))[0]
else:
    # majority vote
    selected_answer = most_frequent(answer_text_list)
```


## RAVQA framework
### Static retrieval
Retrieving from static DPR outputs
```
python main.py ../configs/RAVQA_static_retrieval.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_RAVQA_static_retrieval  \
    --opts train.epochs=10  \
            train.batch_size=4  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=8  \
            train.lr=0.00006  \
            train.scheduler=linear  \
            model_config.loss_ratio.additional_loss=1  \
            data_loader.additional.num_knowledge_passages=5
```

### RA-VQA-FrDPR
DPR is frozen during training
```
python main.py ../configs/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_RA-VQA-FrDPR_FullCorpus  \
    --accelerator auto --devices auto  \
    --modules freeze_question_encoder force_existence  \
    --opts train.epochs=10  \
            train.batch_size=4  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=8  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            data_loader.additional.num_knowledge_passages=5
```

### RA-VQA-NoPR
Only model predictions are used to train the retriever:
```
python main.py ../configs/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name RA-VQA-NoPR  \
    --accelerator auto --devices auto  \
    --modules force_existence  \
    --opts train.epochs=10  \
            train.batch_size=4  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=8  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            model_config.loss_ratio.additional_loss=1  \
            model_config.RAVQA_loss_type=NoPR  \
            data_loader.additional.num_knowledge_passages=5
```

### RA-VQA
Training with both Pseudo Relevance Labels and Model Predictions:
```
python main.py ../configs/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_RA-VQA_FullCorpus  \
    --accelerator auto --devices auto  \
    --modules force_existence  \
    --opts train.epochs=10  \
            train.batch_size=4  \
            valid.step_size=32  \
            valid.batch_size=4  \
            train.additional.gradient_accumulation_steps=8  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            model_config.loss_ratio.additional_loss=1  \
            model_config.RAVQA_loss_type=Approach6  \
            data_loader.additional.num_knowledge_passages=5
```
Testing Example:
```
python main.py ../configs/RAVQA.jsonnet  \
    --mode test  \
    --experiment_name OKVQA_RA-VQA_FullCorpus  \
    --accelerator auto --devices auto  \
    --modules force_existence  \
    --opts data_loader.additional.num_knowledge_passages=5  \
            test.load_model_path=../Experiments/OKVQA_RA-VQA_FullCorpus/train/saved_model/epoch_06.ckpt
```

### RA-VQA-NoCT
Customized Targets are not used to improve answer generation:
```
python main.py ../configs/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name RA-VQA-NoCT  \
    --accelerator auto --devices auto  \
    --opts train.epochs=10  \
            train.batch_size=4  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=8  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            model_config.loss_ratio.additional_loss=1  \
            model_config.RAVQA_loss_type=Approach6  \
            data_loader.additional.num_knowledge_passages=5
```

### RA-VQA on Wikipedia
Train RA-VQA with Wikipedia passages; The embeddings of Wikipedia passages are generated by the DPR paper.
```
python main.py ../configs/RAVQA_wikipedia.jsonnet  \
    --mode train  \
    --experiment_name RA-VQA_Wikipedia  \
    --accelerator auto --devices auto  \
    --modules force_existence  \
    --opts train.epochs=10  \
            train.batch_size=4  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=8  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            model_config.loss_ratio.additional_loss=1  \
            model_config.RAVQA_loss_type=Approach6  \
            data_loader.additional.num_knowledge_passages=5
```


