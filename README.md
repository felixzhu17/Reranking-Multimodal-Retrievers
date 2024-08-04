# Reranking Multimodal Retrievers
This is the research repository of Reranking Multimodal Retrievers for VQA.

The author is Felix Zhu, MPhil student of the Department of Engineering, University of Cambridge.

**This repository is not-for-distribution in its current form.**


## Overview
The training and testing are backboned by pytorch-lightning. The pre-trained Transformer models are from Huggingface-transformers. The training platform is Pytorch.

### Structure
Extension of [https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering/tree/main](https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering/tree/main).

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

Install libraries:
```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.38.2
conda install -c pytorch faiss-gpu -y
pip install setuptools==59.5.0
pip install wandb pytorch-lightning==2.2.1 jsonnet easydict pandas scipy opencv-python fuzzywuzzy scikit-image matplotlib timm scikit-learn sentencepiece tensorboard datasets
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
pip install notebook
```


