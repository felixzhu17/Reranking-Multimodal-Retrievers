from easydict import EasyDict
from ..utils.global_variables import register_to, register_func_to_registry, DataTransform_Registry
from ..utils.util import get_tokenizer
from ..utils.eval_recorder import EvalRecorder
from transformers import AutoTokenizer
import transformers
import copy
import pandas as pd
from torchvision.transforms import ColorJitter, ToTensor
from tqdm import tqdm
from typing import Dict, List
from collections.abc import Iterable, Mapping
from datasets import Dataset, DatasetDict, load_dataset
import functools
from pathlib import Path


def register_transform(fn):
    register_func_to_registry(fn, DataTransform_Registry)
    def _fn_wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return _fn_wrapper

def keep_ds_columns(ds, keep_cols):
    all_colummns = set(ds.features.keys())
    remove_cols = list(all_colummns - set(keep_cols))
    return ds.remove_columns(remove_cols)

def register_transform_functor(cls):
    register_func_to_registry(cls, DataTransform_Registry)
    return cls

class BaseTransform():
    """
    Most general functor definition
    """
    def __init__(
        self,
        *args,
        name=None,
        input_mapping: Dict=None,
        output_mapping: Dict=None,
        use_dummy_data=False,
        global_config=None,
        **kwargs
        ):
        self.name = name or self.__class__.__name__
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.use_dummy_data = use_dummy_data
        self.global_config = global_config

    # @classmethod 
    # def __init__subclass__(cls, **kwargs):
    #     super().__init_subclass__(*args, **kwargs)
    #     register_func_to_registry(cls.__name__, DataTransform_Registry)

    def __call__(self, data, *args, **kwargs):
        preprocessed_data = self._preprocess(data) # any preprocessing should be handled here
        # mapped_data = self._apply_mapping(preprocessed_data, self.input_mapping)
        self._check_input(preprocessed_data)

        # output_data = self._call(**mapped_data) if self.input_mapping else self._call(mapped_data)
        output_data = self._call(preprocessed_data)
        # output_mapped_data = self._apply_mapping(output_data, self.output_mapping)
        self._check_output(output_data)

        return output_data
        
        # _call will expand keyword arguments from data if mapping [input_col_name : output_col_name] is given
        # otherwise received whole data
    
    # def _apply_mapping(self, data, in_out_col_mapping):
    #     """
    #     IMPORTANT: when input_mapping is given, data will be transformed into EasyDict
    #     """
    #     if in_out_col_mapping is None:
    #         return data
    #     assert isinstance(data, Mapping), f"input feature mapping cannot be performed on non-Mapping type objects!"
    #     mapped_data = {}
    #     for input_col, output_col in in_out_col_mapping.items():
    #         mapped_data[output_col] = data[input_col]
    #     return EasyDict(mapped_data)



    def _check_input(self, data):
        """
        Check if the transformed can be applied on data. Override in subclasses
        No constraints by default
        """
        return True
    
    def _check_output(self, data):
        """
        Check if the transformed data fulfills certain conditions. Override in subclasses
        No constraints by default
        """
        return True
        
    
    def _preprocess(self, data):
        """
        Preprocess data for transform.
        """
        return data

    def setup(self, *args, **kwargs):
        """
        setup any reusable resources for the transformed. Will be called before __apply__()
        """
        raise NotImplementedError(f"Must implement {self.name}.setup() to be a valid transform")

    def _call(self, data, *args, **kwargs):
        raise NotImplementedError(f'Must implement {self.name}._call() to be a valid transform')

class RowWiseTransform(BaseTransform):
    """
    Transform each element row-by-row
    """
    # @classmethod 
    # def __init__subclass__(cls, **kwargs):
    #     super().__init_subclass__(*args, **kwargs)
    #     register_func_to_registry(cls.__name__, DataTransform_Registry)

    def __call__(self, data, *args, **kwargs):
        preprocesed_data = self._preprocess(data) # any preprocessing should be handled here
        self._check_input(preprocesed_data)
        for row_n, row_data in enumerate(preprocesed_data):
            mapped_data = self._apply_mapping(row_data, self.input_mapping)
            output_data = self._call(row_n, **mapped_data) if self.input_mapping else self._call(row_n, mapped_data)
            output_mapped_data = self._apply_mapping(output_data, self.output_mapping)
        self._check_output(output_mapped_data)
        return output_mapped_data

    def _call(self, row_n, row_data):
        raise NotImplementedError(f'Must implement {self.name}._call() to be a valid transform')

    def _check_input(self, data):
        return isinstance(data, Iterable)

class HFDatasetTransform(BaseTransform):
    """
    Transform using HuggingFace Dataset utility
    """
    # @classmethod 
    # def __init__subclass__(cls, **kwargs):
    #     super().__init_subclass__(*args, **kwargs)
    #     register_func_to_registry(cls.__name__, DataTransform_Registry)
    def setup(self, rename_col_dict=None, splits_to_process=['train', 'test', 'validation'], *args, **kwargs):
        """
        setup any reusable resources for the transformed. Will be called before __call__()
        For HFDataset, add rename_col_dict for renaming columns conveniently
        """
        self.rename_col_dict = rename_col_dict
        self.splits_to_process = splits_to_process

    def _check_input(self, data):
        return isinstance(data, Dataset) or isinstance(data, DatasetDict)
    
    # def _apply_mapping(self, data, in_out_col_mapping):
    #     if not in_out_col_mapping:
    #         return data
    #     if isinstance(data, DatasetDict):
    #         mapped_data = {out_col_name: data[in_col_name] for in_col_name, out_col_name in in_out_col_mapping.items()}
    #         return mapped_data
    #     else: # data is DatasetDict
    #         data = data.rename_columns(in_out_col_mapping)
    #         mapped_data = keep_ds_columns(data, list(in_out_col_mapping.values()))
    #         return mapped_data
    
def tokenize_function(tokenizer, field, **kwargs):
    def tokenize_function_wrapped(example):
        return tokenizer.batch_encode_plus(example[field], **kwargs)
    return tokenize_function_wrapped

@register_transform_functor
class HFDatasetTokenizeTransform(HFDatasetTransform):
    def setup(self, rename_col_dict, tokenizer_config: EasyDict, tokenize_fields_list: List, splits_to_process=['train', 'test', 'validation']):
        super().setup(rename_col_dict)
        self.tokenize_fields_list = tokenize_fields_list
        self.tokenizer = get_tokenizer(tokenizer_config)
        self.tokenize_kwargs = tokenizer_config.get(
            'tokenize_kwargs', 
            {
             'batched': True,
             'load_from_cache_file': False,
             'padding': 'max_length',
             'truncation': True
             }
        )
        self.splits_to_process = splits_to_process

    def _call(self, dataset):
        results = {}
        for split in ['train', 'test', 'validation']:
            # ds = dataset[split].select((i for i in range(100)))
            if split not in dataset:
                continue
            ds = dataset[split]
            for field_name in self.tokenize_fields_list:
                ds = ds\
                .map(tokenize_function(self.tokenizer, field_name, **self.tokenize_kwargs), batched=True, load_from_cache_file=False) \
                .rename_columns({
                    'input_ids': field_name+'_input_ids',
                    'attention_mask': field_name+'_attention_mask',
                })
            ds = ds.rename_columns(self.rename_col_dict)
            results[split] = ds
        return results

@register_transform_functor
class LoadHFDataset(BaseTransform):
    def setup(self, dataset_name, dataset_path=None, fields=[]):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.fields = fields
    
    def _call(self, data):
        dataset_url = None
        if self.dataset_path:
            dataset_url = f"{self.dataset_path}/{self.dataset_name}"
        else:
            dataset_url = self.dataset_name
        hf_ds = load_dataset(dataset_url)
        return hf_ds

@register_transform_functor
class SplitHFDatasetToTrainTestValidation(HFDatasetTransform):
    def setup(self, test_size, train_test_split_kwargs={}, valid_size=None):
        self.test_size = test_size
        self.valid_size = valid_size
        self.test_valid_total_size = self.test_size + self.valid_size if self.valid_size else self.test_size
        self.train_test_split_kwargs = train_test_split_kwargs
        # assert self.test_valid_total_size <= 1.0
    
    def _call(self, data, *args, **kwargs):
        train_ds = data['train']
        train_dict = train_ds.train_test_split(self.test_valid_total_size, **self.train_test_split_kwargs)
        train_ds = train_dict['train']
        test_ds = train_dict['test']
        if self.valid_size is not None:
            test_valid_dict = train_dict['test'].train_test_split(self.valid_size / self.test_valid_total_size, **self.train_test_split_kwargs)
            test_ds = test_valid_dict['train']
            valid_ds = test_valid_dict['test']

        res_dataset_dict = DatasetDict({
            'train': train_ds,
            'test': test_ds,
            'validation': valid_ds
        })
        print("Split into train/test/validation:",res_dataset_dict)
        return res_dataset_dict

@register_transform_functor
class DummyTransform(BaseTransform):
    def setup(self):
        pass

    def _call(self, data):
        return data

@register_transform_functor
class GetEvaluationRecorder(BaseTransform):
    def setup(self, base_dir=None, eval_record_name='test-evaluation', recorder_prefix='eval_recorder', file_format='json'):
        self.eval_record_name = eval_record_name
        self.recorder_prefix = recorder_prefix
        self.base_dir = base_dir or self.global_config['test_dir']
        self.file_format = file_format
    
    def _call(self, data):
        if data is not None:
            return data # short cut for validation pipeline
        eval_recorder = EvalRecorder.load_from_disk(self.eval_record_name, self.base_dir, file_prefix=self.recorder_prefix, file_format=self.file_format)
        return eval_recorder
    
@register_transform_functor
class MergeAllEvalRecorderAndSave(BaseTransform):
    def setup(
        self, 
        base_dir = None, 
        eval_record_name='merged-test-evaluation', 
        eval_recorder_prefix='merged',
        recorder_prefix='eval_recorder', 
        file_format='json', 
        save_recorder=True
    ):
        self.eval_record_name = eval_record_name
        self.eval_recorder_prefix = eval_recorder_prefix
        self.recorder_prefix = recorder_prefix
        self.base_dir = base_dir
        self.file_format = file_format
        self.save_recorder = save_recorder

    def _call(self, data):
        """_summary_

        :param data: _description_
        """
        eval_recorder = data[0]
        # self.base_dir = self.base_dir or str(Path(eval_recorder.save_dir).parent)
        if len(data) > 1:
            eval_recorder.merge(data[1:]) # merge all evaluation results
        if self.eval_recorder_prefix is not None:
            self.eval_record_name = f"{self.eval_recorder_prefix}-{eval_recorder.name}"
        eval_recorder.rename(self.eval_record_name)
        # eval_recorder.rename(self.eval_record_name, new_base_dir=self.base_dir)
        eval_recorder.save_to_disk(self.recorder_prefix, file_format=self.file_format)
        print("Evaluation recorder merged and saved to", eval_recorder.save_dir)
        return eval_recorder
        