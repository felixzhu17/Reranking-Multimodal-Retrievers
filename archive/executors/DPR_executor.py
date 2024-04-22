import math
import time
import os
import sys
import scipy
import datetime
import numpy as np
import json
import operator

import wandb
import torch
import torch.nn.functional as F
import pandas as pd
import os.path
from runway_for_ml.executors.base_executor import BaseExecutor
from runway_for_ml.utils.global_variables import register_executor
from runway_for_ml.utils.util import batch_depad
from torch.utils.data import DataLoader
from runway_for_ml.configs.configuration import (
    DataPipelineConfig,
    ModelConfig,
)


from pprint import pprint
from tqdm import tqdm
from easydict import EasyDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import CheckpointIO

from src.metrics import MetricsProcessor
from src.models.retriever.retriever_dpr import RetrieverDPR
from src.models.retriever.visual_dpr import VisualDPRForPretraining, VisualDPRForRetrieval, VisualDPRWithMultiModalDocs, VisualDPRWithMultiModalDocsWithOnlyImages
from src.utils.dirs import *
import faiss
import wandb
import datasets
import copy

import logging
logger = logging.getLogger(__name__)


@register_executor
class DPRExecutor(BaseExecutor, MetricsProcessor):
    def __init__(self,
        data_pipeline_config: DataPipelineConfig,
        model_config: ModelConfig,
        mode, # train/infer/eval
        train_config={},
        test_config={},
        log_file_path=None,
        use_data_node=None,
        tokenizer=None,
        global_config=None,
        *args, **kwargs
        ):
        super().__init__(data_pipeline_config, model_config, mode, train_config=train_config, test_config=test_config, log_file_path=log_file_path, global_config=global_config, use_data_node=use_data_node, *args, **kwargs)
        self.multimodal_docs = model_config.get("multimodal_docs", False)
        self.tmp_index = None
        self.validation_step_outputs = []
        self.test_step_outputs = []

    
    def _init_model(self, model_config): 
        """Initialize self.model

        Args:
            model_config (dict): contains key-values for model configuration
        """
        # super()._init_model(model_config) # alternatively, use the default implementation in super()._init_model()
        
        ModelClass = globals()[self.model_config.ModelClass]
        self.model = ModelClass(config=self.config)
        
    
    def prepare_data(self):
        super().prepare_data()
        print(len(self.prepared_data.vqa_data_with_dpr_output.get('lookup', {})))
        if len(self.prepared_data.vqa_data_with_dpr_output.get('lookup', {})) == 0:
            self.prepared_data.vqa_data_with_dpr_output.lookup = {}
            print("Loading lookup table...")
            for data_split in ['train', 'val']:
                ds_split = self.prepared_data.vqa_data_with_dpr_output[data_split]
                lookup_dict = ds_split.to_pandas().set_index("question_id", drop=False).to_dict(orient="index")
                self.prepared_data.vqa_data_with_dpr_output.lookup.update(lookup_dict)
            print("Done loading lookup table.")
    
    def setup(self, stage):
        super().setup(stage)
        self.prepared_data = self.dp.get_data([self.use_data_node], explode=True)
        
        self.data_loaders = self.prepared_data['data_loaders']

        self.train_dataloaders = list(self.data_loaders['train'].values())
        self.valid_dataloaders = list(self.data_loaders['valid'].values())
        self.test_dataloaders = list(self.data_loaders['test'].values())

        self.tokenizers = self.prepared_data['tokenizers']

        self.tokenizer = self.tokenizers['tokenizer']
        self.decoder_tokenizer = self.tokenizers['decoder_tokenizer']

        self.model.resize_token_embeddings(len(self.tokenizer), len(self.decoder_tokenizer))

        checkpoint_to_load = self.global_config.train.get('load_model_path', '')
    
        if not checkpoint_to_load or checkpoint_to_load == '':
            logger.warning("No checkpoint found. First time to train...")
        else:
            # We manually load the state dict
            logger.info(f"Loading from {checkpoint_to_load}")
            state_dict_from_ckpt = torch.load(checkpoint_to_load, map_location=self.device)['state_dict']
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in state_dict_from_ckpt.items() if k in model_dict}
            logger.info(f"Load the following parameters from the given checkpoint: {pretrained_dict.keys()}")
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            
            # When multimodal docs are enabled, load the vision_projection into doc_vision_projection
            if self.multimodal_docs:
                doc_vision_projection_dict = {k.replace('vision_projection', 'doc_vision_projection'): v for k, v in state_dict_from_ckpt.items() if 'vision_projection' in k}
                model_dict.update(doc_vision_projection_dict)
                logger.info(f"Load the following parameters from the given checkpoint: {doc_vision_projection_dict.keys()}")
            
            # 3. load the new state dict
            self.load_state_dict(model_dict)
        
        
    def training_step(self, sample_batched, batch_idx):
        train_batch = EasyDict({
            'input_ids': sample_batched['input_ids'].to(self.device),
            'attention_mask': sample_batched['attention_mask'].to(self.device),
            'labels': sample_batched['labels'].to(self.device),
            'item_input_ids': sample_batched['decoder_input_ids'].to(self.device),
            'item_attention_mask': sample_batched['decoder_input_attention_mask'].to(self.device),
        })

        # if there is vision input, add it to the batch
        pixel_values = sample_batched.get('pixel_values', None)
        if pixel_values is not None:
            train_batch['pixel_values'] = pixel_values.to(self.device)
        
        image_features = sample_batched.get('image_features', None)
        if image_features is not None:
            train_batch['image_features'] = image_features.to(self.device)

        item_image_features = sample_batched.get('item_image_features', None)
        if item_image_features is not None:
            train_batch['item_image_features'] = item_image_features.to(self.device)

        forward_results = self.model(**train_batch)
        batch_loss = forward_results.loss

        # log the current learning rate from shedulers
        current_lrs = self.scheduler.get_last_lr()
        for index, current_lr in enumerate(current_lrs):
            self.log(f"train/lr[{index}]", current_lr, prog_bar=True, on_step=True, logger=True, sync_dist=True)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train/loss", batch_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        data_to_return = {
            'loss': batch_loss,
        }
        return data_to_return
    
    def validation_step(self, sample_batched, batch_idx, dataloader_idx=0):
        pred =  self._compute_query_embeddings_step(sample_batched, batch_idx)
        self.validation_step_outputs.append(pred)
        return pred
    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        if len(validation_step_outputs) == 0:
            return None
        for i in range(len(self.val_dataloader())):
            if len(self.val_dataloader()) == 1:
                validation_step_output = validation_step_outputs
            else:
                validation_step_output = validation_step_outputs[i]
            
            log_dict = self.evaluate_outputs(validation_step_output, self.val_dataloader()[i], self.val_dataloader_names[i])
            self.logging_results(log_dict, prefix=self.val_dataloader_names[i])
        
        # when validation finishes, remove tmp index
        if "freeze_dpr_doc_encoder" not in self.model_config.modules:
            # when validation finishes, remove tmp index
            self.tmp_index = None

        self.validation_step_outputs.clear()

        return None
    
    def test_step(self, sample_batched, batch_idx, dataloader_idx=0):
        pred =  self._compute_query_embeddings_step(sample_batched, batch_idx)
        self.test_step_outputs.append(pred)
        return pred
    
    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        logger.info("reading global step of the checkpoint...")
        if self.trainer.ckpt_path is not None:
            self.ckpt_global_step = torch.load(self.trainer.ckpt_path, map_location=torch.device('cpu'))['global_step']
        elif self.global_config.train.get('load_model_path', '') != "":
            self.ckpt_global_step = torch.load(self.global_config.train.load_model_path, map_location=torch.device('cpu'))['global_step']
        else:
            self.ckpt_global_step = self.global_step

        self.save_HF_model()
        for i in range(len(self.test_dataloader())):
            if len(self.test_dataloader()) == 1:
                test_step_output = test_step_outputs
            else:
                test_step_output = test_step_outputs[i]
            
            log_dict = self.evaluate_outputs(test_step_output, self.test_dataloader()[i], self.test_dataloader_names[i])
            self.logging_results(log_dict, prefix=f"{self.config.test_suffix}_{self.test_dataloader_names[i]}")
        # when testing finishes, remove tmp index
        self.tmp_index = None

        self.test_step_outputs.clear()
        return None

        

    def _compute_query_embeddings_step(self, sample_batched, batch_idx):
        """
        This function is shared by both valid and test
        """
        test_batch = EasyDict({
            'input_ids': sample_batched['input_ids'].to(self.device),
            'attention_mask': sample_batched['attention_mask'].to(self.device),
        })

        # if there is vision input, add it to the batch
        pixel_values = sample_batched.get('pixel_values', None)
        if pixel_values is not None:
            test_batch['pixel_values'] = pixel_values.to(self.device)
        
        image_features = sample_batched.get('image_features', None)
        if image_features is not None:
            test_batch['image_features'] = image_features.to(self.device)
        
        # batch_size x hidden_states
        query_emb = self.model.generate_query_embeddings(**test_batch)
        
        data_to_return = {
            'btach_idx': batch_idx,
            'query_emb': query_emb.cpu(),
            'question_ids': sample_batched['question_ids'],
            'answers': sample_batched['answers'],
            'pos_item_ids': sample_batched['pos_item_ids'],
        }

        return data_to_return
    
    
    def prepare_item_embeddings(self, current_data_loader, mode):

        # Decide which corpus to use for evaluating the VQA queries
        if self.model_config.full_corpus_in_testing:
            passage_id2doc = self.prepared_data.passages.id2doc 
        else:
            passage_id2doc = self.prepared_data.passages.id2doc_train
        
        n_items = len(passage_id2doc)

        if self.trainer.state.stage in ['sanity_check']:
            # sanity check
            logging.warning('No steps have been taken. Reducing number of items to speed up the sanity check.')
            # n_items = 100

        i_batch_size = self.config[mode].batch_size
        
        n_item_batchs = n_items // i_batch_size + 1

        # Create mapping between matrix indice and passage ids
        # Using only train passage corpus
        passage_index2id = {index:passage_id for index, passage_id in enumerate(passage_id2doc.keys()) if index < n_items}
        decoder_input_modules = self.model_config.decoder_input_modules.module_list
        passage_contents = []
        multimodal_docs = self.model_config.get('multimodal_docs', False)
        if multimodal_docs:
            passage_image_features = []
        
        for passage_id in passage_id2doc.keys():
            sample = EasyDict(passage_content=passage_id2doc[passage_id], passage_id=passage_id)
            parsed_data = current_data_loader.dataset.parse_modules(sample, decoder_input_modules, type='decoder_input')
            passage_contents.append(parsed_data.text_sequence)
            if multimodal_docs:
                passage_image_features.append(parsed_data.img.image_features)
        
        
        logger.info(f'Generating embeddings for items; there are {n_items} items.')
        i_count = 0
        item_embeddings = []
        for i_batch_id in tqdm(range(n_item_batchs)):
            i_start = i_batch_id * i_batch_size
            i_end = min((i_batch_id + 1) * i_batch_size, n_items)

            passage_contents_batch = passage_contents[i_start:i_end]
            # print(passage_contents_batch)
            # Encode this batch of data
            item_encoding = self.decoder_tokenizer(passage_contents_batch,
                                padding='longest',
                                max_length=self.model_config.max_decoder_source_length,
                                truncation=True,
                                return_tensors="pt")
            
            item_input_ids, item_attention_mask = item_encoding.input_ids, item_encoding.attention_mask
            test_batch = EasyDict({
                'input_ids': item_input_ids.to(self.device),
                'attention_mask': item_attention_mask.to(self.device),
            })

            if multimodal_docs:
                passage_image_features_batch = passage_image_features[i_start:i_end]    
                item_image_features = torch.FloatTensor(np.stack(passage_image_features_batch))
                test_batch['item_image_features'] = item_image_features.to(self.device)
            
            # batch_size x hidden_states
            item_emb = self.model.generate_item_embeddings(**test_batch)
            for x in item_emb:
                item_embeddings.append(x.cpu().detach().numpy())
            
            i_count += item_emb.shape[0]

        assert i_count == n_items
        
        # Update: use faiss
        hidden_size = item_embeddings[0].shape[-1]
        print("hidden size", hidden_size)

        dataset = pd.DataFrame.from_dict(
            self.prepared_data.passages.id2doc, orient='index', columns=['passage_content'])
        
        dataset['passage_id'] = dataset.index
        dataset = datasets.Dataset.from_pandas(dataset)
        dataset = dataset.select(range(n_items))
        dataset = dataset.add_column("embeddings", item_embeddings)

        if self.trainer.state.stage == 'test' and self.global_rank==0:
            # Save the dataset
            save_path = os.path.join(self.config.test_dir, 'step_{}'.format(self.global_step))
            create_dirs([save_path])
            dataset_path = os.path.join(save_path, "dataset")
            dataset.save_to_disk(dataset_path)
            logger.info(f"Saved dataset to {dataset_path}")
        
        if "exhaustive_search_in_testing" in self.model_config.modules:
            faiss_index = faiss.IndexFlatIP(hidden_size)
        else:
            faiss_index = faiss.IndexHNSWFlat(hidden_size, 128, faiss.METRIC_INNER_PRODUCT)

        # in testing mode, save the generated embeddings
        if self.trainer.state.stage == 'test' and self.global_rank==0:
            
            save_path = os.path.join(self.config.test_dir, 'step_{}'.format(self.global_step))
            create_dirs([save_path])
            
            index_path = os.path.join(save_path, "dataset_hnsw_index.faiss")
            logger.info(f'saving embedding files into {index_path}')
            dataset_copy = copy.deepcopy(dataset)
            to_save_index = faiss.IndexHNSWFlat(hidden_size, 128, faiss.METRIC_INNER_PRODUCT)
            dataset_copy.add_faiss_index("embeddings", custom_index=to_save_index)
            dataset_copy.get_index("embeddings").save(index_path)

        item_embeddings = np.stack(item_embeddings, 0)
        faiss_index.add(item_embeddings)

        self.tmp_index = {
            "faiss_index": faiss_index,
            "passage_index2id": passage_index2id,
            "passage_contents": passage_contents,
        }


    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name, mode='test'):
        # Batching every validation step outputs
        # n_queries x hidden_size
        
        query_embeddings = []
        question_ids = []
        pos_item_ids = []
        for step_output in step_outputs:
            query_embeddings.append(step_output['query_emb'])
            question_ids += step_output['question_ids']
            pos_item_ids.extend(step_output['pos_item_ids'])
        
        query_embeddings = torch.cat(query_embeddings, dim=0)
        n_queries = query_embeddings.shape[0]
        hidden_size = query_embeddings.shape[1]

        ##################################
        ##    Generate embeds for items ##
        ##################################
        
        if self.tmp_index is None:
            # When item embeddings are not indexed, call the function
            # this will not be called more than once during a validation step
            # which reduces the time needed for validating more than one datasets
            logger.info("No tmp exists, start building indexes...")
            self.prepare_item_embeddings(current_data_loader, mode)
        else:
            logger.info("reusing pre-computed indexes...")

        faiss_index = self.tmp_index['faiss_index']
        passage_index2id = self.tmp_index['passage_index2id']
        passage_contents = self.tmp_index['passage_contents']
        

        ##################################
        ##    Search Index              ##
        ##################################

        Ks = self.model_config.Ks

        # Create mapping between matrix indice and question ids
        question_index2id = {index:question_id for index, question_id in enumerate(question_ids)}
        assert len(question_index2id) == n_queries
        logger.info(f'There are {n_queries} queries.')

        # Search the index file
        search_res = faiss_index.search(query_embeddings, k=max(Ks))

        batch_result = []
        for query_id, return_scores, return_passage_index in zip(range(len(query_embeddings)), search_res[0], search_res[1]):
            question_id = question_index2id[query_id]
            # print(question_id, return_scores, return_passage_index)
            # Retrieve content from passages
            top_ranking_passages = [{
                'passage_index': i,
                'passage_id': passage_index2id[i],
                'content': passage_contents[i],
                'score': float(return_scores[index]),
            } for index, i in enumerate(return_passage_index)]
            
            query_item = self.prepared_data.vqa_data_with_dpr_output.lookup[str(question_id)]
            answers = query_item.answers
            gold_answer = query_item.gold_answer

            batched_data = {
                "question_id": question_id,
                "top_ranking_passages": top_ranking_passages,
                "answers": answers,
                "gold_answer": gold_answer,
            }
            if query_item.get("pos_item_ids", None) is not None:
                batched_data["pos_item_ids"] = list(query_item.pos_item_ids)
            if query_item.get("related_item_ids", None) is not None:
                batched_data["related_item_ids"] = list(query_item.related_item_ids)

            batch_result.append(batched_data)

        if self.config.args.log_prediction_tables_with_images:
            artifact = self.wandb_logger.experiment.use_artifact(self.config.args.wandb_artifacts, type='dataset')
        
        # Log results
        columns=["question_id", "input_image", "image_key", "question", "caption", "answers", "gold_answer"]  \
                    + ['p_{}'.format(i) for i in range(max(Ks))]
        test_table = wandb.Table(columns=columns)
        
        to_write_data = {
            'output': [],
        }
        for re in tqdm(batch_result):
            to_write_data['output'].append(re)
            question_id = re['question_id']
            knowledge_item = self.prepared_data.vqa_data_with_dpr_output.lookup[str(question_id)]
            if knowledge_item.get('img_file_name', None) is None:
                knowledge_item['img_file_name'] = knowledge_item['image_path'].split('/')[-1]
            if knowledge_item.get('img_key', None) is None:
                knowledge_item['img_key'] = knowledge_item['image_id']
            if knowledge_item.get('img_caption', None) is None:
                image_caption = ""
            else:
                if isinstance(knowledge_item['img_caption'], str):
                    image_caption = knowledge_item['img_caption']
                else:
                    image_caption = knowledge_item['img_caption']['caption']
                
            table_entry = [
                knowledge_item['question_id'],
                knowledge_item['img_file_name'],
                knowledge_item['img_key'],
                knowledge_item['question'],
                image_caption,
                list(knowledge_item['answers']),
                knowledge_item['gold_answer'],
            ]

            if self.config.args.log_prediction_tables_with_images:
                # Replace image keys with real images
                input_image_file_name = knowledge_item['img_file_name']
                input_image = artifact.get(input_image_file_name)
                if input_image is None:
                    input_image = artifact.get(input_image_file_name)
                
                table_entry[1] = input_image

            
            table_entry+=[p['content'] for p in re['top_ranking_passages']]
            # print(table_entry)
            test_table.add_data(*table_entry)
        
        ##############################
        ##    Compute Metrics       ##
        ##############################
        data_used_for_metrics = EasyDict(
            mode=mode,
            epoch=self.current_epoch,
            batch_retrieval_result=batch_result,
            Ks=Ks,
        )

        log_dict = self.compute_metrics(data_used_for_metrics)

        log_dict.artifacts.test_table = test_table
        log_dict.artifacts.to_write_data = to_write_data
        return log_dict

    
    def logging_results(self, log_dict, prefix='test'):
        
        ### Add test results to wandb / tensorboard
        metrics_to_log = EasyDict()
        artifacts_to_log = log_dict.artifacts
        wandb_artifacts_to_log = dict()
        # Refractor the column names
        for metric, value in log_dict.metrics.items():
            metrics_to_log[f'{prefix}/{metric}'] = value
        
        # include other artifacts / metadata
        metrics_to_log[f'{prefix}/epoch'] = self.current_epoch
        wandb_artifacts_to_log.update({
            f"predictions/step_{self.global_step}_MODE({self.config.mode})_SET({prefix})_rank({self.global_rank})": log_dict.artifacts['test_table']
        })
        pprint(metrics_to_log)
        pprint(wandb_artifacts_to_log)

        logger.info(f"Evaluation results [{self.trainer.state.stage}]: {metrics_to_log}")
        
        if self.trainer.state.stage in ['sanity_check']:
            logging.warning('Sanity check mode, not saving to loggers.')
            return
        
        # Add to loggers
        for metric, value in metrics_to_log.items():
            if type(value) in [float, int, np.float64]:
                self.log(metric, float(value), logger=True, sync_dist=True)
            else:
                logger.info(f'{metric} is not a type that can be logged, skippped.')
        
        # Call wandb to log artifacts; remember to use commit=False so that the data will be logged
        #       with other metrics later.
        if self.config.args.log_prediction_tables:
            self.wandb_logger.experiment.log(wandb_artifacts_to_log, commit=False)
        
        if self.config.mode == "test":
            from utils.numpy_encoder import NpEncoder
            # Save predictions to files for DPR-based VQA systems
            json_path = os.path.join(self.config.test_dir, '{}_predictions.json'.format(prefix.replace("/", "_")))
            with open(json_path, 'w') as json_f:
                json.dump(artifacts_to_log.to_write_data, json_f, indent=4, cls=NpEncoder)
                logger.info('Predictions have been saved to {}'.format(json_path))
        

    def save_HF_model(self):
        '''
        Save models with the Huggingface built-in save_pretrained() function.
        The checkpoints can be loaded by a RAG-like system.
        '''
        if self.global_rank != 0:
            logger.info('global rank is not 0, skip saving models')
            return
        logger.info('Saving model in the Huggingface format...')
        path_save_model = os.path.join(self.config.ckpt_dir, 'step_{}'.format(self.global_step))
        self.model.query_encoder.save_pretrained(os.path.join(path_save_model, 'query_encoder'))
        self.tokenizer.save_pretrained(os.path.join(path_save_model, 'query_encoder_tokenizer'))
        self.model.item_encoder.save_pretrained(os.path.join(path_save_model, 'item_encoder'))
        self.decoder_tokenizer.save_pretrained(os.path.join(path_save_model, 'item_encoder_tokenizer'))
        logger.info('Model has been saved to {}'.format(path_save_model))