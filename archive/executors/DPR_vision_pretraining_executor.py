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
from src.models.retriever.visual_dpr import VisualDPRForPretraining
from src.utils.dirs import *
import faiss
import wandb
import pickle

import logging
logger = logging.getLogger(__name__)

from src.executors.DPR_executor import DPRExecutor

@register_executor
class DPRVisionPretrainingExecutor(DPRExecutor):
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
        
        self.tmp_index = None

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
            'passage_ids': sample_batched['passage_ids'],
            'pos_item_ids': sample_batched['pos_item_ids'],
            'neg_item_ids': sample_batched['neg_item_ids'],
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
            # n_items = 500

        i_batch_size = self.config[mode].batch_size
        
        n_item_batchs = n_items // i_batch_size + 1

        # Create mapping between matrix indice and passage ids
        # Using only train passage corpus
        passage_index2id = {index:passage_id for index, passage_id in enumerate(passage_id2doc.keys()) if index < n_items}
        decoder_input_modules = self.model_config.decoder_input_modules.module_list
        passage_contents = []
        for passage_id in passage_id2doc.keys():
            sample = EasyDict(passage_content=passage_id2doc[passage_id])
            parsed_data = current_data_loader.dataset.parse_modules(sample, decoder_input_modules, type='decoder_input')
            passage_contents.append(parsed_data.text_sequence)
        
        
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
            
            # batch_size x hidden_states
            item_emb = self.model.generate_item_embeddings(**test_batch)
            for x in item_emb:
                item_embeddings.append(x.cpu().detach().numpy())
            
            i_count += item_emb.shape[0]

        assert i_count == n_items
        
        # Update: use faiss
        hidden_size = item_embeddings[0].shape[-1]
        print("hidden size", hidden_size)
        
        if "exhaustive_search_in_testing" in self.model_config.modules:
            faiss_index = faiss.IndexFlatIP(hidden_size)
        else:
            faiss_index = faiss.IndexHNSWFlat(hidden_size, 128, faiss.METRIC_INNER_PRODUCT)

        item_embeddings = np.stack(item_embeddings, 0)
        faiss_index.add(item_embeddings)

        self.tmp_index = {
            "faiss_index": faiss_index,
            "passage_index2id": passage_index2id,
            "passage_contents": passage_contents,
        }

        index_path = os.path.join(self.config.ckpt_dir, "index.pkl")
        if not os.path.exists(self.config.ckpt_dir):
            os.makedirs(self.config.ckpt_dir)
        with open(index_path, 'wb') as f:
            # A new file will be created
            try:
                pickle.dump(self.tmp_index, f)
            except Exception as e:
                logger.error(f"saving failed! {e}")


    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name, mode='test'):
        # Batching every validation step outputs
        # n_queries x hidden_size
        
        query_embeddings = []
        question_ids = []
        pos_item_ids = []
        neg_item_ids = []
        questions = []
        for step_output in step_outputs:
            query_embeddings.append(step_output['query_emb'])
            question_ids += step_output['question_ids']
            pos_item_ids.extend(step_output['pos_item_ids'])
            neg_item_ids.extend(step_output['neg_item_ids'])
            questions.append("") # empty for now
        
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

        for query_id, return_scores, return_passage_index, pos_ids, neg_ids in zip(range(len(query_embeddings)), search_res[0], search_res[1],
                                                        pos_item_ids, neg_item_ids):
            question_id = question_index2id[query_id]
            # Retrieve content from passages
            top_ranking_passages = []
            for index, i in enumerate(return_passage_index):
                if i >= 0:
                    top_ranking_passages.append({
                        'passage_index': i,
                        'passage_id': passage_index2id[i],
                        'content': passage_contents[i],
                        'score': float(return_scores[index]),
                    })
                else:
                    top_ranking_passages.append(top_ranking_passages[-1])

            query_item = self.prepared_data.vqa_data.lookup[str(question_id)]
            pos_item_contents = [self.prepared_data.passages.id2doc[pos_id] for pos_id in pos_ids]

            batch_result.append({
                "question_id": question_id,
                "top_ranking_passages": top_ranking_passages,
                "pos_item_ids": pos_ids,
                "neg_item_ids": neg_ids,
                "pos_item_contents": pos_item_contents,
            })

        if self.config.args.log_prediction_tables_with_images:
            artifact = self.wandb_logger.experiment.use_artifact(self.config.args.wandb_artifacts, type='dataset')
        
        # Log results
        columns=["question_id", "input_image", "image_key",  "pos_item_ids", "pos_item_contents"]  \
                    + ['p_{}'.format(i) for i in range(max(Ks))]
        test_table = wandb.Table(columns=columns)
        
        to_write_data = {
            'output': [],
        }
        for re in tqdm(batch_result):
            to_write_data['output'].append(re)
            question_id = re['question_id']
            knowledge_item = self.prepared_data.vqa_data.lookup[str(question_id)]

            pos_item_contents = [self.prepared_data.passages.id2doc[pos_id] for pos_id in pos_ids]
            table_entry = [
                knowledge_item['img_id'],
                knowledge_item['img_path'],
                knowledge_item['img_path'],
                knowledge_item['pos_item_ids'],
                pos_item_contents,
            ]

            # if self.config.args.log_prediction_tables_with_images:
            #     # Replace image keys with real images
            #     input_image_file_name = knowledge_item['img_file_name']
            #     input_image = artifact.get(input_image_file_name)
            #     if input_image is None:
            #         input_image = artifact.get(input_image_file_name)
                
            #     table_entry[1] = input_image

            
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
        