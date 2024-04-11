

import copy
import math
import os
from turtle import forward
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import Counter, defaultdict
from easydict import EasyDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5PreTrainedModel
from transformers import VisualBertModel, VisualBertConfig, BertTokenizer
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from transformers import BertModel, BertConfig
# from transformers.models.rag.retrieval_rag import CustomHFIndex, CanonicalHFIndex
from transformers import Blip2ForConditionalGeneration, Blip2Config

# For ColBERT model
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.colbert import ColBERT
from src.models.retriever.visual_colbert import *
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.data import Queries
from colbert import Searcher

from transformers.models.rag.retrieval_rag import Index

import pytorch_lightning as pl

import time

import logging
logger = logging.getLogger(__name__)


import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk
import faiss
import pickle
from typing import Iterable, List, Optional, Tuple
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import random
import re

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


PROMPT_PREFIX = """
You are a question answering system. You will need to answer the questions based on the knowledge provided. You need to think and reason step by step. 
If the provided knowledge can be used to answer the question, output [yes].
If the knowledge is not relevant to the question, output [no].
In any case, you should generate an answer even if the provided knowledge is irrelevant.
You shouldn't answer with "unknown" or "no information provided".
Your answer should be less than 5 words.

Here are some examples:
- Question: Which american president is most associated with the stuffed animal seen here?
- Caption:  a group of teddy bears sitting on a chair with wine bottles. 
- Knowledge: we've all played with one at some point; stuffed animal bears are rooted in our country's childhood.  theodore roosevelt, commonly known as teddy, became the 26th president of the united states in 1901.  significantly behind him with longino, decided to tie the bear up for roosevelt.
- Answer: [yes] theodore roosevelt.

- Question: What is on the round plate?
- Caption:  a tray of food with a bowl of fruit and a glass of orange juice. 
- Knowledge: fresh croissants on grey round plate on white wooden surface, top view.  from above, overhead, flat lay.
- Answer: [yes] croissant.

- Question: What's the name of this river?
- Caption:  a red and black boat is sitting on the water. 
- Knowledge: the mississippi river, derived from the old ojibwe word misi-ziibi meaning 'great river' (gichi-ziibi 'big river' at its headwaters), is the second-longest river in the united states; the longest is the missouri river, which flows into the mississippi.  the abandoned distributary diminishes in volume and forms what are known as bayous.  after flowing into cass lake, the name of the river again changed to miskwaawaakokaa-ziibi (red cedar river) and then to gichi-ziibi (big river) after flowing into lake winnibigoshish.
- Answer: [yes] mississippi.

- Question: Why do kids stare?
- Caption:  a woman and two girls sitting on a bench. 
- Knowledge: when my kids stare off into space, i am also annoyed.
- Answer: [no] curious.

- Question: What are the most popular countries for this sport?
- Caption:  a group of young men playing a game of soccer.
- Knowledge: football (soccer) is the nation’s most popular sport, and brazilians are highly enthusiastic fans.  for the country’s beaches, including rio de janeiro’s famous copacabana and ipanema.
- Answer: [yes] brazil.

- Question: What character is this umbrella designed after?
- Caption:  a group of children playing with an umbrella.
- Knowledge: a fantastic but more intense muppet adventure.  charles cassady jr., common sense media parents recommendpopular with kids common sense says age 10+ (i) a fantastic but more intense muppet adventure. , common sense media parents recommendpopular with kids common sense says age 10+ (i) a fantastic but more intense muppet adventure.
- Answer: [no] sock monkey.

- Question: How do you care for this equipment?
- Caption:  a man walking on the beach with a surfboard.
- Knowledge: perform hand hygiene with an alcohol-based hand rub (preferably) or soap and water whenever ungloved hands touch contaminated ppe items.e.3.  patient-care equipment that is required for use by other patients before use.place an appropriate container with a lid outside the door for equipment that requires disinfection or sterilization.keep adequate equipment required for cleaning or disinfection inside the isolation room or area, and ensure scrupulous daily cleaning of the isolation room or area.set up a telephone or other method of communication in the isolation room or area to enable patients, family members or visitors to communicate with health-care workers.
- Answer: [no] wax.

- Question: What company produces the most of the cooking appliance in this photo?
- Caption:  a kitchen with white cabinets and a stove and sink.
- Knowledge: this aerial photo shows appliance park in 1963.  of the domestic refrigerator market, and ge would eventually produce more than a million units. refrigerator innovation did not stop there.
- Answer: [yes] ge.
"""

class HFIndexBase(Index):
    def __init__(self, vector_size, dataset, index_initialized=False):
        self.vector_size = vector_size
        self.dataset = dataset
        self._index_initialized = index_initialized
        self._check_dataset_format(with_index=index_initialized)
        dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32")

    def _check_dataset_format(self, with_index: bool):
        if not isinstance(self.dataset, Dataset):
            raise ValueError(f"Dataset should be a datasets.Dataset object, but got {type(self.dataset)}")
        # if len({"title", "text", "embeddings"} - set(self.dataset.column_names)) > 0:
        #     raise ValueError(
        #         "Dataset should be a dataset with the following columns: "
        #         "title (str), text (str) and embeddings (arrays of dimension vector_size), "
        #         f"but got columns {self.dataset.column_names}"
        #     )
        if with_index and "embeddings" not in self.dataset.list_indexes():
            raise ValueError(
                "Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it "
                "or `dataset.load_faiss_index` to load one from the disk."
            )

    def init_index(self):
        raise NotImplementedError()

    def is_initialized(self):
        return self._index_initialized

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return np.array(ids), np.array(vectors)  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)



class CustomHFIndex(HFIndexBase):
    """
    A wrapper around an instance of [`~datasets.Datasets`]. The dataset and the index are both loaded from the
    indicated paths on disk.
    Args:
        vector_size (`int`): the dimension of the passages embeddings used by the index
        dataset_path (`str`):
            The path to the serialized dataset on disk. The dataset should have 3 columns: title (str), text (str) and
            embeddings (arrays of dimension vector_size)
        index_path (`str`)
            The path to the serialized faiss index on disk.
    """

    def __init__(self, vector_size: int, dataset, index_path=None):
        super().__init__(vector_size, dataset, index_initialized=index_path is None)
        self.index_path = index_path

    @classmethod
    def load_from_disk(cls, vector_size, dataset_path, index_path):
        logger.info(f"Loading passages from {dataset_path}")
        if dataset_path is None or index_path is None:
            raise ValueError(
                "Please provide `dataset_path` and `index_path` after calling `dataset.save_to_disk(dataset_path)` "
                "and `dataset.get_index('embeddings').save(index_path)`."
            )
        dataset = load_from_disk(dataset_path)
        return cls(vector_size=vector_size, dataset=dataset, index_path=index_path)

    def init_index(self):
        if not self.is_initialized():
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
            self._index_initialized = True



def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class RagModelForGPT(pl.LightningModule):
    '''
    Class for RAG, re-implementation
    '''
    def __init__(self, config: EasyDict, prepared_data) -> None:
        super().__init__()

        self.config = config
        self.prepared_data = prepared_data
        self.tokenizers = self.prepared_data['tokenizers']

        self.retriever_tokenizer = self.tokenizers['tokenizer']
        self.generator_tokenizer = self.tokenizers['decoder_tokenizer']

        # read from prepared data
        self.passage_id2doc = self.prepared_data['passages'].id2doc
        
        # Initialising question encoder
        QueryEncoderModelClass = globals()[self.config.model_config.QueryEncoderModelClass]
        
        self.use_colbert = True if "ColBERT" in self.config.model_config.QueryEncoderModelClass  else False

        if self.use_colbert:
            if "$" in self.config.model_config.QueryEncoderModelVersion:
                self.config.model_config.QueryEncoderModelVersion = os.path.join(self.config.root_exp_dir, self.config.model_config.QueryEncoderModelVersion.replace('$', ''))

            colbert_config = ColBERTConfig(
                bsize=None,
                use_ib_negatives=True,
                checkpoint=self.config.model_config.QueryEncoderModelVersion,
                rank=self.global_rank,
            )
            colbert_config.checkpoint = colbert_config.checkpoint
            if QueryEncoderModelClass != ColBERT:
                # custom ColBERT model, need to input our own config
                self.question_encoder = QueryEncoderModelClass(
                    name=colbert_config.checkpoint, 
                    colbert_config=colbert_config,
                    global_config=self.config)

                # Additionally, we load the vision projection weights for VisualColBERT
                checkpoint_to_load = os.path.join(
                    self.config.model_config.QueryEncoderModelVersion,
                    "vision_projection.pt",
                )
                if os.path.exists(checkpoint_to_load):
                    # We manually load the state dict
                    print(f"Loading from {checkpoint_to_load}")
                    state_dict_from_ckpt = torch.load(checkpoint_to_load, map_location=self.device)
                    self.question_encoder.vision_projection.load_state_dict(state_dict_from_ckpt)
                    print(f"Load the following parameters to vision_projection from the given checkpoint: {state_dict_from_ckpt.keys()}")
                else:
                    logger.warning("No vision projection weights found.")

            else:
                self.question_encoder = QueryEncoderModelClass(
                    name=colbert_config.checkpoint, 
                    colbert_config=colbert_config)
                

            # self.question_encoder.raw_tokenizer = self.retriever_tokenizer

            # Resize the bert embedding space to accommodate special tokens
            # logger.info(f'tokenizer lengths = {len(self.tokenizer.tok)} and {len(self.decoder_tokenizer.tok)}')
            # self.model.bert.resize_token_embeddings(
            #     max(len(self.tokenizer.tok), len(self.decoder_tokenizer.tok))
            # )

        else:
            QueryEncoderConfigClass = globals()[self.config.model_config.QueryEncoderConfigClass]
            
            if "$" in self.config.model_config.QueryEncoderModelVersion:
                self.config.model_config.QueryEncoderModelVersion = os.path.join(self.config.root_exp_dir, self.config.model_config.QueryEncoderModelVersion.replace('$', ''))
            
            question_encoder_model_config = QueryEncoderConfigClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion)
            self.question_encoder = QueryEncoderModelClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion,
                                                        config=question_encoder_model_config)
            self.retiever_hidden_size = question_encoder_model_config.hidden_size
            self.question_encoder.resize_token_embeddings(len(self.retriever_tokenizer))

        
        # Initialising generator
        # GeneratorModelClass = globals()[self.config.model_config.GeneratorModelClass]
        # GeneratorConfigClass = globals()[self.config.model_config.GeneratorConfigClass]
        # generator_model_config = GeneratorConfigClass.from_pretrained(self.config.model_config.GeneratorModelVersion)
        # self.generator = GeneratorModelClass.from_pretrained(self.config.model_config.GeneratorModelVersion,
        #                                             config=generator_model_config)
        
        # self.generator_tokenizer = self.generator_tokenizer.tokenizer

        # self.generator.resize_token_embeddings(len(self.generator_tokenizer))
        
        # self.loss_fct = CrossEntropyLoss(ignore_index=-100)
        # # label smoother imported from huggingface transformers
        # label_smoothing_factor = self.config.train.get('label_smoothing_factor', 0)
        # if label_smoothing_factor != 0:
        #     from transformers.trainer_pt_utils import LabelSmoother
        #     self.label_smoother = LabelSmoother(epsilon=label_smoothing_factor)
        # else:
        #     self.label_smoother = None
        
        self.init_retrieval()

        self.retrieve = self.main_retrieve

    
    def init_retrieval(self):

        # Prepend EXPERIMENT_FOLDER to all paths
        for k, v in self.config.model_config.index_files.items():
            if "$" in v:
                self.config.model_config.index_files[k] = os.path.join(self.config.root_exp_dir, v.replace('$', ''))
            
        if self.use_colbert:
            # Use ColBERT index

            index_path = self.config.model_config.index_files.index_path
            index_root = os.path.dirname(index_path)
            index_name = os.path.basename(index_path)

            if self.device == torch.device('cpu'):
                total_visible_gpus = 0
            else:
                total_visible_gpus = 1
            
            with Run().context(RunConfig(nranks=1, rank=self.global_rank, root=index_root, experiment=index_name)):
                config = ColBERTConfig(
                    total_visible_gpus=total_visible_gpus,
                )
                self.index = Searcher(index=f"temp_index.nbits=8", config=config)

            # Load embedding
            logger.info(f"Loading embedding data from {self.config.model_config.index_files.embedding_path}")
            with open(self.config.model_config.index_files.embedding_path, 'rb') as f:
                embedding_data = pickle.load(f)
                self.item_embeddings = {}
                # self.item_embedding_mask = {}
                for index, item_embeddings, item_embedding_mask in tqdm(zip(list(range(len(embedding_data['item_embeddings']))), embedding_data['item_embeddings'], embedding_data['item_embedding_mask'])):
                    self.item_embeddings[index] = (item_embeddings, item_embedding_mask)
                self.passage_index2id = embedding_data['passage_index2id']
            self.data_source = 'custom'
            return
        
        if self.config.model_config.index_files.index_passages_path == '':
            # use wikidata
            self.index = CanonicalHFIndex(
                vector_size=self.retiever_hidden_size,
                dataset_name=self.config.model_config.index_files.index_dataset,
                dataset_split=self.config.model_config.index_files.index_dataset_split,
                index_name=self.config.model_config.index_files.index_name,
                index_path=None,
                use_dummy_dataset=True if self.config.model_config.index_files.index_dummy else False,
            )
            self.data_source = 'wiki'
        else:
            # use custom corpus
            self.index = CustomHFIndex.load_from_disk(
                vector_size=self.retiever_hidden_size,
                dataset_path=self.config.model_config.index_files.index_passages_path,
                index_path=self.config.model_config.index_files.index_path,
            )
            self.data_source = 'custom'
        print("initializing retrieval")
        self.index.init_index()
        self.table_dict = self.index.dataset.to_pandas().set_index("table_id", drop=False).to_dict(orient="index")
        print("init done.")

    def main_retrieve(self, 
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, 
                    labels: torch.Tensor, 
                    question_ids: List, 
                    input_text_sequences: List, 
                    image_features: torch.Tensor = None,
                    n_docs=None,
                    **kwargs):
        """ Main retrieval function, retrieve documents using retriever

        Args:
            input_ids (torch.Tensor): [description]
            attention_mask (torch.Tensor): [description]
            labels (torch.Tensor): [description]
            question_ids (List): [description]
            input_text_sequences (List): [description]
            n_docs ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if n_docs is None:
            n_docs = self.config.model_config.num_knowledge_passages

        batch_size = input_ids.shape[0]

        # Generate query embeddings and obtain item embeddings from index
        if self.use_colbert:
            # start = time.time()
            # print("input_ids.shape", input_ids.shape)
            # print(self.retriever_tokenizer.tok.batch_decode(input_ids))
            # print("attention_mask.shape", attention_mask.shape)
            # print("image_features.shape", image_features.shape)
            if image_features is not None:
                Q = (input_ids, attention_mask, image_features)
            else:
                Q = (input_ids, attention_mask)
            question_hidden_states = self.question_encoder.query(*Q)
            # print("embedding:")
            # print(question_hidden_states)

            # input()
            # print(f"Query embedding time: {time.time() - start}")
            
            # start = time.time()

            custom_quries = {i: query for i, query in enumerate(input_text_sequences)}
            queries = Queries(data=custom_quries)
            
            if n_docs < 5:
                n_docs_retrieve = 5
            else:
                n_docs_retrieve = n_docs
            
            ranking = self.index._search_all_Q(queries, question_hidden_states.cpu().detach(), k=n_docs_retrieve, progress=False)
            
            # print(f"Retrieval time: {time.time() - start}")
            
            # pprint(ranking.todict())
            retrieval_results = ranking.todict()

            doc_scores = []
            all_retrieved_doc_indices = []

            # start = time.time()
            
            for query_index, retrieved_docs in retrieval_results.items():
                retrieved_doc_indices = []
                retrieved_doc_scores = []
                if n_docs != n_docs_retrieve:
                    retrieved_docs = random.sample(retrieved_docs, n_docs)
                
                for doc_index, _, doc_score in retrieved_docs:
                    retrieved_doc_indices.append(doc_index)
                    retrieved_doc_scores.append(doc_score)
                
                # Get embeddings
                retrieved_item_embeddings = []
                retrieved_item_embeding_mask = []
                for i in retrieved_doc_indices:
                    emb_tuple = self.item_embeddings[i]
                    retrieved_item_embeddings.append(torch.Tensor(emb_tuple[0]))
                    retrieved_item_embeding_mask.append(torch.Tensor(emb_tuple[1]))
                
                retrieved_item_embeddings = torch.stack(retrieved_item_embeddings).to(self.device)
                retrieved_item_embeding_mask = torch.stack(retrieved_item_embeding_mask).to(self.device)
                
                retrieved_query_embedding = question_hidden_states[[query_index]]

                self.question_encoder.colbert_config.nway = len(retrieved_doc_indices)
                Q_duplicated = retrieved_query_embedding.repeat_interleave(self.question_encoder.colbert_config.nway, dim=0).contiguous()
                scores = self.question_encoder.score(Q_duplicated, retrieved_item_embeddings, retrieved_item_embeding_mask)

                doc_scores.append(scores)
                all_retrieved_doc_indices.append(retrieved_doc_indices)
                # print("processing time", time.time() - start)
            
            # batch_size x n_docs
            doc_scores = torch.stack(doc_scores)
            ids = np.array(all_retrieved_doc_indices)
        else:
            # Use question_encoder to encode question inputs
            query_outputs = self.question_encoder(input_ids=input_ids,
                                                attention_mask=attention_mask)
            question_hidden_states = query_outputs.pooler_output
            # print('question_hidden_states', question_hidden_states.shape)

            # start_time = time.time()
            ids, vectors = self.index.get_top_docs(question_hidden_states.cpu().detach().numpy(), n_docs)
            # print(
            #     f"index search time: {time.time() - start_time} sec, batch size {question_hidden_states.shape}"
            # )
            # print(ids)

            # question_hidden_states: batch_size x hidden_size
            # item_hidden_states: batch_size x n_docs x hidden_size
            item_hidden_states = torch.Tensor(vectors).type_as(question_hidden_states)

            # print('item_hidden_states', item_hidden_states.shape)

            # batch_size x n_docs
            doc_scores = (question_hidden_states.unsqueeze(dim=1) * item_hidden_states).sum(dim=-1)
        
        
        if 'add_null_document' in self.config.model_config.modules:
            null_doc_scores = (question_hidden_states * self.null_embedding.unsqueeze(dim=0)).sum(dim=-1)
            # null_doc_scores: batch_size
            # print('null_doc_scores', null_doc_scores)
            
        doc_scores_cpu = doc_scores.cpu().detach().numpy()

        # start = time.time()

        retrieved_docs = []
        for b in range(batch_size):
            doc_data = []
            if self.use_colbert:
                retrieved_doc_indices = all_retrieved_doc_indices[b]
                retrieved_doc_ids = [self.passage_index2id[i] for i in retrieved_doc_indices]
                contents = [{"id": i, "title": "","text": self.passage_id2doc[i]} for i in retrieved_doc_ids]
            else:
                contents = self.index.get_doc_dicts(ids[b])
            
            for i in range(n_docs):
                if self.data_source == 'wiki':
                    content = 'title: ' + contents[i]['title'] + " content: " + contents[i]['text']
                else:
                    content = contents[i]['text']
                # content = ' '.join(['<BOK>', content, '<EOK>'])
                content = ' '.join([content])
                passage_data = {
                    'passage_id': str(ids[b, i]),
                    'title': contents[i]['title'],
                    'content': content,
                    'score': doc_scores_cpu[b, i]
                }
                doc_data.append(passage_data)
            retrieved_docs.append(doc_data)
        # print(f"retrieval postprocessing time: {time.time() - start} sec, batch size {batch_size}")

        assert len(retrieved_docs) == batch_size

        if 'add_null_document' in self.config.model_config.modules:
            # print('doc_scores', doc_scores.shape) # batch_size x n_docs
            doc_scores = torch.cat([
                null_doc_scores.reshape(batch_size, 1), # batch_size x 1
                doc_scores,
            ], dim=-1)
            # print('after doc_scores', doc_scores.shape) # batch_size x n_docs
            # input()

        return EasyDict(
            retrieved_docs=retrieved_docs,
            doc_scores=doc_scores,
            question_hidden_states=question_hidden_states,
        )

    def static_retrieve(self, 
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, 
                    labels: torch.Tensor, 
                    question_ids: List, 
                    input_text_sequences: List, 
                    n_docs=None,
                    **kwargs):
        """A dummy retrieval function, retrieve from static results

        Args:
            input_ids (torch.Tensor): [description]
            attention_mask (torch.Tensor): [description]
            labels (torch.Tensor): [description]
            question_ids (List): [description]
            input_text_sequences (List): [description]
            n_docs ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if n_docs is None:
            n_docs = self.config.model_config.num_knowledge_passages
        
        batch_size = input_ids.shape[0]

        #####   Dummy Retrieval ####
        retrieved_docs = []
        doc_scores = []
        for question_id in question_ids:
            annotation = self.questionId2topPassages[str(question_id)]
            top_passages = annotation[:n_docs]
            retrieved_docs.append(top_passages)
            scores = [p['score'] for p in top_passages]
            doc_scores.append(scores)
        
        doc_scores = torch.FloatTensor(doc_scores).type_as(input_ids)

        assert len(retrieved_docs) == batch_size

        return EasyDict(
            retrieved_docs=retrieved_docs,
            doc_scores=doc_scores,
        )

    def prepare_inputs_for_generator(self, 
                input_text_sequences, retrieved_docs, labels, n_docs=None):
        
        if n_docs is None:
            n_docs = self.config.model_config.num_knowledge_passages
        
        batch_size = len(input_text_sequences)

        extended_input_text_sequences = []

        for index, input_text_sequence in enumerate(input_text_sequences):
            scores = []
            for doc in retrieved_docs[index]:
                # input_text_sequence = input_text_sequence.split("<EOQ>")[0]
                # input_text_sequence = input_text_sequence.replace("<BOQ>", "").strip()
                
                # remove content between <BOV> and <EOV>
                input_text_sequence = re.sub(r'<BOV>.*?<EOV>', '', input_text_sequence)

                input_text_sequence = input_text_sequence.replace("<BOQ>", "- Question:")
                input_text_sequence = input_text_sequence.replace("<EOQ>", "")
                input_text_sequence = input_text_sequence.replace("<BOC>", "\n- Caption: ")
                input_text_sequence = input_text_sequence.replace("<EOC>", "")
                input_text_sequence = input_text_sequence.replace("<BOV>", "")
                input_text_sequence = input_text_sequence.replace("<EOV>", "")
                input_text_sequence = input_text_sequence.replace("<SOV>", "")
                input_text_sequence = input_text_sequence.strip()

                extended_input_text_sequences.append( 
                    ' '.join([input_text_sequence, "\n- Knowledge:", doc['content'], "\n- Answer:"]) 
                ) # "Knowledge:", doc['content'], 
                scores.append(doc['score'])
        
        
        return EasyDict(
            generator_input_text_sequences=extended_input_text_sequences,
            generator_input_ids=None,
            generator_attention_mask=None,
            generator_decoder_input_ids=None,
            generator_labels=None,
        )

    def forward(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels: torch.Tensor,
                      question_ids: List,
                      input_text_sequences: List,
                    **kwargs):
        
        n_docs = 1 #self.config.model_config.num_knowledge_passages
        pixel_values = kwargs.get('pixel_values', None)
        image_features = kwargs.get('image_features', None)
        
        batch_size = input_ids.shape[0]

        # Retrieve docs for given question inputs
        retrieval_results = self.retrieve(input_ids, attention_mask, labels, question_ids, input_text_sequences, image_features=image_features, n_docs=n_docs)
        retrieved_docs, doc_scores = retrieval_results.retrieved_docs, retrieval_results.doc_scores
        
        answers = kwargs.get('answers', None)
        assert answers is not None
        get_retrieval_labels_results = self.get_retrieval_labels(
            question_ids=question_ids,
            batch_answers=answers,
            batch_retrieved_docs=retrieved_docs,
        )
        retrieval_labels = get_retrieval_labels_results.retrieval_labels
        
        
        if 'add_null_document' in self.config.model_config.modules:
            n_docs += 1
        
        if 'force_existence' in self.config.model_config.modules:
            # Force the label to be in the retrieved document
            selected_answers = get_retrieval_labels_results.selected_answers
            target_encoding = self.generator_tokenizer(selected_answers,
                    padding='longest',
                    max_length=self.config.model_config.max_target_length,
                    truncation=True)
            labels = target_encoding.input_ids
            labels = torch.LongTensor(labels).type_as(input_ids)
        else:
            labels = labels.repeat_interleave(n_docs, 0)

        pixel_values = pixel_values.repeat_interleave(n_docs, 0)

        # prepare inputs for generator
        generator_inputs = self.prepare_inputs_for_generator(input_text_sequences=input_text_sequences,
                                            retrieved_docs=retrieved_docs,
                                            labels=labels, n_docs=n_docs)
        

        generator_outputs = self.generator(
                            pixel_values=pixel_values,
                            input_ids=generator_inputs.generator_input_ids,
                            attention_mask=generator_inputs.generator_attention_mask,
                            decoder_input_ids=generator_inputs.generator_decoder_input_ids,
                            labels=labels,
                            return_dict=True)
        
        logits = generator_outputs.logits

        loss_dict = self.get_loss(
            seq_logits=logits,
            doc_scores=doc_scores,
            target=generator_inputs.generator_labels,
            exclude_bos_score=False,
            n_docs=n_docs,
            retrieval_labels=retrieval_labels,
        )
        # print(loss_dict)

        # we use the loss produced by BLIP 2 directly
        # TODO: match with our own get_loss implementation
        # loss_dict = {
        #     "nll_loss": generator_outputs.loss,
        # }

        # aggregate loss
        total_loss = 0
        for loss_name, loss_ratio in self.config.model_config.loss_ratio.items():
            if loss_ratio != 0:
                total_loss += loss_dict[loss_name] * loss_ratio
        
        
        # additional_loss_ratio = self.config.model_config.additional_loss_ratio
        # regularization_ratio = self.config.model_config.regularization_ratio
        # marginalise_loss_ratio = self.config.model_config.marginalise_loss_ratio

        # if additional_loss_ratio != 0:
        #     # Add in-retrieval contrastive loss to the retriever!
        #     answers = kwargs.get('answers', None)
        #     assert answers is not None
        #     retrieval_labels = self.get_retrieval_labels(
        #         batch_answers=answers,
        #         batch_retrieved_docs=retrieved_docs,
        #     )
        #     # print('retrieval_labels', retrieval_labels)
        #     doc_scores_softmaxed = F.softmax(doc_scores, dim=-1)
        #     retrieval_labels = retrieval_labels.to(doc_scores_softmaxed.device)
        #     # print('doc_scores_softmaxed', doc_scores_softmaxed)
        #     retrieval_loss = F.binary_cross_entropy(doc_scores_softmaxed, retrieval_labels)
        #     # print(loss, retrieval_loss)
        #     # input()
        #     loss += retrieval_loss * additional_loss_ratio

        
        
        # if regularization_ratio != 0 and batch_size > 1:
        #     question_hidden_states = retrieval_results.question_hidden_states
        #     cor = 0.0
        #     for i in range(batch_size):
        #         for j in range(i + 1, batch_size):
        #             cor += self.DistanceCorrelation(question_hidden_states[i], question_hidden_states[j])
            
        #     print('cor', cor)
        #     loss += regularization_ratio * cor[0]
        
        # if marginalise_loss_ratio != 0:
        #     loss += marginalise_loss_ratio * loss_dict.dist_loss

        # function to extract grad
        def set_grad(var):
            def hook(grad):
                var.grad = grad
                print('setting grad:', grad)
            return hook
        
        # answers = kwargs.get('answers', None)
        # assert answers is not None
        # retrieval_labels = self.get_retrieval_labels(
        #     batch_answers=answers,
        #     batch_retrieved_docs=retrieved_docs,
        # )
        # print(F.softmax(doc_scores, dim=-1))
        # print(retrieval_labels)
        # print('-------------')
        # # register_hook for Z
        # doc_scores.register_hook(set_grad(doc_scores))
        return EasyDict(loss=total_loss,
                        loss_dict=loss_dict,
                        doc_scores=doc_scores.cpu().detach().numpy())


    def generate(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels: torch.Tensor,
                      question_ids: List,
                      input_text_sequences: List,
                      n_docs: int=None,
                      **kwargs):

        batch_size = input_ids.shape[0]
        image_features = kwargs.get('image_features', None)
        
        # Retrieve docs for given question inputs
        retrieval_results = self.main_retrieve(input_ids, attention_mask, labels, question_ids, input_text_sequences, image_features=image_features, n_docs=n_docs)
        retrieved_docs, doc_scores = retrieval_results.retrieved_docs, retrieval_results.doc_scores
        

        if n_docs is None:
            n_docs = self.config.model_config.num_knowledge_passages
            if 'add_null_document' in self.config.model_config.modules:
                n_docs += 1

        # populate labels
        labels = labels.repeat_interleave(n_docs, 0)

        # prepare inputs for generator
        generator_inputs = self.prepare_inputs_for_generator(
                                            input_text_sequences=input_text_sequences,
                                            retrieved_docs=retrieved_docs,
                                            labels=labels,
                                            n_docs=n_docs)

        # get pixel values
        pixel_values = kwargs.get('pixel_values', None)
        # populate pixel values
        if pixel_values is not None:
            pixel_values = pixel_values.repeat_interleave(n_docs, 0)
        
        # print("start generation")
        # print("pixel_values", pixel_values.shape)

        generator_input_text_sequences = generator_inputs.generator_input_text_sequences
        
        generated_sequences = []

        # print(generator_input_text_sequences)
        for prompt in generator_input_text_sequences:
            query = PROMPT_PREFIX + "\n" + prompt
            get_response = False
            while not get_response:
                try:
                    res = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        max_tokens=16,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": ""},
                            {"role": "user", "content": query},
                            # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                            # {"role": "user", "content": "Where was it played?"}
                        ]
                    )
                    get_response = True
                except Exception as e:
                    print(f"openai.ChatCompletion.create error: {str(e)} \n-----\n retrying...")
                    time.sleep(3)
                    continue
            
            choice = res.choices[0]
            generated_sequence = choice['message']['content']
            generated_sequences.append(generated_sequence)

        print("generated_sequences", generated_sequences)
        generation_outputs = generated_sequences
        generation_seq_scores = torch.ones(len(generated_sequences)).to(self.device)

        cleaned_generated_sequences = []
        for index, generated_sentence in enumerate(generated_sequences):
            generated_sentence = generated_sentence.lower().strip()
            reduce_mark = False
            if "[yes]" in generated_sentence:
                generated_sentence = generated_sentence.replace("[yes]", "")
                reduce_mark = False
            elif "[no]" in generated_sentence:
                generated_sentence = generated_sentence.replace("[no]", "")
                reduce_mark = True
            elif "[unknown]" in generated_sentence:
                generated_sentence = generated_sentence.replace("[unknown]", "")
                reduce_mark = True
            elif "[guess]" in generated_sentence:
                generated_sentence = generated_sentence.replace("[guess]", "")
                reduce_mark = True
            else:
                reduce_mark = True
            
            # remove "[" and "]"
            generated_sentence = generated_sentence.replace("[", "")
            generated_sentence = generated_sentence.replace("]", "")

            if reduce_mark:
                generation_seq_scores[index] = generation_seq_scores[index] - 1000
            generated_sentence = generated_sentence.strip()
            cleaned_generated_sequences.append(generated_sentence)

        print("generation_seq_scores", generation_seq_scores)
        print("cleaned_generated_sequences", cleaned_generated_sequences)
        # input()
        generation_seq_scores = generation_seq_scores.reshape(batch_size, n_docs)

        # decode the generation outputs
        generation_outputs = cleaned_generated_sequences
        generation_outputs_decoded = cleaned_generated_sequences

        # Find answer proposals from n_docs outputs for each question
        outputs = []
        generation_outputs_for_docs = []

        # Re-forward the generator, and use generation outputs as labels
        # obtain the loss of each (question, passage) pair

        # reshape generation_outputs
        # generation_outputs = generation_outputs.reshape(batch_size, n_docs, -1)

        # doc_scores_log --> log_softmax --> log(g(z))
        # generation_seq_scores --> log(p(y|x, z))
        # log(g(z)p(y|x, z)) = doc_scores + generation_seq_scores
        # batch_size x n_docs + batch_size x n_docs
        doc_scores_log = F.log_softmax(doc_scores, dim=-1)
        # print('doc_scores_log', doc_scores_log)
        # print('generation_seq_scores', generation_seq_scores)
        loss_with_doc_scores = doc_scores_log + generation_seq_scores

        for b in range(batch_size):
            # use topk to get indices of top candidates
            top_cand_inds = (loss_with_doc_scores[b]).topk(1)[1]
            outputs.append(generation_outputs[b*n_docs+top_cand_inds])
            answer_proposals = generation_outputs_decoded[b*n_docs:(b+1)*n_docs]
            generation_outputs_for_docs.append(answer_proposals)
            # print(-loss[b])
            # print(answer_proposals)
        # outputs = torch.cat(outputs)
    

        return EasyDict(outputs=outputs, 
                        retrieved_docs=retrieved_docs, 
                        doc_scores=doc_scores.cpu().detach().numpy(),
                        loss_with_doc_scores=loss_with_doc_scores.cpu().detach().numpy(),
                        generation_outputs_for_docs=generation_outputs_for_docs)

    def get_loss(
        self, seq_logits, doc_scores, target, reduce_loss=True, epsilon=0.0, exclude_bos_score=False, ignore_index=-100, n_docs=None, retrieval_labels=None,
    ):
        """Compute loss

        Args:
            seq_logits (_type_): _description_
            doc_scores (_type_): _description_
            target (_type_): _description_
            reduce_loss (bool, optional): _description_. Defaults to True.
            epsilon (float, optional): _description_. Defaults to 0.0.
            exclude_bos_score (bool, optional): _description_. Defaults to False.
            ignore_index (int, optional): _description_. Defaults to -100.
            n_docs (_type_, optional): _description_. Defaults to None.
            retrieval_labels (_type_, optional): _description_. Defaults to None.

        Returns:
            EasyDict: every loss requested
        """

        if n_docs is None:
            n_docs = self.config.model_config.num_knowledge_passages
        
        loss_dict = EasyDict()
        
        # bos_token_id is None for T5
        bos_token_id = self.generator.config.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        
        batch_size = seq_logits.shape[0] // n_docs
        seq_len = seq_logits.shape[1]
        # seq_logits dim = (batch*n_docs, seq_len , #vocabs)
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            batch_size, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x vocab_size
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('doc_logprobs', doc_logprobs.shape)

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        if use_bos:
            second_token_scores = seq_logprobs[:, :, 1:2, :]
            remainder = seq_logprobs[:, :, 2:, :]
            rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)
        else:
            # print('using T5 doc probs!')
            remainder = seq_logprobs[:, :, 1:, :]
            rag_logprobs = torch.cat([first_token_scores + doc_logprobs, remainder], dim=2)

        # print("target", target)
        # print(self.generator_tokenizer.batch_decode(target))
        # sequence_token_ids = torch.argmax(nn.functional.log_softmax(seq_logits, dim=-1), dim=-1)
        # print('sequence_token_ids', sequence_token_ids)
        # print(self.generator_tokenizer.batch_decode(sequence_token_ids))
        # input()
        # if self.use_decoder_only_language_model:
        #     # if using OPT, we need to shift the sequence logits to match the labels
        #     logits = logits[:, -labels.size(1) :, :]
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous().to(logits.device)

        # Compute NLL Loss for seq_logprobs
        new_target = target.reshape(batch_size, n_docs, -1).unsqueeze(-1)
        assert new_target.dim() == seq_logprobs.dim()

        pad_mask = new_target.eq(ignore_index)

        if pad_mask.any() and ignore_index < 0:
            # fill -100 to be 0, avoid indexing error using gather
            new_target.masked_fill_(pad_mask, 0)

        ll = seq_logprobs.gather(dim=-1, index=new_target)
        if pad_mask.any():
            ll.masked_fill_(pad_mask, 0.0)
        
        ll = ll.squeeze(-1) # batch_size x n_docs x seq_len

        nll_loss = -ll
        loss_dict.nll_loss = nll_loss

        if self.config.model_config.loss_ratio.rag_loss != 0:
            # Compute RAG loss
            # reuse new_target
            rag_ll = rag_logprobs.gather(dim=-1, index=new_target)
            # reuse pad_mask
            if pad_mask.any():
                rag_ll.masked_fill_(pad_mask, 0.0)
            
            rag_ll = rag_ll.squeeze(-1) # batch_size x n_docs x seq_len

            # reduce directly since we don't use it elsewhere
            # in training, the marginalisation is performed
            # print('ll', ll.shape)
            # print(rag_ll)
            rag_ll = rag_ll.sum(2)  # sum over tokens
            # print(rag_ll)
            rag_ll = rag_ll.logsumexp(1)  # sum over docs
            # print(rag_ll)
            rag_ll = rag_ll.sum() # sum over batches
            # print(rag_ll)
            # print('after logsumexp', ll.shape)
            rag_loss = -rag_ll
            # print(rag_loss)
            # input('done')

            loss_dict.rag_loss = rag_loss

        if self.config.model_config.loss_ratio.additional_loss != 0:
            if retrieval_labels is not None:
                first_token_scores = first_token_scores.detach()

                # batch_size x n_docs x voc_size
                first_token_scores = first_token_scores.squeeze(2)
                # batch_size x n_docs
                first_token_prediction = torch.argmax(first_token_scores, dim=-1)
                # print('first_token_prediction', first_token_prediction)

                # batch_size x n_docs
                first_token_target = target.reshape(batch_size, n_docs, -1)[:, :, 0]
                # print('first_token_target', first_token_target)

                prediction_labels = (first_token_prediction == first_token_target)
                # print(prediction_labels)
                retrieval_labels = retrieval_labels.to(seq_logits.device).float()
                # print(retrieval_labels)

                RAVQA_loss_type = self.config.model_config.RAVQA_loss_type
                if RAVQA_loss_type == 'Approach1':
                    ##############   approach 1:  ##################
                    # ignore documents with wrong predictions and a negative pseudo label
                    # correct prediction + positive pseudo label = 1
                    # wrong prediction + positive pseudo label = 1
                    # correct prediction + negative pseudo label = 1
                    # wrong prediction + negative pseudo label = -100
                    merged_labels = torch.logical_or(prediction_labels, retrieval_labels).float()
                    ignore_mask = (merged_labels==0)
                    # print(merged_labels)
                elif RAVQA_loss_type == 'Approach2':
                    ##############   approach 2:  ##################
                    #  reduce documents with wrong predictions and with a negative pseudo label
                    #  ignore ducuments with correct predictions but with a negative pseudo label
                    merged_labels = torch.logical_or(prediction_labels, retrieval_labels).float()
                    ignore_mask = torch.logical_and((prediction_labels==1), (retrieval_labels==0))
                elif RAVQA_loss_type == 'Approach3':
                    ##############   approach 3:  ##################
                    #  ignore documents with wrong predictions and a negative pseudo label
                    #  ignore ducuments with correct predictions but with a negative pseudo label
                    merged_labels = torch.logical_or(prediction_labels, retrieval_labels).float()
                    ignore_mask = (retrieval_labels==0)

                elif RAVQA_loss_type == 'Approach4':
                    ##############   approach 4:  ##################
                    # correct prediction + positive pseudo label = 1
                    # wrong prediction + positive pseudo label = 1
                    # correct prediction + negative pseudo label = -100
                    # wrong prediction + negative pseudo label = 0
                    merged_labels = retrieval_labels
                    merged_labels = merged_labels.float()
                    ignore_mask = torch.logical_and((prediction_labels==1), (retrieval_labels==0))

                elif RAVQA_loss_type == 'Approach5':
                    ##############   approach 5:  ##################
                    # correct prediction + positive pseudo label = 1
                    # wrong prediction + positive pseudo label = -100
                    # correct prediction + negative pseudo label = -100
                    # wrong prediction + negative pseudo label = -100
                    merged_labels = torch.logical_and(prediction_labels, retrieval_labels).float()
                    ignore_mask = (merged_labels==0)
                
                elif RAVQA_loss_type == 'Approach6':
                    ##############   approach 6:  ##################
                    # correct prediction + positive pseudo label = 1
                    # wrong prediction + positive pseudo label = -100
                    # correct prediction + negative pseudo label = -100
                    # wrong prediction + negative pseudo label = 0
                    merged_labels = torch.logical_and(prediction_labels, retrieval_labels).float()
                    ignore_mask = torch.logical_or(
                        torch.logical_and((prediction_labels==0), (retrieval_labels==1)),
                        torch.logical_and((prediction_labels==1), (retrieval_labels==0)),
                        )
                elif RAVQA_loss_type == 'NoPR':
                    ##############   approach NoPR:  ##################
                    # correct prediction = 1
                    # wrong prediction = 0
                    merged_labels = prediction_labels.float()
                    ignore_mask = torch.zeros_like(merged_labels).bool().to(merged_labels.device)


                doc_scores_softmaxed = F.softmax(doc_scores, dim=-1)

                dist_loss = F.binary_cross_entropy(doc_scores_softmaxed, merged_labels, reduction='none')
                dist_loss.masked_fill_(ignore_mask, 0.0)

                count_nonzero = torch.count_nonzero(dist_loss)
                if count_nonzero == 0:
                    dist_loss = 0
                else:
                    dist_loss = dist_loss.sum() / torch.count_nonzero(dist_loss)

                loss_dict.additional_loss = dist_loss
            else:
                loss_dict.additional_loss = 0
        
        if reduce_loss:
            mask = (pad_mask == 0)
            nll_loss = nll_loss.sum()
            nll_loss = nll_loss / torch.sum(mask)
            loss_dict.nll_loss = nll_loss

            

        return loss_dict
        


    def get_retrieval_labels(self, 
                            question_ids: List,
                            batch_answers: List, 
                            batch_retrieved_docs: List):
        
        def most_frequent(List):
            return max(set(List), key = List.count)

        retrieved_docs = batch_retrieved_docs
        log_result = {
            'recall': [],
            'precision': [],
            'gold_precision': [],
            'gold_recall': [],
        }
        labels = []
        selected_answers = []
        for question_id, answer_list, docs in zip(question_ids, batch_answers, retrieved_docs):
            
            filtered_answer_list = [ans for ans in answer_list if ans != '']
            gold_answer = most_frequent(filtered_answer_list)
            unique_answers = list(set(answer_list))
            counts = Counter(filtered_answer_list)
            answer_list_by_frequency = sorted(filtered_answer_list, key=lambda x: -counts[x])
            
            doc_texts = [doc['content'] for doc in docs]
            
            found_answers = []
            found_gold_answers = []

            
            if 'add_null_document' in self.config.model_config.modules:
                doc_texts = doc_texts[1:]

            this_batch_labels = [0] * len(doc_texts)
            K = len(doc_texts)

            def check_contain_entity(ans, doc_to_check):
                doc_id = doc_to_check['title']
                triplet = self.data_loader.data.fvqa_data.triplets.get(doc_id, None)
                if triplet is None:
                    logger.error(f'triplet id {doc_id} not found in the data!')
                    return False
                else:
                    triplet_entities = [triplet.e1_label.lower(), triplet.e2_label.lower()]
                    if ans in triplet_entities:
                        return True
                    else:
                        return False
            

            if 'use_entity_in_retrieval_labels' in self.config.model_config.modules:
                for index, passage_data in enumerate(docs):
                    for answer in unique_answers:
                        if check_contain_entity(answer.lower(), passage_data):
                            found_answers.append(answer)
                            this_batch_labels[index] = 1
                            break
                    if check_contain_entity(gold_answer.lower(), passage_data):
                        found_gold_answers.append(answer)
                        this_batch_labels[index] = 1

                for index, passage_data in enumerate(doc_texts):
                    # by default the gold answer is selected, regardless the existence of answer
                    selected_answer = gold_answer
                    # select answer that appears in the document and with highest frequency
                    if gold_answer.lower() in passage_data.lower():
                        pass # no change, by default the gold answer is selected
                    else:
                        for answer in answer_list_by_frequency:
                            if answer == gold_answer:
                                continue # not consider gold answer
                            if answer.lower() in passage_data.lower():
                                selected_answer = answer
                                break
                    selected_answers.append(selected_answer)

            elif 'use_triplet_in_retrieval_labels' in self.config.model_config.modules:
                item = self.data_loader.data.vqa_data.lookup.get(question_id, None)
                ref_triplet_ids = []
                for i in item.facts.values():
                    ref_triplet_ids.extend(i)
                
                for index, passage_data in enumerate(docs):
                    
                    if passage_data['title'] in ref_triplet_ids:
                        this_batch_labels[index] = 1
                        found_answers.append(passage_data['title'])
                        found_gold_answers.append(passage_data['title'])

                for index, passage_data in enumerate(doc_texts):
                    # by default the gold answer is selected, regardless the existence of answer
                    selected_answer = gold_answer
                    # select answer that appears in the document and with highest frequency
                    if gold_answer.lower() in passage_data.lower():
                        pass # no change, by default the gold answer is selected
                    else:
                        for answer in answer_list_by_frequency:
                            if answer == gold_answer:
                                continue # not consider gold answer
                            if answer.lower() in passage_data.lower():
                                selected_answer = answer
                                break
                    selected_answers.append(selected_answer)
                
            else:
                for index, passage_data in enumerate(doc_texts):
                    for answer in unique_answers:
                        if answer.lower() in passage_data.lower():
                            found_answers.append(answer)
                            this_batch_labels[index] = 1
                            break
                    if gold_answer.lower() in passage_data.lower():
                        found_gold_answers.append(answer)
                        this_batch_labels[index] = 1

                for index, passage_data in enumerate(doc_texts):
                    # by default the gold answer is selected, regardless the existence of answer
                    selected_answer = gold_answer
                    # select answer that appears in the document and with highest frequency
                    if gold_answer.lower() in passage_data.lower():
                        pass # no change, by default the gold answer is selected
                    else:
                        for answer in answer_list_by_frequency:
                            if answer == gold_answer:
                                continue # not consider gold answer
                            if answer.lower() in passage_data.lower():
                                selected_answer = answer
                                break
                    selected_answers.append(selected_answer)

            labels.append(this_batch_labels)
                    
            if len(found_answers) > 0:
                # At least one answer is retireved
                log_result['recall'].append(1)
            else:
                log_result['recall'].append(0)
            # The proportion of retrieved knowledge has an answer
            log_result['precision'].append(len(found_answers) / K)

            if len(found_gold_answers) > 0:
                # if gold answer is found
                log_result['gold_recall'].append(1)
            else:
                log_result['gold_recall'].append(0)
            # The proportion of retrieved knowledge has the gold answer
            log_result['gold_precision'].append(len(found_gold_answers) / K)

        labels = torch.FloatTensor(labels)
        return EasyDict(
            retrieval_labels=labels,
            selected_answers=selected_answers,
        )

    @staticmethod
    def DistanceCorrelation(tensor_1, tensor_2):
        # tensor_1, tensor_2: [channel]
        # ref: https://en.wikipedia.org/wiki/Distance_correlation
        channel = tensor_1.shape[0]
        zeros = torch.zeros(channel, channel).to(tensor_1.device)
        zero = torch.zeros(1).to(tensor_1.device)
        tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
        """cul distance matrix"""
        a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
        tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
        a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
        """cul distance correlation"""
        A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
        B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
        dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
        dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
        dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
        return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)





