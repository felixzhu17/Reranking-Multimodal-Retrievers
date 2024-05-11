from colbert.modeling.colbert import ColBERT

from transformers import ViTModel, ViTConfig
from transformers import CLIPVisionConfig, CLIPVisionModel
from transformers import ViTMAEConfig, ViTMAEModel
from transformers.modeling_utils import ModuleUtilsMixin
import torch
import torch.nn as nn


import logging
logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class VisualColBERTForPretraining(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)
        self.global_config = global_config
        self.model_config = global_config.model_config
        
        VisionModelConfigClass = globals()[self.model_config.VisionModelConfigClass]
        VisionModelClass = globals()[self.model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(self.model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(self.model_config.VisionModelVersion, config=vision_model_config)
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_image_encoder' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the ViT image encoder.")
            for name, param in self.vision_model.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        

    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        # Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Q = self.linear(Q)

        # mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        # Q = Q * mask

        outputs = self.vision_model(pixel_values)
        last_hidden_states = outputs.last_hidden_state[:, 0] # bz  x 768
        batch_size = last_hidden_states.shape[0]
        
        last_hidden_states = self.vision_projection(last_hidden_states) # bz x 32*128
        
        last_hidden_states = last_hidden_states.view(
            -1, self.mapping_network_prefix_length, self.lm_embedding_size
        )
        # last_hidden_states = last_hidden_states.view(
        #     batch_size, -1, self.lm_embedding_size
        # )
        Q = last_hidden_states #torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states

        return torch.nn.functional.normalize(Q, p=2, dim=2)
    

class VisualColBERTForRetrieval(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)

        self.global_config = global_config
        self.model_config = global_config.model_config
        
        VisionModelConfigClass = globals()[self.model_config.VisionModelConfigClass]
        VisionModelClass = globals()[self.model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(self.model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(self.model_config.VisionModelVersion, config=vision_model_config)
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_image_encoder' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the ViT image encoder.")
            for name, param in self.vision_model.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        
        

    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        outputs = self.vision_model(pixel_values)
        last_hidden_states = outputs.last_hidden_state[:, 0] # bz  x 768
        batch_size = last_hidden_states.shape[0]
        
        last_hidden_states = self.vision_projection(last_hidden_states) # bz x 32*128
        
        last_hidden_states = last_hidden_states.view(
            -1, self.mapping_network_prefix_length, self.lm_embedding_size
        )
        
        Q = torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states

        return torch.nn.functional.normalize(Q, p=2, dim=2)




class VisualColBERTForRetrievalWithoutVisionModel(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)

        self.global_config = global_config
        self.model_config = global_config.model_config
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        
        

    def query(self, input_ids, attention_mask, image_features):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        image_features = image_features.to(self.device)
        
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask
        
        # image_features: batch_size (x num_ROIs) x ViT hidden_size
        last_hidden_states = image_features
        batch_size = last_hidden_states.shape[0]
        
        last_hidden_states = self.vision_projection(last_hidden_states) # bz (x num_ROIs) x 32*128
        
        # last_hidden_states = last_hidden_states.view(
        #     -1, self.mapping_network_prefix_length, self.lm_embedding_size
        # )
        
        last_hidden_states = last_hidden_states.reshape(
            batch_size, -1, self.lm_embedding_size
        )

        Q = torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states

        return torch.nn.functional.normalize(Q, p=2, dim=2)




class VisualColBERTForPretrainingWithoutVisionModel(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)

        self.global_config = global_config
        self.model_config = global_config.model_config
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )

        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        
        

    def query(self, input_ids, attention_mask, image_features):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        image_features = image_features.to(self.device)

        # Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Q = self.linear(Q)

        # mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        # Q = Q * mask

        last_hidden_states = image_features
        batch_size = last_hidden_states.shape[0]
        
        last_hidden_states = self.vision_projection(last_hidden_states) # bz x 32*128
        
        last_hidden_states = last_hidden_states.view(
            -1, self.mapping_network_prefix_length, self.lm_embedding_size
        )
        
        Q = last_hidden_states
        
        return torch.nn.functional.normalize(Q, p=2, dim=2)



class VisualColBERTForRetrievalMultipleMapping(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)

        self.global_config = global_config
        self.model_config = global_config.model_config
        
        self.vision_projections = nn.ModuleList()

        for proj_name, proj_config in self.model_config.vision_projections.items():
            mapping_network_prefix_length = proj_config.mapping_network_prefix_length
            vision_embedding_size = proj_config.vision_embedding_size
            lm_embedding_size = proj_config.lm_embedding_size

            proj_layer = MLP(
                (
                    vision_embedding_size,
                    (lm_embedding_size * mapping_network_prefix_length) // 2,
                    lm_embedding_size * mapping_network_prefix_length,
                )
            )

            checkpoint_to_load = proj_config.weight_path
    
            if not checkpoint_to_load or checkpoint_to_load == '':
                logger.error(f"No checkpoint found. First time to train {proj_name}...")
            else:
                # We manually load the state dict
                logger.info(f"Loading from {checkpoint_to_load}")
                state_dict_from_ckpt = torch.load(checkpoint_to_load, map_location=self.device)['state_dict']
                model_dict = proj_layer.state_dict()
                # 1. filter out unnecessary keys
                pretrained_dict = {
                    k.replace("model.vision_projection.", ""): v for k, v in state_dict_from_ckpt.items() if k.startswith("model.vision_projection")
                }
                logger.info(f"Load the following parameters from the given checkpoint: {pretrained_dict.keys()}")
                # 2. overwrite entries in the existing state dict
                
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                proj_layer.load_state_dict(model_dict)
            
            # print(pretrained_dict)
            # print('---------------')
            self.vision_projections.append(proj_layer)
            # for n, p in proj_layer.named_parameters():
            #     print(n, p)

        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        
        

    def query(self, input_ids, attention_mask, image_features):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        image_features = image_features.to(self.device)

        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask
        
        # image_features: batch_size (x num_ROIs) x ViT hidden_size
        last_hidden_states = image_features
        batch_size = last_hidden_states.shape[0]
        
        transformed_last_hidden_states = []
        for proj_layer in self.vision_projections:
            projected_embeddings = proj_layer(last_hidden_states) # batch_size (x num_ROIs) x prefix_len*lm_embedding_size
            transformed_last_hidden_states.append(projected_embeddings)
        
        # batch_size x num_ROIs*num_proj x lm_embedding_size
        transformed_last_hidden_states = torch.cat(transformed_last_hidden_states, axis=1)
        
        last_hidden_states = transformed_last_hidden_states.reshape(
            batch_size, -1, self.lm_embedding_size
        )

        Q = torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states

        return torch.nn.functional.normalize(Q, p=2, dim=2)



class ColBERTWithMultimodalDocs(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)

        self.global_config = global_config
        self.model_config = global_config.model_config
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.multimodal_docs = self.model_config.get("multimodal_docs", False)

        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )

        self.doc_vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        if 'freeze_doc_encoder_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network in the document encoder.")
            for name, param in self.doc_vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        
        

    def query(self, input_ids, attention_mask, image_features=None):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask
        
        if image_features is not None:
            image_features = image_features.to(self.device)
            # image_features: batch_size (x num_ROIs) x ViT hidden_size
            last_hidden_states = image_features
            batch_size = last_hidden_states.shape[0]
        
            last_hidden_states = self.vision_projection(last_hidden_states) # bz (x num_ROIs) x 32*128
        
        
            last_hidden_states = last_hidden_states.reshape(
                batch_size, -1, self.lm_embedding_size
            )

            Q = torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states
            # print('Q', Q.shape)
        
        return torch.nn.functional.normalize(Q, p=2, dim=2)


    def doc(self, input_ids, attention_mask, image_features=None, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask

        if self.multimodal_docs and image_features is not None:
            image_features = image_features.to(self.device)
            # image_features: batch_size (x num_ROIs) x ViT hidden_size
            last_hidden_states = image_features
            batch_size = last_hidden_states.shape[0]
        
            last_hidden_states = self.doc_vision_projection(last_hidden_states) # bz (x num_ROIs) x 32*128
        
        
            last_hidden_states = last_hidden_states.reshape(
                batch_size, -1, self.lm_embedding_size
            )
            # print('input_ids', input_ids.shape,  input_ids[:4])
            # print(D[:4])
            # print(D.shape)
            # input()
            # print(image_features[:4])
            # print(image_features.shape)
            # input()
            D = torch.cat([D, last_hidden_states], dim=1) # concatenate hidden states
            image_mask = torch.ones(D.shape[0], D.shape[1] - mask.shape[1], 1).to(self.device)
            # concatenate the mask
            mask = torch.cat([mask, image_mask], dim=1)
            # print('mask', mask.shape)
            # print('D', D.shape)
            

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D


class ColBERTWithMultimodalDocsOnlyImages(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)

        self.global_config = global_config
        self.model_config = global_config.model_config
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.multimodal_docs = self.model_config.get("multimodal_docs", False)

        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )

        self.doc_vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        if 'freeze_doc_encoder_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network in the document encoder.")
            for name, param in self.doc_vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        
        

    def query(self, input_ids, attention_mask, image_features=None):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask
        
        if image_features is not None:
            image_features = image_features.to(self.device)
            # image_features: batch_size (x num_ROIs) x ViT hidden_size
            last_hidden_states = image_features
            batch_size = last_hidden_states.shape[0]
        
            last_hidden_states = self.vision_projection(last_hidden_states) # bz (x num_ROIs) x 32*128
        
        
            last_hidden_states = last_hidden_states.reshape(
                batch_size, -1, self.lm_embedding_size
            )

            Q = torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states
            # print('Q', Q.shape)
        
        return torch.nn.functional.normalize(Q, p=2, dim=2)


    def doc(self, input_ids, attention_mask, image_features=None, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask

        if self.multimodal_docs and image_features is not None:
            image_features = image_features.to(self.device)
            # image_features: batch_size (x num_ROIs) x ViT hidden_size
            last_hidden_states = image_features
            batch_size = last_hidden_states.shape[0]
        
            last_hidden_states = self.doc_vision_projection(last_hidden_states) # bz (x num_ROIs) x 32*128
        
        
            last_hidden_states = last_hidden_states.reshape(
                batch_size, -1, self.lm_embedding_size
            )
            # print('input_ids', input_ids.shape,  input_ids[:4])
            # print(D[:4])
            # print(D.shape)
            # input()
            # print(image_features[:4])
            # print(image_features.shape)
            # input()
            D = last_hidden_states # use image features only
            image_mask = torch.ones(D.shape[0], D.shape[1], 1).to(self.device)
            # concatenate the mask
            mask = image_mask # use image masks only
            # print('mask', mask.shape)
            # print('D', D.shape)
            

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D


class VisualColBERTForPretrainingWithTransformerMapping(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)
        self.global_config = global_config
        self.model_config = global_config.model_config
        
        VisionModelConfigClass = globals()[self.model_config.VisionModelConfigClass]
        VisionModelClass = globals()[self.model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(self.model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(self.model_config.VisionModelVersion, config=vision_model_config)
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        transformer_mapping_config_base = self.model_config.transformer_mapping_config_base
        from transformers import BertConfig
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        from transformers.models.bert.modeling_bert import BertEncoder, BertLayer

        print(transformer_mapping_config)
        self.vision_projection_input_linear = nn.Linear(self.vision_embedding_size, self.vision_embedding_size)
        self.vision_projection = BertEncoder(transformer_mapping_config)
        self.vision_projection_linear = nn.Linear(self.vision_embedding_size, self.lm_embedding_size)
        
        # get weights from bert-base-uncased and assign them to self.vision_projection
        from transformers.models.bert.modeling_bert import BertModel
        bert_model = BertModel.from_pretrained(transformer_mapping_config_base)
        # copy weights to self.vision_projection
        bert_model_state_dict = bert_model.state_dict()
        vision_projection_state_dict = self.vision_projection.state_dict()
        # print("bert_model_state_dict", bert_model_state_dict.keys())
        # print("vision_projection_state_dict", vision_projection_state_dict.keys())
        for k in bert_model_state_dict.keys():
            clean_k = k.replace("encoder.", "")
            if clean_k in vision_projection_state_dict.keys():
                vision_projection_state_dict[clean_k] = bert_model_state_dict[k]
                logger.info(f"copying {clean_k} from bert to vision_projection")
        self.vision_projection.load_state_dict(vision_projection_state_dict)
        del bert_model
        
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_image_encoder' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the ViT image encoder.")
            for name, param in self.vision_model.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        

    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)
        # print("pixel_values", pixel_values.shape)
        # Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Q = self.linear(Q)
        # mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        # Q = Q * mask

        outputs = self.vision_model(pixel_values)
        # print("outputs", outputs.last_hidden_state.shape)
        last_hidden_states = outputs.last_hidden_state # bz x 50 x 768
        last_hidden_states = self.vision_projection_input_linear(last_hidden_states)
        vision_projection_output = self.vision_projection(last_hidden_states)
        last_hidden_states = vision_projection_output.last_hidden_state # bz x 50 x 768
        # print("last_hidden_states", last_hidden_states.shape)
        
        last_hidden_states = self.vision_projection_linear(last_hidden_states) # bz x 50 x 128
        
        Q = last_hidden_states #torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states
        # print("Q", Q.shape)
        # input("forward done")
        return torch.nn.functional.normalize(Q, p=2, dim=2)


class VisualColBERTForPretrainingWithTransformerMappingComposed(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)
        self.global_config = global_config
        self.model_config = global_config.model_config
        
        VisionModelConfigClass = globals()[self.model_config.VisionModelConfigClass]
        VisionModelClass = globals()[self.model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(self.model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(self.model_config.VisionModelVersion, config=vision_model_config)
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        transformer_mapping_config_base = self.model_config.transformer_mapping_config_base

        from transformers import BertConfig
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        from transformers.models.bert.modeling_bert import BertEncoder
        
        # full transformer
        print(transformer_mapping_config)

        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.vision_projection_input_linear = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        
        self.vision_projection_input_linear2 = nn.Linear(self.vision_embedding_size, self.vision_embedding_size)
        
        self.vision_projection = BertEncoder(transformer_mapping_config)
        self.vision_projection_linear = nn.Linear(self.vision_embedding_size, self.lm_embedding_size)
        
        # get weights from bert-base-uncased and assign them to self.vision_projection
        from transformers.models.bert.modeling_bert import BertModel
        bert_model = BertModel.from_pretrained(transformer_mapping_config_base)
        # copy weights to self.vision_projection
        bert_model_state_dict = bert_model.state_dict()
        vision_projection_state_dict = self.vision_projection.state_dict()
        # print("bert_model_state_dict", bert_model_state_dict.keys())
        # print("vision_projection_state_dict", vision_projection_state_dict.keys())
        for k in bert_model_state_dict.keys():
            clean_k = k.replace("encoder.", "")
            if clean_k in vision_projection_state_dict.keys():
                vision_projection_state_dict[clean_k] = bert_model_state_dict[k]
                logger.info(f"copying {clean_k} from bert to vision_projection")
        self.vision_projection.load_state_dict(vision_projection_state_dict)
        del bert_model
        
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_image_encoder' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the ViT image encoder.")
            for name, param in self.vision_model.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
    
    @staticmethod
    def init_weights(module):
        for n, p in module.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        batch_size = input_ids.shape[0]

        # print("pixel_values", pixel_values.shape)
        # Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Q = self.linear(Q)
        # mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        # Q = Q * mask

        outputs = self.vision_model(pixel_values, output_hidden_states=True)
        # print("outputs", outputs.last_hidden_state.shape)
        first_token_hidden_states = outputs.last_hidden_state[:, 0] # bz  x 768
        
        first_token_hidden_states = self.vision_projection_input_linear(first_token_hidden_states) # bz x 32*128
        
        first_token_hidden_states = first_token_hidden_states.view(
            -1, self.mapping_network_prefix_length, self.lm_embedding_size
        )
        # print("first_token_hidden_states", first_token_hidden_states.shape)

        last_hidden_states = outputs.hidden_states[-2][:, 1:] # select the second last layer
        last_hidden_states = self.vision_projection_input_linear2(last_hidden_states)
        
        # print("last_hidden_states", last_hidden_states.shape)
        vision_projection_output = self.vision_projection(last_hidden_states)
        
        last_hidden_states = vision_projection_output.last_hidden_state # bz x 50 x 768
        # print("last_hidden_states", last_hidden_states.shape)
        last_hidden_states = self.vision_projection_linear(last_hidden_states) # bz x 50 x 128
        
        last_hidden_states = torch.cat([first_token_hidden_states, last_hidden_states], dim=1)
        Q = last_hidden_states #torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states
        # print("Q", Q.shape)
        # input("forward done")
        return torch.nn.functional.normalize(Q, p=2, dim=2)




class VisualColBERTForPretrainingWithShallowTransformerMapping(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)
        self.global_config = global_config
        self.model_config = global_config.model_config
        
        VisionModelConfigClass = globals()[self.model_config.VisionModelConfigClass]
        VisionModelClass = globals()[self.model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(self.model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(self.model_config.VisionModelVersion, config=vision_model_config)
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        transformer_mapping_config_base = self.model_config.transformer_mapping_config_base

        from transformers import BertConfig
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        from transformers.models.bert.modeling_bert import BertEncoder
        
        # shallow transformer
        transformer_mapping_config.num_hidden_layers = 1
        print(transformer_mapping_config)

        self.vision_projection_input_linear = nn.Linear(self.vision_embedding_size, self.vision_embedding_size)
        self.vision_projection = BertEncoder(transformer_mapping_config)

        # from transformers import T5Config
        # transformer_mapping_config = T5Config.from_pretrained(transformer_mapping_config_base)
        # from transformers.models.t5.modeling_t5 import T5Stack
        # # shallow transformer
        # transformer_mapping_config.num_layers = 1
        # transformer_mapping_config.is_decoder = False
        # transformer_mapping_config.use_cache = False
        # # self.vision_projection_input_layernorm = nn.LayerNorm(self.vision_embedding_size)
        # # self.vision_projection_input_linear = nn.Linear(self.vision_embedding_size, transformer_mapping_config.d_model)
        # self.vision_projection = T5Stack(transformer_mapping_config, embed_tokens=False)
        
        self.vision_projection_linear = nn.Linear(self.vision_embedding_size, self.lm_embedding_size)
        
        # init model with xavier
        self.init_weights(self.vision_projection)
        # self.init_weights(self.vision_projection_input_linear)
        self.init_weights(self.vision_projection_linear)

        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_image_encoder' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the ViT image encoder.")
            for name, param in self.vision_model.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
    
    @staticmethod
    def init_weights(module):
        for n, p in module.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        batch_size = input_ids.shape[0]

        # print("pixel_values", pixel_values.shape)
        # Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Q = self.linear(Q)
        # mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        # Q = Q * mask

        outputs = self.vision_model(pixel_values, output_hidden_states=True)
        # print("outputs", outputs.last_hidden_state.shape)
        # last_hidden_states = outputs.last_hidden_state # bz x 50 x 768
        last_hidden_states = outputs.hidden_states[-2] # select the second last layer
        
        # last_hidden_states = self.vision_projection_input_layernorm(last_hidden_states)
        # print("last_hidden_states", last_hidden_states.shape)
        last_hidden_states = self.vision_projection_input_linear(last_hidden_states)
        # print("last_hidden_states", last_hidden_states.shape)
        vision_projection_output = self.vision_projection(last_hidden_states)
        
        last_hidden_states = vision_projection_output.last_hidden_state # bz x 50 x 768
        # print("last_hidden_states", last_hidden_states.shape)
        last_hidden_states = self.vision_projection_linear(last_hidden_states) # bz x 50 x 128
        
        Q = last_hidden_states #torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states
        # print("Q", Q.shape)
        # input("forward done")
        return torch.nn.functional.normalize(Q, p=2, dim=2)




class VisualColBERTForPretrainingWithShallowTransformerMappingComposed(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)
        self.global_config = global_config
        self.model_config = global_config.model_config
        
        VisionModelConfigClass = globals()[self.model_config.VisionModelConfigClass]
        VisionModelClass = globals()[self.model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(self.model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(self.model_config.VisionModelVersion, config=vision_model_config)
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        transformer_mapping_config_base = self.model_config.transformer_mapping_config_base

        from transformers import BertConfig
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        from transformers.models.bert.modeling_bert import BertEncoder
        
        # shallow transformer
        transformer_mapping_config.num_hidden_layers = 1
        print(transformer_mapping_config)

        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.vision_projection_input_linear = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        
        self.vision_projection_input_linear2 = nn.Linear(self.vision_embedding_size, self.vision_embedding_size)
        
        self.vision_projection = BertEncoder(transformer_mapping_config)

        # from transformers import T5Config
        # transformer_mapping_config = T5Config.from_pretrained(transformer_mapping_config_base)
        # from transformers.models.t5.modeling_t5 import T5Stack
        # # shallow transformer
        # transformer_mapping_config.num_layers = 1
        # transformer_mapping_config.is_decoder = False
        # transformer_mapping_config.use_cache = False
        # # self.vision_projection_input_layernorm = nn.LayerNorm(self.vision_embedding_size)
        # # self.vision_projection_input_linear = nn.Linear(self.vision_embedding_size, transformer_mapping_config.d_model)
        # self.vision_projection = T5Stack(transformer_mapping_config, embed_tokens=False)
        
        self.vision_projection_linear = nn.Linear(self.vision_embedding_size, self.lm_embedding_size)
        
        # init model with xavier
        # self.init_weights(self.vision_projection)
        # self.init_weights(self.vision_projection_input_linear)
        # self.init_weights(self.vision_projection_linear)

        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_image_encoder' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the ViT image encoder.")
            for name, param in self.vision_model.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        if 'freeze_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.named_parameters():
                if 'vision_projection' in name:
                    print(f"freezed: {name}")
                    param.requires_grad = False

        if 'freeze_flmr' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the FLMR mapping network.")
            for name, param in self.vision_projection_input_linear.named_parameters():
                print(f"freezed: {name}")
                param.requires_grad = False
    
    @staticmethod
    def init_weights(module):
        for n, p in module.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        batch_size = input_ids.shape[0]

        # print("pixel_values", pixel_values.shape)
        # Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Q = self.linear(Q)
        # mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        # Q = Q * mask

        outputs = self.vision_model(pixel_values, output_hidden_states=True)
        # print("outputs", outputs.last_hidden_state.shape)
        first_token_hidden_states = outputs.last_hidden_state[:, 0] # bz  x 768
        
        first_token_hidden_states = self.vision_projection_input_linear(first_token_hidden_states) # bz x 32*128
        
        first_token_hidden_states = first_token_hidden_states.view(
            -1, self.mapping_network_prefix_length, self.lm_embedding_size
        )
        # print("first_token_hidden_states", first_token_hidden_states.shape)

        last_hidden_states = outputs.hidden_states[-2][:, 1:] # select the second last layer
        last_hidden_states = self.vision_projection_input_linear2(last_hidden_states)
        
        # print("last_hidden_states", last_hidden_states.shape)
        vision_projection_output = self.vision_projection(last_hidden_states)
        
        last_hidden_states = vision_projection_output.last_hidden_state # bz x 50 x 768
        # print("last_hidden_states", last_hidden_states.shape)
        last_hidden_states = self.vision_projection_linear(last_hidden_states) # bz x 50 x 128
        
        last_hidden_states = torch.cat([first_token_hidden_states, last_hidden_states], dim=1)
        Q = last_hidden_states #torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states
        # print("Q", Q.shape)
        # input("forward done")
        return torch.nn.functional.normalize(Q, p=2, dim=2)


class VisualColBERTForPretrainingWithShallowTransformerMappingComposedLoRA(ColBERT, ModuleUtilsMixin):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)
        self.global_config = global_config
        self.model_config = global_config.model_config
        from peft import PeftModelForFeatureExtraction, get_peft_config
        lora_config = {
            "peft_type": "LORA",
            "task_type": "FEATURE_EXTRACTION",
            "inference_mode": False,
            "r": 16,
            "target_modules": ["query", "value"],
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "fan_in_fan_out": False,
            "bias": "none",
        }
        peft_config = get_peft_config(lora_config)
        self.model.bert = PeftModelForFeatureExtraction(self.model.bert, peft_config)
        self.model.bert.print_trainable_parameters()

        VisionModelConfigClass = globals()[self.model_config.VisionModelConfigClass]
        VisionModelClass = globals()[self.model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(self.model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(self.model_config.VisionModelVersion, config=vision_model_config)
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        transformer_mapping_config_base = self.model_config.transformer_mapping_config_base

        from transformers import BertConfig
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        from transformers.models.bert.modeling_bert import BertEncoder
        
        # shallow transformer
        transformer_mapping_config.num_hidden_layers = 1
        if 'enable_cross_attention' in self.model_config.modules:
            # add cross attention
            transformer_mapping_config.is_decoder = True
            transformer_mapping_config.add_cross_attention = True
        print(transformer_mapping_config)

        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        self.vision_projection_input_linear = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        
        self.vision_projection_input_linear2 = nn.Linear(self.vision_embedding_size, self.vision_embedding_size)
        
        self.vision_projection = BertEncoder(transformer_mapping_config)

        # from transformers import T5Config
        # transformer_mapping_config = T5Config.from_pretrained(transformer_mapping_config_base)
        # from transformers.models.t5.modeling_t5 import T5Stack
        # # shallow transformer
        # transformer_mapping_config.num_layers = 1
        # transformer_mapping_config.is_decoder = False
        # transformer_mapping_config.use_cache = False
        # # self.vision_projection_input_layernorm = nn.LayerNorm(self.vision_embedding_size)
        # # self.vision_projection_input_linear = nn.Linear(self.vision_embedding_size, transformer_mapping_config.d_model)
        # self.vision_projection = T5Stack(transformer_mapping_config, embed_tokens=False)
        
        self.vision_projection_linear = nn.Linear(self.vision_embedding_size, self.lm_embedding_size)
        
        # init model with xavier
        # self.init_weights(self.vision_projection)
        # self.init_weights(self.vision_projection_input_linear)
        # self.init_weights(self.vision_projection_linear)

        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_image_encoder' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the ViT image encoder.")
            for name, param in self.vision_model.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
    
    @staticmethod
    def init_weights(module):
        for n, p in module.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        batch_size = input_ids.shape[0]

        # print("pixel_values", pixel_values.shape)
        bert_encoded_hidden_states = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(bert_encoded_hidden_states)
        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        outputs = self.vision_model(pixel_values, output_hidden_states=True)
        # print("outputs", outputs.last_hidden_state.shape)
        first_token_hidden_states = outputs.last_hidden_state[:, 0] # bz  x 768
        
        first_token_hidden_states = self.vision_projection_input_linear(first_token_hidden_states) # bz x 32*128
        
        first_token_hidden_states = first_token_hidden_states.view(
            -1, self.mapping_network_prefix_length, self.lm_embedding_size
        )
        # print("first_token_hidden_states", first_token_hidden_states.shape)

        last_hidden_states = outputs.hidden_states[-2][:, 1:] # select the second last layer
        last_hidden_states = self.vision_projection_input_linear2(last_hidden_states)
        
        # print("last_hidden_states", last_hidden_states.shape)
        if 'enable_cross_attention' in self.model_config.modules:
            # print("last_hidden_states", last_hidden_states.shape)
            # print("Q", Q.shape)
            # print("bert_encoded_hidden_states", bert_encoded_hidden_states.shape)
            # print("mask", mask.squeeze(-1).shape)
            encoder_extended_attention_mask = self.invert_attention_mask(mask.squeeze(-1))
            # print('encoder_extended_attention_mask', encoder_extended_attention_mask, encoder_extended_attention_mask.shape)

            vision_projection_output = self.vision_projection(
                last_hidden_states,
                encoder_hidden_states=bert_encoded_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
            )
        else:
            vision_projection_output = self.vision_projection(last_hidden_states)
        
        last_hidden_states = vision_projection_output.last_hidden_state # bz x 50 x 768
        # print("last_hidden_states", last_hidden_states.shape)
        last_hidden_states = self.vision_projection_linear(last_hidden_states) # bz x 50 x 128
        
        concat_hidden_states = torch.cat([first_token_hidden_states, last_hidden_states], dim=1)
        Q = concat_hidden_states
        # print("Q", Q.shape)
        # input("forward done")
        return torch.nn.functional.normalize(Q, p=2, dim=2)


class VisualColBERTForPretrainingWithShallowTransformerMappingMAE(ColBERT):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)
        self.global_config = global_config
        self.model_config = global_config.model_config
        
        # VisionModelConfigClass = globals()[self.model_config.VisionModelConfigClass]
        # VisionModelClass = globals()[self.model_config.VisionModelClass]
        # vision_model_config = VisionModelConfigClass.from_pretrained(self.model_config.VisionModelVersion)
        # self.vision_model = VisionModelClass.from_pretrained(self.model_config.VisionModelVersion, config=vision_model_config)
        
        
        from src.models.models_vit import vit_base_patch16
        self.vision_model = vit_base_patch16(
            global_pool=True,
        )
        self.vision_model.load_state_dict(torch.load('/data/maevit/mae_finetuned_vit_base.pth')['model'])

        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        transformer_mapping_config_base = self.model_config.transformer_mapping_config_base

        from transformers import BertConfig
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        from transformers.models.bert.modeling_bert import BertEncoder
        
        # shallow transformer
        transformer_mapping_config.num_hidden_layers = 1
        print(transformer_mapping_config)

        self.vision_projection_input_linear = nn.Linear(self.vision_embedding_size, self.vision_embedding_size)
        self.vision_projection = BertEncoder(transformer_mapping_config)

        # from transformers import T5Config
        # transformer_mapping_config = T5Config.from_pretrained(transformer_mapping_config_base)
        # from transformers.models.t5.modeling_t5 import T5Stack
        # # shallow transformer
        # transformer_mapping_config.num_layers = 1
        # transformer_mapping_config.is_decoder = False
        # transformer_mapping_config.use_cache = False
        # # self.vision_projection_input_layernorm = nn.LayerNorm(self.vision_embedding_size)
        # # self.vision_projection_input_linear = nn.Linear(self.vision_embedding_size, transformer_mapping_config.d_model)
        # self.vision_projection = T5Stack(transformer_mapping_config, embed_tokens=False)
        
        self.vision_projection_linear = nn.Linear(self.vision_embedding_size, self.lm_embedding_size)
        
        # init model with xavier
        self.init_weights(self.vision_projection)
        # self.init_weights(self.vision_projection_input_linear)
        self.init_weights(self.vision_projection_linear)

        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_image_encoder' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the ViT image encoder.")
            for name, param in self.vision_model.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
    
    @staticmethod
    def init_weights(module):
        for n, p in module.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        batch_size = input_ids.shape[0]

        # print("pixel_values", pixel_values.shape)
        # Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Q = self.linear(Q)
        # mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        # Q = Q * mask

        outputs = self.vision_model.forward_custom(pixel_values)
        # print("outputs", outputs.last_hidden_state.shape)
        last_hidden_states = outputs # bz x 50 x 768
        
        # last_hidden_states = self.vision_projection_input_layernorm(last_hidden_states)
        # print("last_hidden_states", last_hidden_states.shape)
        last_hidden_states = self.vision_projection_input_linear(last_hidden_states)
        # print("last_hidden_states", last_hidden_states.shape)
        vision_projection_output = self.vision_projection(last_hidden_states)
        
        last_hidden_states = vision_projection_output.last_hidden_state # bz x 50 x 768
        # print("last_hidden_states", last_hidden_states.shape)
        last_hidden_states = self.vision_projection_linear(last_hidden_states) # bz x 50 x 128
        
        Q = last_hidden_states #torch.cat([Q, last_hidden_states], dim=1) # concatenate hidden states
        # print("Q", Q.shape)
        # input("forward done")
        return torch.nn.functional.normalize(Q, p=2, dim=2)






################# Place for new models #################

class VisualColBERTForPretrainingWithShallowTransformerMappingComposedWithCrossAttn(ColBERT, ModuleUtilsMixin):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config)
        self.global_config = global_config
        self.model_config = global_config.model_config
        
        VisionModelConfigClass = globals()[self.model_config.VisionModelConfigClass]
        VisionModelClass = globals()[self.model_config.VisionModelClass]
        vision_model_config = VisionModelConfigClass.from_pretrained(self.model_config.VisionModelVersion)
        self.vision_model = VisionModelClass.from_pretrained(self.model_config.VisionModelVersion, config=vision_model_config)
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        
        transformer_mapping_config_base = self.model_config.transformer_mapping_config_base

        from transformers import BertConfig
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        from transformers.models.bert.modeling_bert import BertEncoder
        # shallow transformer
        transformer_mapping_config.num_hidden_layers = self.model_config.get('transformer_mapping_num_hidden_layers', 1)
        # add cross attention
        if 'enable_cross_attention' in self.model_config.modules:
            transformer_mapping_config.is_decoder = True
            transformer_mapping_config.add_cross_attention = True
        
        self.mapping_network_prefix_length = self.model_config.mapping_network_prefix_length
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size
        self.lm_hidden_embedding_size = self.bert.config.hidden_size
        
        self.vision_projection_input_linear = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )
        
        
        self.vision_projection_input_linear2 = nn.Linear(self.vision_embedding_size, self.lm_hidden_embedding_size)
        
        self.vision_projection = BertEncoder(transformer_mapping_config)

        # from transformers import T5Config
        # transformer_mapping_config = T5Config.from_pretrained(transformer_mapping_config_base)
        # from transformers.models.t5.modeling_t5 import T5Stack
        # # shallow transformer
        # transformer_mapping_config.num_layers = 1
        # transformer_mapping_config.is_decoder = False
        # transformer_mapping_config.use_cache = False
        # # self.vision_projection_input_layernorm = nn.LayerNorm(self.vision_embedding_size)
        # # self.vision_projection_input_linear = nn.Linear(self.vision_embedding_size, transformer_mapping_config.d_model)
        # self.vision_projection = T5Stack(transformer_mapping_config, embed_tokens=False)
        
        self.vision_projection_linear = nn.Linear(self.lm_hidden_embedding_size, self.lm_embedding_size)
        
        # init model with xavier
        # self.init_weights(self.vision_projection)
        # self.init_weights(self.vision_projection_input_linear)
        # self.init_weights(self.vision_projection_linear)
        from copy import deepcopy
        
        if "separate_question_encoder" in self.model_config.modules:
            self.query_encoder = deepcopy(self.bert)
            self.query_linear = deepcopy(self.linear)
            
            # # freeze parameters of the document encoder
            # for name, param in self.bert.named_parameters():
            #     # print(f"freezed: {name}")
            #     param.requires_grad = False
            # for name, param in self.linear.named_parameters():
            #     # print(f"freezed: {name}")
            #     param.requires_grad = False
        else:
            self.query_encoder = self.bert
            self.query_linear = self.linear

        if 'enable_doc_encoder_lora' in self.model_config.modules:
            from peft import PeftModelForFeatureExtraction, get_peft_config
            lora_config = {
                "peft_type": "LORA",
                "task_type": "FEATURE_EXTRACTION",
                "inference_mode": False,
                "r": 16,
                "target_modules": ["query", "value"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "fan_in_fan_out": False,
                "bias": "none",
            }
            peft_config = get_peft_config(lora_config)
            self.model.bert = PeftModelForFeatureExtraction(self.model.bert, peft_config)
            self.model.bert.print_trainable_parameters()
            
            if "separate_question_encoder" in self.model_config.modules:
                logger.warning("separate_question_encoder is enabled. The query encoder is also initialized with LORA.")
                self.query_encoder = PeftModelForFeatureExtraction(self.query_encoder, peft_config)
                self.query_encoder.print_trainable_parameters()
            
            
        if 'freeze_colbert_doc_encoder' in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning("freezing the ColBERT document encoder. If the query encoder is not separated, the query encoder is also frozen.")
            for name, param in self.bert.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
            for name, param in self.linear.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False
        
        if 'freeze_image_encoder' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the ViT image encoder.")
            for name, param in self.vision_model.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        if 'freeze_mapping_network' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.named_parameters():
                if 'vision_projection' in name:
                    print(f"freezed: {name}")
                    param.requires_grad = False

        if 'freeze_flmr' in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the FLMR mapping network.")
            for name, param in self.vision_projection_input_linear.named_parameters():
                print(f"freezed: {name}")
                param.requires_grad = False
                
        self.mask_instruction = ('mask_instruction' in self.model_config.modules)
    
    @staticmethod
    def init_weights(module):
        for n, p in module.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def query_mask(self, input_ids, skiplist):
        if not self.mask_instruction:
            return super().mask(input_ids, skiplist)
        # print(input_ids.tolist())
        # find the position of ":" in input_ids
        # mask the tokens before the position
        sep_id = 1024
        sep_positions = torch.argmax((input_ids == sep_id).int(), dim=1).tolist()
        # if any of the positions is lower than 1, set to 1
        for i, x in enumerate(sep_positions):
            if x < 1:
                sep_positions[i] = 1
                logger.error(f"can not find the separator in the input_ids: {input_ids[i].tolist()}")
        # sep_positions = [max(1, x) for x in sep_positions] # as a safety pin
        # print(sep_positions)
        mask = [
            [(x not in skiplist) and (x != 0) and (index > sep_positions[seq_index] or index < 2) for index, x in enumerate(d)] for seq_index, d in enumerate(input_ids.cpu().tolist())
        ]
        # print(mask)
        # input()
        return mask
    
    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        batch_size = input_ids.shape[0]

        # print("pixel_values", pixel_values.shape)
        bert_encoded_hidden_states = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(bert_encoded_hidden_states)
        mask = torch.tensor(self.query_mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        outputs = self.vision_model(pixel_values, output_hidden_states=True)
        # print("outputs", outputs.last_hidden_state.shape)
        first_token_hidden_states = outputs.last_hidden_state[:, 0] # bz  x 768
        
        first_token_hidden_states = self.vision_projection_input_linear(first_token_hidden_states) # bz x 32*128
        
        first_token_hidden_states = first_token_hidden_states.view(
            -1, self.mapping_network_prefix_length, self.lm_embedding_size
        )
        # print("first_token_hidden_states", first_token_hidden_states.shape)

        last_hidden_states = outputs.hidden_states[-2][:, 1:] # select the second last layer
        last_hidden_states = self.vision_projection_input_linear2(last_hidden_states)
        if 'enable_cross_attention' in self.model_config.modules:
            # print("last_hidden_states", last_hidden_states.shape)
            # print("Q", Q.shape)
            # print("bert_encoded_hidden_states", bert_encoded_hidden_states.shape)
            # print("mask", mask.squeeze(-1).shape)
            encoder_extended_attention_mask = self.invert_attention_mask(mask.squeeze(-1))
            # print('encoder_extended_attention_mask', encoder_extended_attention_mask, encoder_extended_attention_mask.shape)
            # print("last_hidden_states", last_hidden_states.shape)
            # print("bert_encoded_hidden_states", bert_encoded_hidden_states.shape)
            # print("encoder_extended_attention_mask", encoder_extended_attention_mask.shape)
            
            vision_projection_output = self.vision_projection(
                last_hidden_states,
                encoder_hidden_states=bert_encoded_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
            )
        else:
            vision_projection_output = self.vision_projection(last_hidden_states)
        
        last_hidden_states = vision_projection_output.last_hidden_state # bz x 50 x 768
        # print("last_hidden_states", last_hidden_states.shape)
        
        last_hidden_states = self.vision_projection_linear(last_hidden_states) # bz x 50 x 128
        
        concat_hidden_states = torch.cat([first_token_hidden_states, last_hidden_states], dim=1)
        Q = concat_hidden_states
        # print("Q", Q.shape)
        # input("forward done")
        return torch.nn.functional.normalize(Q, p=2, dim=2)


class VisualColBERTForRetrievalWithShallowTransformerMappingComposed(VisualColBERTForPretrainingWithShallowTransformerMappingComposedWithCrossAttn):
    def __init__(self, name='bert-base-uncased', colbert_config=None, global_config=None):
        super().__init__(name, colbert_config, global_config=global_config)
        
    
    def query(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)
        # print("input_ids", input_ids.shape)
        # print("attention_mask", attention_mask.shape)
        # print("pixel_values", pixel_values.shape)

        batch_size = input_ids.shape[0]

        # print("pixel_values", pixel_values.shape)
        bert_encoded_hidden_states = self.query_encoder(input_ids, attention_mask=attention_mask)[0]
        Q = self.query_linear(bert_encoded_hidden_states)
        mask = torch.tensor(self.query_mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask
        
        outputs = self.vision_model(pixel_values, output_hidden_states=True)
        # print("outputs", outputs.last_hidden_state.shape)
        first_token_hidden_states = outputs.last_hidden_state[:, 0] # bz  x 768
        
        first_token_hidden_states = self.vision_projection_input_linear(first_token_hidden_states) # bz x 32*128
        
        first_token_hidden_states = first_token_hidden_states.view(
            -1, self.mapping_network_prefix_length, self.lm_embedding_size
        )
        # print("first_token_hidden_states", first_token_hidden_states.shape)

        last_hidden_states = outputs.hidden_states[-2][:, 1:] # select the second last layer
        last_hidden_states = self.vision_projection_input_linear2(last_hidden_states)
        if 'enable_cross_attention' in self.model_config.modules:
            # print("last_hidden_states", last_hidden_states.shape)
            # print("Q", Q.shape)
            # print("bert_encoded_hidden_states", bert_encoded_hidden_states.shape)
            # print("mask", mask.squeeze(-1).shape)
            encoder_mask = torch.ones_like(mask).to(mask.device, dtype=mask.dtype)
            # encoder_mask = mask
            if bert_encoded_hidden_states.shape[1] > 32:
                bert_encoded_hidden_states = bert_encoded_hidden_states[:, :32]
                encoder_mask = encoder_mask[:, :32]
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_mask.squeeze(-1))
            # print('encoder_extended_attention_mask', encoder_extended_attention_mask, encoder_extended_attention_mask.shape)
            
            vision_projection_output = self.vision_projection(
                last_hidden_states,
                encoder_hidden_states=bert_encoded_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
            )
        else:
            vision_projection_output = self.vision_projection(last_hidden_states)
        
        last_hidden_states = vision_projection_output.last_hidden_state # bz x 50 x 768
        # print("last_hidden_states", last_hidden_states.shape)
        
        last_hidden_states = self.vision_projection_linear(last_hidden_states) # bz x 50 x 128
        
        concat_hidden_states = torch.cat([Q, first_token_hidden_states, last_hidden_states], dim=1)
        # print(Q.shape, first_token_hidden_states.shape, last_hidden_states.shape)
        Q = concat_hidden_states
        # print("Q", Q.shape)
        # input("forward done")
        return torch.nn.functional.normalize(Q, p=2, dim=2)