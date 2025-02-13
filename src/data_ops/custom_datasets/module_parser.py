from ast import Raise
from typing import Optional
from easydict import EasyDict
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import random

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class ModuleParser:
    """
    This is a module inherited by the dataset class
    This class is used to parse the sample to form input/output/decoder_input data
    It should be able to process both text-based features and image-based features
    Process:
        (1) Sample-level Sub Parsers:
            Add data fields to the sample
        (2) Processing:
            Aggregating features from individual sub parsers
            Strings under the same field will be automatically concatenated
            Use different fields for different image-based features
        (3) Post-processing:
            Add post-processing units to process the data after parsing
            e.g. tokenization, adding new sample-level features
    """

    def __init__(self) -> None:
        pass

    def QuestionInput(self, sample: EasyDict, module: EasyDict) -> EasyDict:
        """
        Parse the question input
        Simple add the question to the text sequence
        """
        return_dict = EasyDict(
            text_sequence="",
        )
        if module.option == "default":
            input_sequence = " ".join(
                [module.separation_tokens.start]
                + [sample.question]
                + [module.separation_tokens.end]
            )

        return_dict.text_sequence = input_sequence
        return return_dict

    def InstructionInput(self, sample: EasyDict, module: EasyDict) -> EasyDict:
        """
        Add instruction to the text sequence
        """
        return_dict = EasyDict(
            text_sequence="",
        )
        if module.option == "default":
            if sample.get("question", None) is not None:
                input_sequence = " ".join(
                    [module.separation_tokens.start]
                    + [sample.question]
                    + [module.separation_tokens.end]
                )
            else:
                prompt = random.choice(module.prompts)
                input_sequence = " ".join(
                    [module.separation_tokens.start]
                    + [prompt]
                    + [module.separation_tokens.end]
                )

        return_dict.text_sequence = input_sequence.strip()
        return return_dict

    def EmptyTextInput(self, sample: EasyDict, module: EasyDict) -> EasyDict:
        """
        Return an empty input
        """
        return_dict = EasyDict(
            text_sequence="",
        )
        return return_dict

    def TextBasedVisionInput(
        self, sample: EasyDict, module: EasyDict
    ) -> Optional[EasyDict]:
        """
        Default TextBasedVisionInput module parser
        object: text-based objects, with attributes and OCR'ed texts
        caption: iamge captions
        """
        return_dict = EasyDict(
            text_sequence="",
        )

        # Input from Vision
        vision_sentences = []
        if module.option == "object":
            vision_sentences += [module.separation_tokens.start]
            for obj in sample.objects:
                attribute_max = module.get("attribute_max", 0)
                if attribute_max > 0:
                    # find suitable attributes
                    suitable_attributes = []
                    for attribute, att_score in zip(
                        obj["attributes"], obj["attribute_scores"]
                    ):
                        if (
                            att_score > module.attribute_thres
                            and len(suitable_attributes) < attribute_max
                        ):
                            suitable_attributes.append(attribute)
                    # append to the sentence
                    vision_sentences += suitable_attributes
                vision_sentences.append(obj["class"])
                vision_sentences.append(module.separation_tokens.sep)

            ocr = module.get("ocr", 0)
            if ocr > 0:
                text_annotations = sample.img_ocr
                filtered_descriptions = []
                for text_annoation in text_annotations:
                    description = text_annoation["description"].strip()
                    description = description.replace(
                        "\n", " "
                    )  # remove line switching
                    # vision_sentences += [description]
                    # print('OCR feature:', description)
                    if description not in filtered_descriptions:
                        filtered_descriptions.append(description)
                # print('OCR feature:', filtered_descriptions)
                vision_sentences += filtered_descriptions

            vision_sentences += [module.separation_tokens.end]
            return_dict.text_sequence = " ".join(vision_sentences)

        elif module.option == "caption":
            if isinstance(sample.img_caption, dict):
                caption = sample.img_caption["caption"]
            else:
                caption = sample.img_caption
            return_dict.text_sequence = " ".join(
                [module.separation_tokens.start]
                + [caption]
                + [module.separation_tokens.end]
            )

        return return_dict

    def GenerationOutput(
        self, sample: EasyDict, module: EasyDict
    ) -> Optional[EasyDict]:
        """
        Parse the default generation output from gold_answer
        """
        return_dict = EasyDict(
            text_sequence="",
        )
        output_sequence = sample.gold_answer
        return_dict.text_sequence = output_sequence
        return return_dict

    def VisionInput(self, sample: EasyDict, module: EasyDict) -> Optional[EasyDict]:
        """
        Parse the vision input
        """
        img_path = sample.img_path
        if module.option == "from_file":
            # img = cv2.imread(img_path)
            if img_path is None:
                # Create a black image of size 512*512
                img = np.zeros((512, 512, 3), np.uint8)
                # Convert to PIL Image
                img = Image.fromarray(img)
            else:
                img = Image.open(img_path).convert("RGB")

            if module.get("resize", None) is not None:
                img = img.resize((module.resize, module.resize))
            return_dict = EasyDict(
                img=img,
            )
        elif module.option == "from_embeddings":
            use_ROI = module.get("use_ROI", False)
            if use_ROI and self.config.model_config.get("num_ROIs", 0) != 0:
                num_ROIs = self.config.model_config.get("num_ROIs", 0)
                img = self.image_dataset_with_embeddings[img_path]
                # add ROI features with the global image features
                ROIs = sample.ROIs
                image_features = []
                image_features.append(
                    np.array(img["image_features"])
                )  # global image features
                # print(img_path, img['image_features'].shape)
                if num_ROIs > len(ROIs):
                    ROIs = ROIs + [ROIs[-1]] * (num_ROIs - len(ROIs))
                for ROI in ROIs[:num_ROIs]:
                    image_features.append(
                        np.array(
                            self.image_dataset_with_embeddings[ROI]["image_features"]
                        )
                    )  # ROI image features
                    # print(ROI, self.image_dataset_with_embeddings[ROI]['image_features'].shape)
                stacked_image_features = np.stack(image_features)

                img = {
                    "image_features": stacked_image_features,
                }
                return_dict = EasyDict(
                    image_features=img,
                )
            else:
                image_features = self.image_dataset_with_embeddings[img_path]
                return_dict = EasyDict(
                    image_features=image_features,
                )
        elif module.option == "path_only":
            return_dict = EasyDict(
                img_path=img_path,
            )
        else:
            img = self.images[img_path]
            return_dict = EasyDict(
                img=img,
            )

        return return_dict

    def KnowledgeInput(self, sample: EasyDict, module: EasyDict) -> Optional[EasyDict]:
        """
        Parse the knowledge input
        """
        return_dict = EasyDict(
            text_sequence="",
        )
        return_dict.text_sequence = " ".join(
            [module.separation_tokens.start]
            + [sample.passage_content]
            + [module.separation_tokens.end]
        )
        return return_dict

    def PassageVisionInput(
        self, sample: EasyDict, module: EasyDict
    ) -> Optional[EasyDict]:
        """
        Parse the vision input
        """
        passage_id = sample.passage_id
        # print('passage_id:', passage_id)
        img_path = self.passages.id2img_path[str(passage_id)]
        # print('img_path:', img_path)
        if module.option == "from_file":
            img = cv2.imread(img_path)
        elif module.option == "from_embeddings":
            img = self.image_dataset_with_embeddings[img_path]
        else:
            img = self.images[img_path]

        return_dict = EasyDict(
            img=img,
        )
        return return_dict

    def SimilarityOutput(
        self, sample: EasyDict, module: EasyDict
    ) -> Optional[EasyDict]:
        """
        Generate the similarity output
        """
        label = [1]
        label += [0] * len(sample.neg_passage_ids)
        return_dict = EasyDict(
            label=label,
        )
        return return_dict

    def parse_modules(
        self,
        sample: EasyDict,
        modules: EasyDict,
        type: str,
        process_modules: Optional[EasyDict] = None,
    ) -> Optional[EasyDict]:
        """
        Parse the sample to form input/output/decoder_input
        Args:
            sample: sample to be parsed
            modules: modules to be parsed
            type: type of the module
        Returns:
            parsed sample
        """
        data_collection = []
        if type == "input":
            for input_module in modules:
                parser_func = getattr(self, input_module.type)
                parsed_data = parser_func(sample, input_module)
                data_collection.append(parsed_data)
        elif type == "decoder_input":
            for input_module in modules:
                parser_func = getattr(self, input_module.type)
                parsed_data = parser_func(sample, input_module)
                data_collection.append(parsed_data)
        elif type == "output":
            for output_module in modules:
                parser_func = getattr(self, output_module.type)
                parsed_data = parser_func(sample, output_module)
                data_collection.append(parsed_data)
        else:
            raise ValueError("Unknown type: {}".format(type))

        # Process the sample data after aggregating from individual sub parsers
        # before returning to colln_func
        processed_data = data_collection
        if process_modules is None:
            # Run default processing unit
            processed_data = self.DefaultProcessing(processed_data)
        else:
            # Run provided processing unit
            for process_module in process_modules:
                process_func = getattr(self, process_module.type)
                processed_data = process_func(processed_data)

        return processed_data

    def DefaultProcessing(self, data_to_process: EasyDict) -> EasyDict:
        """
        Process the sample data after aggregating from individual sub parsers
        """
        processed_data = EasyDict()
        for data_entry in data_to_process:
            for key, value in data_entry.items():
                if key not in processed_data:
                    processed_data[key] = value
                else:
                    if type(value) == str:
                        # automatically concatenate strings with the same key
                        processed_data[key] += " " + value
                    else:
                        raise TypeError(
                            "Undefined processing type: {}".format(type(value))
                        )

        return processed_data

    def PostProcessInputTokenization(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for input tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        task_prefix = ""
        encoding = self.tokenizer(
            [task_prefix + sequence for sequence in text_sequences],
            padding="longest",
            max_length=self.config.model_config.max_source_length,
            truncation=True,
            return_tensors="pt",
        )
        data_to_process.update(
            {
                "input_ids": encoding.input_ids,
                "attention_mask": encoding.attention_mask,
                "input_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessDecoderInputTokenization(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing for decoder input tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        encoding = self.decoder_tokenizer(
            [sequence for sequence in text_sequences],
            padding="longest",
            max_length=self.config.model_config.max_decoder_source_length,
            truncation=True,
            return_tensors="pt",
        )
        data_to_process.update(
            {
                "decoder_input_ids": encoding.input_ids,
                "decoder_input_attention_mask": encoding.attention_mask,
                "decoder_input_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessOutputTokenization(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for output tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        target_encoding = self.decoder_tokenizer(
            text_sequences,
            padding="longest",
            max_length=self.config.model_config.max_target_length,
            truncation=True,
        )
        labels = target_encoding.input_ids
        output_sequence_ids = target_encoding.input_ids  # For teacher force training
        output_sequence_ids = torch.LongTensor(output_sequence_ids)
        output_sequence_attention_mask = torch.LongTensor(
            target_encoding.attention_mask
        )  # For teacher force training

        # replace padding token id's of the labels by -100
        labels = [
            [
                (label if label != self.decoder_tokenizer.pad_token_id else -100)
                for label in labels_example
            ]
            for labels_example in labels
        ]

        labels = torch.LongTensor(labels)
        assert labels.shape == output_sequence_ids.shape

        data_to_process.update(
            {
                "labels": labels,
                "output_sequence_ids": output_sequence_ids,
                "output_sequence_attention_mask": output_sequence_attention_mask,
                "output_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessBlipOutputTokenization(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for output tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        target_encoding = self.decoder_tokenizer.tokenizer(
            text_sequences,
            padding="longest",
            max_length=self.config.model_config.max_target_length,
            truncation=True,
        )
        labels = target_encoding.input_ids
        output_sequence_ids = target_encoding.input_ids  # For teacher force training
        output_sequence_ids = torch.LongTensor(output_sequence_ids)
        output_sequence_attention_mask = torch.LongTensor(
            target_encoding.attention_mask
        )  # For teacher force training

        # replace padding token id's of the labels by -100
        labels = [
            [
                (
                    label
                    if label != self.decoder_tokenizer.tokenizer.pad_token_id
                    else -100
                )
                for label in labels_example
            ]
            for labels_example in labels
        ]

        labels = torch.LongTensor(labels)
        assert labels.shape == output_sequence_ids.shape

        data_to_process.update(
            {
                "labels": labels,
                "output_sequence_ids": output_sequence_ids,
                "output_sequence_attention_mask": output_sequence_attention_mask,
                "output_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessColBERTQuestionInputTokenization(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing for input tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        task_prefix = ""
        self.tokenizer.query_maxlen = self.config.model_config.max_source_length
        Q_ids, Q_mask = self.tokenizer.tensorize(
            [task_prefix + sequence for sequence in text_sequences]
        )
        data_to_process.update(
            {
                "input_ids": Q_ids,
                "attention_mask": Q_mask,
                "input_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessFLMRQuestionInputTokenization(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing for input tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        task_prefix = ""
        self.tokenizer.query_maxlen = self.config.model_config.max_source_length
        encoding = self.tokenizer(
            [task_prefix + sequence for sequence in text_sequences]
        )
        Q_ids, Q_mask = encoding["input_ids"], encoding["attention_mask"]

        data_to_process.update(
            {
                "input_ids": Q_ids,
                "attention_mask": Q_mask,
                "input_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessLLaVAQuestionInputTokenization(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing for input tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        task_prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: "

        # The encoding of LLaVA requires that ".:" are all transformed
        text_sequences = [
            sequence.replace(".:", ":").replace("?:", ":") + " <image>\n"
            for sequence in text_sequences
        ]

        encoding = self.tokenizer(
            [task_prefix + sequence for sequence in text_sequences],
            max_length=self.config.model_config.max_source_length,
            truncation=True,
            padding="longest",
        )
        Q_ids, Q_mask = encoding["input_ids"], encoding["attention_mask"]

        Q_ids = torch.LongTensor(Q_ids)
        Q_mask = torch.LongTensor(Q_mask)

        sep_id = 29901
        img_token_id = self.tokenizer.encode("<image>")[1]
        img_token_span = 1176

        # print("sep_id", sep_id, self.tokenizer.decode(sep_id, skip_special_tokens=True))

        # print(self.tokenizer.batch_decode(Q_ids, skip_special_tokens=False))

        image_mask = []
        instruction_mask = []
        question_mask = []

        # for Q in Q_ids:
        #     for token in Q:
        #         print(f"{token} -> {self.tokenizer.decode(token, skip_special_tokens=True)}")

        for Q in Q_ids:
            # the second. : appears in the task prefix
            # first_start_index = torch.where(Q==sep_id)[0][0]
            # print(torch.where(Q==sep_id))
            sep_index = torch.where(Q == sep_id)[0][1]
            img_token_index = torch.where(Q == img_token_id)[0][0]
            max_len = len(Q)
            image_mask.append(
                [0] * img_token_index
                + [1] * img_token_span
                + [0] * (max_len - img_token_index - 1)
            )
            instruction_mask.append(
                [1] * (sep_index + 1) + [0] * (max_len + img_token_span - sep_index)
            )
            question_mask.append(
                [0] * (sep_index + 1)
                + [1] * (img_token_index - sep_index - 1)
                + [0] * (max_len + img_token_span - img_token_index - 1)
            )

            # print(sep_index)
            # print(self.tokenizer.batch_decode(Q[:sep_index], skip_special_tokens=False))
            # print(self.tokenizer.batch_decode(Q[sep_index+1:], skip_special_tokens=False))
            # input()

        # print(image_mask)
        # print(instruction_mask)
        # print(question_mask)
        # print("===============================")

        image_mask = torch.LongTensor(image_mask)
        question_mask = torch.LongTensor(question_mask)
        instruction_mask = torch.LongTensor(instruction_mask)

        data_to_process.update(
            {
                "input_ids": Q_ids,
                "attention_mask": Q_mask,
                "input_text_sequences": text_sequences,
                "image_mask": image_mask,
                "instruction_mask": instruction_mask,
                "question_mask": question_mask,
            }
        )
        return data_to_process

    def PostProcessColBERTItemInputTokenization(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing for decoder input tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        self.decoder_tokenizer.doc_maxlen = (
            self.config.model_config.max_decoder_source_length
        )
        D_ids, D_mask = self.decoder_tokenizer.tensorize(text_sequences)
        data_to_process.update(
            {
                "decoder_input_ids": D_ids,
                "decoder_input_attention_mask": D_mask,
                "decoder_input_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessFLMRItemInputTokenization(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing for decoder input tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        self.decoder_tokenizer.doc_maxlen = (
            self.config.model_config.max_decoder_source_length
        )
        encoding = self.decoder_tokenizer(text_sequences)
        D_ids, D_mask = encoding["input_ids"], encoding["attention_mask"]
        data_to_process.update(
            {
                "decoder_input_ids": D_ids,
                "decoder_input_attention_mask": D_mask,
                "decoder_input_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessQWenQuestionInputTokenization(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing for input tokenization
        """
        assert "img_path" in data_to_process.keys()
        img_path = data_to_process.pop("img_path")

        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        task_prefix = ""
        self.tokenizer.pad_token_id = self.tokenizer.eod_id

        def preprocess(
            sources,
            tokenizer,
            max_len: int,
            system_message: str = "You are a helpful assistant.",
        ):
            roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

            im_start = tokenizer.im_start_id
            im_end = tokenizer.im_end_id
            nl_tokens = tokenizer("\n").input_ids
            _system = tokenizer("system").input_ids + nl_tokens
            _user = tokenizer("user").input_ids + nl_tokens
            _assistant = tokenizer("assistant").input_ids + nl_tokens
            _separation = tokenizer(":").input_ids
            _img_token = tokenizer.img_end_id

            batch_max = max_len
            max_len = 512  # tolerance for errors

            # Apply prompt templates
            input_ids, targets = [], []
            image_masks = []
            instruction_masks = []
            question_masks = []

            for i, source in enumerate(sources):
                if roles[source[0]["from"]] != roles["user"]:
                    source = source[1:]

                input_id, target = [], []
                system = (
                    [im_start]
                    + _system
                    + tokenizer(system_message).input_ids
                    + [im_end]
                    + nl_tokens
                )
                input_id += system
                target += (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * (len(system) - 3)
                    + [im_end]
                    + nl_tokens
                )
                assert len(input_id) == len(target)

                image_mask = [0] * len(input_id)
                instruction_mask = [0] * len(input_id)
                question_mask = [0] * len(input_id)

                for j, sentence in enumerate(source):
                    role = roles[sentence["from"]]
                    _input_id = (
                        tokenizer(role).input_ids
                        + nl_tokens
                        + tokenizer(sentence["value"]).input_ids
                        + [im_end]
                        + nl_tokens
                    )
                    input_id += _input_id

                    first_part = tokenizer(role).input_ids + nl_tokens

                    second_part = tokenizer(sentence["value"]).input_ids
                    last_part = [im_end] + nl_tokens

                    if _img_token in second_part:
                        image_part = second_part[: second_part.index(_img_token) + 2]
                        instruction_and_question_part = second_part[
                            second_part.index(_img_token) + 2 :
                        ]
                    else:
                        image_part = []
                        instruction_and_question_part = second_part

                    # split the second part with _separation token
                    if _separation[0] in instruction_and_question_part:
                        instruction_part = instruction_and_question_part[
                            : instruction_and_question_part.index(_separation[0])
                        ]
                        question_part = instruction_and_question_part[
                            instruction_and_question_part.index(_separation[0]) + 1 :
                        ]
                    else:
                        instruction_part = []
                        question_part = instruction_and_question_part

                    # print("first_part", first_part)
                    # print(tokenizer.decode(first_part))
                    # print("image_part", image_part)
                    # print(tokenizer.decode(image_part))
                    # print("instruction_part", instruction_part)
                    # print(tokenizer.decode(instruction_part))
                    # print("question_part", question_part)
                    # print(tokenizer.decode(question_part))
                    # print("last_part", last_part)
                    # print(tokenizer.decode(last_part))
                    # input()

                    image_mask += (
                        [0] * len(first_part)
                        + [1] * len(image_part)
                        + [0] * len(instruction_part)
                        + [0] * len(question_part)
                        + [0] * len(last_part)
                    )
                    instruction_mask += (
                        [0] * len(first_part)
                        + [0] * len(image_part)
                        + [1] * len(instruction_part)
                        + [0] * len(question_part)
                        + [0] * len(last_part)
                    )
                    question_mask += (
                        [0] * len(first_part)
                        + [0] * len(image_part)
                        + [0] * len(instruction_part)
                        + [1] * len(question_part)
                        + [0] * len(last_part)
                    )

                    if role == "<|im_start|>user":
                        _target = (
                            [im_start]
                            + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                            + [im_end]
                            + nl_tokens
                        )
                    elif role == "<|im_start|>assistant":
                        _target = (
                            [im_start]
                            + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids)
                            + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                            + [im_end]
                            + nl_tokens
                        )
                    else:
                        raise NotImplementedError
                    target += _target

                assert len(input_id) == len(target)

                # if len(input_id) > batch_max:
                #     batch_max = len(input_id)

                input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
                target += [IGNORE_TOKEN_ID] * (max_len - len(target))
                image_mask += [0] * (max_len - len(image_mask))
                instruction_mask += [0] * (max_len - len(instruction_mask))
                question_mask += [0] * (max_len - len(question_mask))

                # print(tokenizer.decode(input_id[:batch_max]))
                # print(image_mask[:batch_max])
                # print(instruction_mask[:batch_max])
                # print(question_mask[:batch_max])
                # print(sum(image_mask[:batch_max]))
                # print(sum(instruction_mask[:batch_max]))
                # print(sum(question_mask[:batch_max]))

                # input()

                input_ids.append(input_id[:max_len])
                targets.append(target[:max_len])
                image_masks.append(image_mask[:max_len])
                instruction_masks.append(instruction_mask[:max_len])
                question_masks.append(question_mask[:max_len])

            input_ids = torch.tensor(input_ids, dtype=torch.int)[:, :batch_max]
            targets = torch.tensor(targets, dtype=torch.long)[:, :batch_max]
            image_masks = torch.tensor(image_masks, dtype=torch.int)[:, :batch_max]
            instruction_masks = torch.tensor(instruction_masks, dtype=torch.int)[
                :, :batch_max
            ]
            question_masks = torch.tensor(question_masks, dtype=torch.int)[
                :, :batch_max
            ]

            # print(question_masks.shape)

            return dict(
                input_ids=input_ids,
                labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
                image_mask=image_masks,
                instruction_mask=instruction_masks,
                question_mask=question_masks,
            )

        sources = []

        for img, text_sequence in zip(img_path, text_sequences):
            input_sequence = []
            if img is not None:
                input_sequence.append(f"<img>{img}</img>\n")

            input_sequence.append(text_sequence)

            sources.append([{"from": "user", "value": " ".join(input_sequence)}])

        encoding = preprocess(
            sources, self.tokenizer, self.config.model_config.max_source_length
        )
        input_ids, labels, attention_mask = (
            encoding["input_ids"],
            encoding["labels"],
            encoding["attention_mask"],
        )
        image_mask, instruction_mask, question_mask = (
            encoding["image_mask"],
            encoding["instruction_mask"],
            encoding["question_mask"],
        )
        # from pprint import pprint
        # pprint(self.tokenizer.batch_decode(input_ids, skip_special_tokens=False))
        # input()
        data_to_process.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "input_text_sequences": text_sequences,
                "image_mask": image_mask,
                "instruction_mask": instruction_mask,
                "question_mask": question_mask,
            }
        )
        return data_to_process

    def PostProcessConcatenateLabels(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for concatenating labels
        """
        assert "label" in data_to_process.keys()
        label = data_to_process.pop("label")
        labels = []
        for l in label:
            labels += l
        data_to_process.update(
            {
                "labels": torch.LongTensor(labels),
            }
        )
        return data_to_process

    def PostProcessVisionInputProcessing(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for input tokenization
        """
        assert "img" in data_to_process.keys()
        images = data_to_process.pop("img")
        inputs = self.vit_image_processor(images, return_tensors="pt")
        data_to_process.update(**inputs)

        return data_to_process

    def PostProcessBlip2VisionInputProcessing(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing for input tokenization
        """
        assert "img" in data_to_process.keys()
        images = data_to_process["img"]
        inputs = self.decoder_tokenizer(images, return_tensors="pt")
        data_to_process.update(
            {
                "decoder_pixel_values": inputs.pixel_values,
            }
        )
        return data_to_process

    def PostProcessVisionInputFromEmbeddings(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing: stack encoded image features
        """
        assert "image_features" in data_to_process.keys()
        images = data_to_process.pop("image_features")

        image_features = [img_dict["image_features"] for img_dict in images]
        stacked_image_features = np.stack(image_features)
        ts_stacked_image_features = torch.FloatTensor(stacked_image_features)

        data_to_process.update(
            {
                "image_features": ts_stacked_image_features,
            }
        )
        return data_to_process

    def PostProcessItemVisionInputFromEmbeddings(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing: stack encoded image features
        """
        assert "img" in data_to_process.keys()
        images = data_to_process.pop("img")

        image_features = [img_dict["image_features"] for img_dict in images]
        stacked_image_features = np.stack(image_features)
        ts_stacked_image_features = torch.FloatTensor(stacked_image_features)
        # print("ts_stacked_image_features.shape", ts_stacked_image_features.shape)
        data_to_process.update(
            {
                "item_image_features": ts_stacked_image_features,
            }
        )
        return data_to_process

    def post_processing(
        self,
        processed_batch_data: EasyDict,
        postprocess_modules: Optional[EasyDict] = None,
    ) -> EasyDict:
        """
        Post-processing the processed data of the whole batch
        Called by colln_func after processing each sample
        """
        postprocessed_batch_data = processed_batch_data
        if postprocess_modules is None:
            # Do nothing and return
            return postprocessed_batch_data
        else:
            # Run provided post-processing unit
            for postprocess_module in postprocess_modules:
                process_func = getattr(self, postprocess_module.type)
                postprocessed_batch_data = process_func(postprocessed_batch_data)

        return postprocessed_batch_data
