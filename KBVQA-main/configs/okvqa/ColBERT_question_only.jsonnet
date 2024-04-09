local base = import 'ColBERT.jsonnet';

local override = {
    model_config: {
        "input_modules": {
            "module_list":[
                {"type": "QuestionInput",  "option": "default", 
                        "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},
                // {"type": "TextBasedVisionInput",  "option": "caption",
                //         "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
                // {"type": "TextBasedVisionInput",  "option": "object", 
                //         "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
                //         "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
            ],
            "postprocess_module_list": [
                {"type": "PostProcessColBERTQuestionInputTokenization", "option": "default"},
            ],
        },
        "decoder_input_modules": {
            "module_list":[
                {"type": "KnowledgeInput",  "option": "default",
                        "separation_tokens": {'start': '<BOK>', 'end': '<EOK>'}},
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
};

std.mergePatch(base, override)