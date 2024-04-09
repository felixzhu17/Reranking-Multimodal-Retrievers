local base = import 'RAG_colbert_with_vision.jsonnet';

local override = {
    model_config: {
        "num_beams": 2,
        "RAVQA_loss_type": "Approach6",
        "loss_ratio": {
            "nll_loss": 1,
            "rag_loss": 0,
            "additional_loss": 1,
        },
        "modules": [
            "force_existence",
        ],
    },
    executor: {
        init_kwargs: {
            "freeze_question_encoder_step": null,
        },
    },
};

std.mergePatch(base, override)