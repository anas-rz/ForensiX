"""Shared OpenAI CLIP ViT-L/14 image encoder for CLIP-based detectors.

Weights: Hugging Face ``openai/clip-vit-large-patch14`` (converted from OpenAI's
release, MIT). OpenAI's model card lists deployed use as out of scope; that is
an intended-use statement, not a license term, but users should be aware.
"""

from safetensors import safe_open

from forensix.hub import WeightFile, fetch

CLIP_VIT_L14 = WeightFile(
    url="https://huggingface.co/openai/clip-vit-large-patch14/resolve/"
    "32bd64288804d66eefd0ccbe215aa642df71cc41/model.safetensors",
    sha256="a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
    filename="clip-vit-large-patch14.safetensors",
)

# OpenAI CLIP preprocessing statistics
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_vit_l14(state: dict | None = None):
    """``CLIPVisionModelWithProjection`` with OpenAI ViT-L/14 weights; its
    ``image_embeds`` equal OpenAI CLIP's ``encode_image``.

    Pass ``state`` (``vision_model.*`` and optionally ``visual_projection.*``
    keys) to load a fine-tuned copy of the same architecture instead.
    """
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    config = CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
        projection_dim=768,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
    )
    model = CLIPVisionModelWithProjection(config)
    if state is not None:
        if "visual_projection.weight" in state:
            model.load_state_dict(state)
        else:  # vision tower only (callers that don't use the projection)
            model.vision_model.load_state_dict(
                {k.removeprefix("vision_model."): v for k, v in state.items()}
            )
        return model
    # the checkpoint also holds the text tower; load only the vision side
    with safe_open(fetch(CLIP_VIT_L14), framework="pt") as f:
        state = {
            k: f.get_tensor(k)
            for k in f.keys()
            if k.startswith(("vision_model.", "visual_projection."))
            # a buffer in older transformers versions, not a weight
            and not k.endswith("position_ids")
        }
    model.load_state_dict(state)
    return model
