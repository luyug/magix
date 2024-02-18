from .llama_model import FlaxLlamaModel, FlaxLlamaForCausalLM
from .mistral_model import FlaxMistralModel, FlaxMistralForCausalLM

ENCODER_MODEL_MAPPING = {
    "llama": FlaxLlamaModel,
    "mistral": FlaxMistralModel,
}

CAUSAL_LM_MODEL_MAPPING = {
    "llama": FlaxLlamaForCausalLM,
    "mistral": FlaxMistralForCausalLM,
}