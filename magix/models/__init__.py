from .llama_model import FlaxLlamaModel, FlaxLlamaForCausalLM
from .mistral_model import FlaxMistralModel, FlaxMistralForCausalLM
from .bert_model import FlaxBertModel
from .t5_model import FlaxT5EncoderModel
from .gemma_model import FlaxGemmaModel, FlaxGemmaForCausalLM

ENCODER_MODEL_MAPPING = {
    "llama": FlaxLlamaModel,
    "mistral": FlaxMistralModel,
    "bert": FlaxBertModel,
    "t5": FlaxT5EncoderModel,
    "gemma": FlaxGemmaModel,
}

CAUSAL_LM_MODEL_MAPPING = {
    "llama": FlaxLlamaForCausalLM,
    "mistral": FlaxMistralForCausalLM,
    "gemma": FlaxGemmaForCausalLM,
}