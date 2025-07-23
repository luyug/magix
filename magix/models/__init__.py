from .llama_model import FlaxLlamaModel, FlaxLlamaForCausalLM
from .llama_model_scan import FlaxLlamaForCausalLM as FlaxLlamaForCausalLMScan
from .mistral_model import FlaxMistralModel, FlaxMistralForCausalLM
from .mistral_model_scan import FlaxMistralForCausalLM as FlaxMistralForCausalLMScan
from .bert_model import FlaxBertModel
from .t5_model import FlaxT5EncoderModel
from .gemma_model import FlaxGemmaModel, FlaxGemmaForCausalLM
from .qwen2_model import FlaxQwen2ForCausalLM
from .qwen2_model_scan import FlaxQwen2ForCausalLM as FlaxQwen2ForCausalLMScan
from .qwen3_model import FlaxQwen3ForCausalLM
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
    "qwen2": FlaxQwen2ForCausalLM,
    "qwen3": FlaxQwen3ForCausalLM,
    "llama_scan": FlaxLlamaForCausalLMScan,
    "mistral_scan": FlaxMistralForCausalLMScan,
    "qwen2_scan": FlaxQwen2ForCausalLMScan,
}