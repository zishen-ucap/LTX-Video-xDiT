from typing import Optional, Dict, Any, Union, List, Optional, Tuple, Type
import torch
import torch.distributed
import torch.nn as nn

from diffusers.models import LTXVideoTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, USE_PEFT_BACKEND, unscale_lora_layers

from xfuser.model_executor.models import xFuserModelBaseWrapper
from xfuser.logger import init_logger
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.core.distributed import (
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_pipeline_parallel_rank,
    get_pp_group,
    get_world_group,
    get_cfg_group,
    get_sp_group,
    get_runtime_state, 
    initialize_runtime_state
)

from xfuser.model_executor.models.transformers.register import xFuserTransformerWrappersRegister
from xfuser.model_executor.models.transformers.base_transformer import xFuserTransformerBaseWrapper

logger = init_logger(__name__)


@xFuserTransformerWrappersRegister.register(LTXVideoTransformer3DModel)
class xFuserLTXVideoTransformer3DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: LTXVideoTransformer3DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[nn.Conv2d],
            submodule_name_to_wrap=["attn1", "attn2"]
        )
    
    @xFuserBaseWrapper.forward_check_condition
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)
        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                )

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
