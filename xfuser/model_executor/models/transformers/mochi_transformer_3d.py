from typing import Optional, Dict, Any, Union, List, Optional, Tuple, Type
import torch
import torch.distributed
import torch.nn as nn

from diffusers.models.embeddings import PatchEmbed

from diffusers.models import MochiTransformer3DModel
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


@xFuserTransformerWrappersRegister.register(MochiTransformer3DModel)
class xFuserMochiTransformer3DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: MochiTransformer3DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[nn.Conv2d, PatchEmbed],
            submodule_name_to_wrap=["attn1"]
        )
    
    @xFuserBaseWrapper.forward_check_condition
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        return_dict: bool = True,
        image_rotary_emb : Tuple[torch.Tensor, torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        
        p = self.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p
        
        temb, encoder_hidden_states = self.time_embed(
            timestep, encoder_hidden_states, encoder_attention_mask, hidden_dtype=hidden_states.dtype
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)


        for i, block in enumerate(self.transformer_blocks):
    
            if self.gradient_checkpointing:
                
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
        hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
