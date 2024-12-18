import os
from typing import Any, List, Tuple, Callable, Optional, Union, Dict
import logging

import torch
import torch.distributed

from diffusers import MochiPipeline
from diffusers.pipelines.mochi.pipeline_mochi import (
    MochiPipelineOutput,
    retrieve_timesteps,
    linear_quadratic_schedule
)

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
import numpy as np

import math

from xfuser.config import EngineConfig

from xfuser.core.distributed import (
    get_pipeline_parallel_world_size,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
    get_sp_group,
    get_runtime_state,
    is_dp_last_group,
)

from xfuser.model_executor.pipelines import xFuserPipelineBaseWrapper
from .register import xFuserPipelineWrapperRegister


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fid_computation.log')
        ]
    )

@xFuserPipelineWrapperRegister.register(MochiPipeline)
class xFuserMochiPipeline(xFuserPipelineBaseWrapper):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        **kwargs,
    ):
        pipeline = MochiPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(pipeline, engine_config)
    
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 16,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
    ):
        setup_logging()

        self.scheduler = FlowMatchEulerDiscreteScheduler(base_image_seq_len=256, base_shift=0.5, invert_sigmas=True, max_image_seq_len=4096, max_shift=1.15, num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.default_height
        width = width or self.default_width

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        get_runtime_state().set_video_input_parameters(
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

        # 3. Prepare text embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        # if self.do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        #     prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        prompt_embeds = self._process_cfg_split_batch(negative_prompt_embeds, prompt_embeds)
        prompt_attention_mask = self._process_cfg_split_batch(negative_prompt_attention_mask, prompt_attention_mask)


        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timestep
        # from https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
        threshold_noise = 0.025
        sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
        sigmas = np.array(sigmas)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        p_t = self.transformer.config.patch_size or 1
        batch_size_latent, num_channels_latent, num_frames_latent, height_latent, width_latent = latents.shape
        post_patch_height = height_latent//p_t
        post_patch_width = width_latent//p_t
        image_rotary_emb = self.transformer.rope(
            self.transformer.pos_frequencies,
            num_frames_latent,
            post_patch_height,
            post_patch_width,
            device=latents.device,
            dtype=torch.float32,
        )
        
        latents, prompt_embeds, prompt_attention_mask, image_rotary_emb = self._init_sync_pipeline(
            latents=latents, 
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            image_rotary_emb=image_rotary_emb,  
            latents_frames=(latents.size(1) + p_t - 1) // p_t,
            num_frames_latent=num_frames_latent,
            post_patch_height=post_patch_height,
            post_patch_width=post_patch_width
        )
       
        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
        
                latent_model_input = torch.cat([latents] * (2// get_classifier_free_guidance_world_size())) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
           
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    return_dict=False,
                    image_rotary_emb=image_rotary_emb
                )[0]
                
                if self.do_classifier_free_guidance:
                    if get_classifier_free_guidance_world_size() == 1:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    elif get_classifier_free_guidance_world_size() == 2:
                        noise_pred_uncond, noise_pred_text = get_cfg_group().all_gather(
                            noise_pred, separate_tensors=True
                        )
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0] 
                
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()
        if get_sequence_parallel_world_size() > 1:

            latents = get_sp_group().all_gather(latents, dim=-2)

        if is_dp_last_group():
            if output_type == "latent":
                video = latents

            else:
                has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
                has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
                if has_latents_mean and has_latents_std:
                    latents_mean = (
                        torch.tensor(self.vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents_std = (
                        torch.tensor(self.vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
                else:
                    latents = latents / self.vae.config.scaling_factor
                
                video = self.vae.decode(latents, return_dict=False)[0]
                video = self.video_processor.postprocess_video(video, output_type=output_type)

        else:
            video = [None for _ in range(batch_size)]
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return MochiPipelineOutput(frames=video)


    def _init_sync_pipeline(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        latents_frames: Optional[int] = None,
        num_frames_latent: int = None,
        post_patch_height: Optional[int] = None,
        post_patch_width: Optional[int] = None,
    ):
        latents = super()._init_video_sync_pipeline(latents)

        if get_runtime_state().split_text_embed_in_sp:
            if prompt_embeds.shape[-2] % get_sequence_parallel_world_size() == 0:
                sequence_parallel_world_size = get_sequence_parallel_world_size()
                sequence_parallel_rank = get_sequence_parallel_rank()
                prompt_embeds = torch.chunk(prompt_embeds, sequence_parallel_world_size, dim=-2)[sequence_parallel_rank]
                prompt_attention_mask = torch.chunk(prompt_attention_mask, sequence_parallel_world_size, dim=-1)[sequence_parallel_rank]
            else:
                get_runtime_state().split_text_embed_in_sp = False                

        # 从宽高维度切割image_rotary_emb
        if image_rotary_emb is not None:

            s, f1, f2 = image_rotary_emb[0].shape

            image_rotary_emb = (
                torch.cat(
                    [
                        image_rotary_emb[0].reshape(num_frames_latent, -1, f1, f2)
                        [
                            :, start_token_idx:end_token_idx
                        ]
                        .reshape(-1, f1, f2)
                        
                        for start_token_idx, end_token_idx in get_runtime_state().pp_patches_token_start_end_idx_global
                    ],
                    dim=0,
                ),
                torch.cat(
                    [
                        image_rotary_emb[1].reshape(num_frames_latent, -1, f1, f2)
                        [
                            :, start_token_idx:end_token_idx
                        ]
                        .reshape(-1, f1, f2)
                        
                        for start_token_idx, end_token_idx in get_runtime_state().pp_patches_token_start_end_idx_global
                    ],
                    dim=0,
                ),
            )
        
        return latents, prompt_embeds, prompt_attention_mask, image_rotary_emb


    @property
    def interrupt(self):
        return self._interrupt

    @property
    def guidance_scale(self):
        return self._guidance_scale
