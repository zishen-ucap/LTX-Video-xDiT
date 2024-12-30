import os
from typing import Any, List, Tuple, Callable, Optional, Union, Dict

import torch
import torch.distributed

from diffusers import LTXPipeline
from diffusers.pipelines.ltx.pipeline_ltx import (
    LTXPipelineOutput,
    retrieve_timesteps,
    calculate_shift,
)
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback

import numpy as np

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



@xFuserPipelineWrapperRegister.register(LTXPipeline)
class xFuserLTXPipeline(xFuserPipelineBaseWrapper):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        **kwargs,
    ):
        pipeline = LTXPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(pipeline, engine_config)

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        frame_rate: int = 25,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 3,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `512`):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, defaults to `704`):
                The width in pixels of the generated image. This is set to 848 by default for the best results.
            num_frames (`int`, defaults to `161`):
                The number of video frames to generate
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, defaults to `3 `):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `None`):
                The interpolation factor between random noise and denoised latents at the decode timestep.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ltx.LTXPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `128 `):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ltx.LTXPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ltx.LTXPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

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
        self._attention_kwargs = attention_kwargs
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
        prompt_embeds = self._process_cfg_split_batch(
            negative_prompt_embeds, prompt_embeds
        )
        prompt_attention_mask = self._process_cfg_split_batch(
            negative_prompt_attention_mask, prompt_attention_mask
        )
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        video_sequence_length = latent_num_frames * latent_height * latent_width
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            video_sequence_length,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare micro-conditions
        latent_frame_rate = frame_rate / self.vae_temporal_compression_ratio
        rope_interpolation_scale = (
            1 / latent_frame_rate,
            self.vae_spatial_compression_ratio,
            self.vae_spatial_compression_ratio,
        )

        # 切割张量
        latent_model_input = torch.cat([latents] * (2 // get_classifier_free_guidance_world_size())) if self.do_classifier_free_guidance else latents
        latent_model_input = latent_model_input.to(prompt_embeds.dtype)
        image_rotary_emb = self.transformer.rope(latent_model_input, latent_num_frames, latent_height, latent_width, rope_interpolation_scale)
        latents, prompt_embeds, prompt_attention_mask, image_rotary_emb = self._init_sync_pipeline(
            latents, prompt_embeds, prompt_attention_mask, image_rotary_emb, 
            latent_num_frames
        )

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * (2 // get_classifier_free_guidance_world_size())) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    image_rotary_emb = image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                if self.do_classifier_free_guidance:
                    if get_classifier_free_guidance_world_size() == 1:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    elif get_classifier_free_guidance_world_size() == 2:
                        noise_pred_uncond, noise_pred_text = get_cfg_group().all_gather(
                            noise_pred, separate_tensors=True
                        )
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()


        if get_sequence_parallel_world_size() > 1:
            latents = get_sp_group().all_gather(latents, dim=-2)

        if is_dp_last_group():
            if not output_type == "latent":
                latents = self._unpack_latents(
                latents,
                latent_num_frames,
                latent_height,
                latent_width,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
                )
                latents = self._denormalize_latents(
                    latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                )
                latents = latents.to(prompt_embeds.dtype)

                if not self.vae.config.timestep_conditioning:
                    timestep = None
                else:
                    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                    if not isinstance(decode_timestep, list):
                        decode_timestep = [decode_timestep] * batch_size
                    if decode_noise_scale is None:
                        decode_noise_scale = decode_timestep
                    elif not isinstance(decode_noise_scale, list):
                        decode_noise_scale = [decode_noise_scale] * batch_size

                    timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                    decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                        :, None, None, None, None
                    ]
                    latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

                video = self.vae.decode(latents, timestep, return_dict=False)[0]
                video = self.video_processor.postprocess_video(video, output_type=output_type)

            else:
                video = latents

        else:
            video = [None for _ in range(batch_size)]
        
        
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return LTXPipelineOutput(frames=video)

    def _init_sync_pipeline(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_frames_latent: int = None,
    ):
        latents = torch.cat(
                    [
                        latents
                        [
                            :, start_token_idx*num_frames_latent:end_token_idx*num_frames_latent
                        ]
                        
                        
                        for start_token_idx, end_token_idx in get_runtime_state().pp_patches_token_start_end_idx_global
                    ],
                    dim=0,
                )
        if get_runtime_state().split_text_embed_in_sp:
            if (prompt_embeds.shape[-2] % get_sequence_parallel_world_size() == 0) and (prompt_attention_mask.shape[-2] % get_sequence_parallel_world_size() == 0):
                sequence_parallel_world_size = get_sequence_parallel_world_size()
                sequence_parallel_rank = get_sequence_parallel_rank()
                prompt_embeds = torch.chunk(prompt_embeds, sequence_parallel_world_size, dim=-2)[sequence_parallel_rank]
                prompt_attention_mask = torch.chunk(prompt_attention_mask, sequence_parallel_world_size, dim=-1)[sequence_parallel_rank]
            else:
                get_runtime_state().split_text_embed_in_sp = False                

        if image_rotary_emb is not None:
            image_rotary_emb = (
                torch.cat(
                    [
                        image_rotary_emb[0]
                        [
                           :,start_token_idx*num_frames_latent:end_token_idx*num_frames_latent
                        ]
                        for start_token_idx, end_token_idx in get_runtime_state().pp_patches_token_start_end_idx_global
                    ],
                    dim=0,
                ),
                torch.cat(
                    [
                        image_rotary_emb[1]
                        [
                           :,start_token_idx*num_frames_latent:end_token_idx*num_frames_latent
                        ]
                        for start_token_idx, end_token_idx in get_runtime_state().pp_patches_token_start_end_idx_global
                    ],
                    dim=0,
                ),
            )
        
        return latents, prompt_embeds, prompt_attention_mask, image_rotary_emb
    
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
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt
