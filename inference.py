import os
import numpy as np
import functools
import torch

from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image

from paras import parse_args
from accelerate import PartialState
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict


def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list=[]
    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt_batch,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def compute_embeddings(
    prompt_batch, original_sizes, crop_coords, proportion_empty_prompts, text_encoders, tokenizers, is_train=True
):
    target_size = (resolution, resolution)
    original_sizes = torch.tensor(original_sizes, dtype=torch.long)
    crops_coords_top_left = torch.tensor(crop_coords, dtype=torch.long)

    prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train )
    add_text_embeds = pooled_prompt_embeds

    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)    
    add_time_ids = torch.cat([original_sizes, crops_coords_top_left, add_time_ids], dim=-1)
    add_time_ids = add_time_ids.to(device, dtype=prompt_embeds.dtype)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision, use_auth_token=True
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def append_dims(x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError( f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]
    
def denoisecm(model, x_t, t,prompt_embeds, encoded_text, sample=False):
    dims = x_t.ndim
    device=x_t.device
    weight_dtype=x_t.dtype
    alpha_t=append_dims(torch.sqrt( alphas_cumprod[t]),dims).to(device)
    sigma_t=append_dims(torch.sqrt(1- alphas_cumprod[t]),dims).to(device)
    model_output = model(x_t, t, encoder_hidden_states=prompt_embeds, added_cond_kwargs={k: v.to(weight_dtype) for k, v in encoded_text.items()}).sample
    denoised = (x_t - sigma_t * model_output) / alpha_t
    return denoised


if __name__ == "__main__":
    args = parse_args()
    distributed_state = PartialState()
    device = distributed_state.device
    batch_size = args.batch_size
    resolution=args.resolution
    output_dir=args.output_dir
    steps_list=[args.infer_steps]
    prompts= [args.prompt]*batch_size
    

    model_dir = args.base_model_path
    noise_scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler")
    tokenizer_one = AutoTokenizer.from_pretrained(
        model_dir, subfolder="tokenizer",  revision=None, use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        model_dir, subfolder="tokenizer_2", revision=None, use_fast=False
    )
    # text encoder
    text_encoder_cls_one = import_model_class_from_model_name_or_path(model_dir, revision=None)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(model_dir, revision=None, subfolder="text_encoder_2")
    text_encoder_one = text_encoder_cls_one.from_pretrained(model_dir, subfolder="text_encoder")
    text_encoder_two = text_encoder_cls_two.from_pretrained(model_dir, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(model_dir,subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet")
   
    lora_config = LoraConfig(
        r=64,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ],
    )
    
    # train lora
    unet = get_peft_model(unet, lora_config)
    lora_path=args.lora_path
    unet.load_adapter(lora_path, adapter_name="default")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    # unet.enable_xformers_memory_efficient_attention()
    
    weight_dtype = torch.bfloat16
    unet.to(device)
    vae.to(device)
    text_encoder_one.to(device)
    text_encoder_two.to(device)
    unet.to(weight_dtype)
    vae.to(weight_dtype)
    text_encoder_one.to(weight_dtype)
    text_encoder_two.to(weight_dtype)
    vae.eval()
    unet.eval()


    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
    )

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device).to(weight_dtype)
    stepsn = noise_scheduler.config.num_train_timesteps
    eval_step=9
    ts = torch.linspace(1000-1, 0, eval_step, device=device,dtype=int)

    for cur_steps in steps_list:
        tmppath = os.path.join(output_dir, str(cur_steps))
        os.makedirs(tmppath,exist_ok=True)
        if cur_steps==1:
            ts=ts[[0,8]]
        if cur_steps==2:
            ts=ts[[0,4,8]]
        if cur_steps==3:
            ts=ts[[0,4,7,8]]
        if cur_steps==4:
            ts=ts[[0,2,4,7,8]]
        if cur_steps==5:
            ts=ts[[0,2,4,6,7,8]]
        if cur_steps==6:
            ts=ts[[0,2,4,5,6,7,8]]
        if cur_steps==7:
            ts=ts[[0,2,3,4,5,6,7,8]]

       
        prompt = prompts
        
        bsz = len(prompt)
        noisy_latents = torch.randn(batch_size,4,int(resolution/8),int(resolution/8)).to(device).to(weight_dtype)  
        orig_size = [(resolution, resolution)]*len(prompt)
        crop_coords = [(0,0)]*len(prompt)
        encoded_text = compute_embeddings_fn(prompt, orig_size, crop_coords)
        encoder_hidden_states = encoded_text.pop("prompt_embeds")
        sample=noisy_latents.clone()
        for j in range(0,len(ts)-1):
            x= denoisecm(unet, sample, ts[[j]].repeat(bsz),encoder_hidden_states, encoded_text)
            sample = noise_scheduler.add_noise(x, torch.randn_like(x), torch.tensor([ts[j+1]], device=ts.device) )
            sample=sample.to(weight_dtype)
        img = vae.decode(x/vae.config.scaling_factor, return_dict=False)[0]
        if img.shape[0]==1:
            img = img.permute(0,2,3,1).detach()
        else:
            img = img.squeeze(0).permute(0,2,3,1).detach()
        img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        for i0 in range(bsz):
            name=prompt[i0]
            im = Image.fromarray(img[i0])
            imgpath = os.path.join(tmppath, name+f'_{i0}.jpg')
            im.save(imgpath)
 
        
    