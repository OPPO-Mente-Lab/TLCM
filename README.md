# TLCM: Training-efficient Latent Consistency Model for Image Generation with 2-8 Steps

<p align="center">
   📃 <a href="https://arxiv.org/html/2406.05768v5" target="_blank">Paper</a> • 
   🤗 <a href="https://huggingface.co/AIGCer-OPPO/TLCM" target="_blank">Checkpoints</a> 
</p>

<!-- **TLCM: Training-efficient Latent Consistency Model for Image Generation with 2-8 Steps** -->

we propose an innovative two-stage data-free consistency distillation (TDCD) approach to accelerate latent consistency model. The first stage improves consistency constraint  by data-free sub-segment consistency distillation (DSCD). The second stage enforces the
global consistency across inter-segments through data-free consistency distillation (DCD). Besides, we explore various
 techniques to promote TLCM’s performance in data-free manner, forming Training-efficient Latent Consistency
 Model (TLCM) with 2-8 step inference.

TLCM demonstrates a high level of flexibility by enabling adjustment of sampling steps within the range of 2 to 8 while still producing competitive outputs compared
to full-step approaches.

- [Install Dependency](#install-dependency)
- [Example Use](#example-use)
- [Art Gallery](#art-gallery)
- [Addition](#addition)
- [Citation](#citation)

## Install Dependency

```
pip install diffusers 
pip install transformers accelerate
```

## Example Use

We provide an example inference script in the directory of this repo. 
You should download the Lora path from [here](https://huggingface.co/AIGCer-OPPO/TLCM) and use a base model, such as [SDXL1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) , as the recommended option.
After that, you can activate the generation with the following code:
```
python inference.py --prompt {Your prompt} --output_dir {Your output directory} --lora_path {Lora_directory} --base_model_path {Base_model_directory} --infer-steps 4
```
More parameters are presented in paras.py. You can modify them according to your requirements.


<p style="font-size: 24px; font-weight: bold; color: #FF5733; text-align: center;">
    <span style=" padding: 10px; border-radius: 5px;">
        🚀 Update 🚀
    </span>
</p>


We adapt diffuser pipeline for our pipeline, so now you can now use a simpler version below with the base model SDXL 1.0, and we highly recommend it :
```
import torch,diffusers
from diffusers import LCMScheduler,AutoPipelineForText2Image
from peft import LoraConfig, get_peft_model

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lora_path = 'path/to/the/lora'
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

pipe = AutoPipelineForText2Image.from_pretrained(model_id,torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
unet=pipe.unet
unet = get_peft_model(unet, lora_config)
unet.load_adapter(lora_path, adapter_name="default")
pipe.unet=unet
pipe.to('cuda')

eval_step=4 # the step can be changed within 2-8 steps

prompt = "An astronaut riding a horse in the jungle"
# disable guidance_scale by passing 0
image = pipe(prompt=prompt, num_inference_steps=eval_step, guidance_scale=0).images[0]
```


We also adapt our methods based on [**FLUX**](https://huggingface.co/black-forest-labs/FLUX.1-dev) model. 
You can down load the corresponding LoRA model [here]() and load it with the base model for faster sampling.
The sampling script for faster FLUX sampling as below:
```
import os,torch
from diffusers import FluxPipeline
from scheduling_flow_match_tlcm import FlowMatchEulerTLCMScheduler
from peft import LoraConfig, get_peft_model

model_id = "black-forest-labs/FLUX.1-dev"
lora_path = "path/to/the/lora/folder"
lora_config = LoraConfig(
    r=64,
    target_modules=[
        "to_k", "to_q", "to_v", "to_out.0",
        "proj_in",
        "proj_out",
        "ff.net.0.proj",
        "ff.net.2",
        # new
        "context_embedder", "x_embedder",
        "linear", "linear_1", "linear_2",
        "proj_mlp",
        "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
        "ff_context.net.0.proj", "ff_context.net.2"
        ],
        )

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.scheduler = FlowMatchEulerTLCMScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda:0')
transformer = pipe.transformer
transformer = get_peft_model(transformer, lora_config)
transformer.load_adapter(lora_path, adapter_name="default", is_trainable=False)
pipe.transformer=transformer

eval_step=2 # the step can be changed within 2-8 steps

prompt = "An astronaut riding a horse in the jungle"
# disable guidance_scale by passing 0
image = pipe(prompt=prompt, num_inference_steps=eval_step, guidance_scale=0).images[0]

```
## Art Gallery
Here we present some examples based on **SDXL** with different samping steps.

<div align="center">
    <p>2-Steps Sampling</p>
</div>
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="assets/SDXL/2s/dog.jpg" alt="图片1" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/2s/girl1.jpg" alt="图片2" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/2s/girl2.jpg" alt="图片3" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/2s/rose.jpg" alt="图片4" width="180" style="margin: 10px;" />
</div>

<div align="center">
    <p>3-Steps Sampling</p>
</div>
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="assets/SDXL/3s/batman.jpg" alt="图片1" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/3s/horse.jpg" alt="图片2" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/3s/living room.jpg" alt="图片3" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/3s/woman.jpg" alt="图片4" width="180" style="margin: 10px;" />
</div>

<div align="center">
    <p>4-Steps Sampling</p>
</div>
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="assets/SDXL/4s/boat.jpg" alt="图片1" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/4s/building.jpg" alt="图片2" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/4s/mountain.jpg" alt="图片3" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/4s/wedding.jpg" alt="图片4" width="180" style="margin: 10px;" />
</div>

<div align="center">
    <p>8-Steps Sampling</p>
</div>
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="assets/SDXL/8s/car.jpg" alt="图片1" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/8s/cat.jpg" alt="图片2" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/8s/robot.jpg" alt="图片3" width="180" style="margin: 10px;" />
    <img src="assets/SDXL/8s/woman.jpg" alt="图片4" width="180" style="margin: 10px;" />
</div>

We also present some examples based on **FLUX**.
<div align="center">
    <p>3-Steps Sampling</p>
</div>
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/3s/portrait.jpg" alt="图片1" width="180" />
        <br />
        <span>Seasoned female journalist...</span><br>
        <span>eyes behind glasses...</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/3s/hallway.jpg" alt="图片2" width="180" />
        <br/>
        <span>A grand hallway</span><br>
        <span>inside an opulent palace...</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/3s/starnight.jpg" alt="图片3" width="180" />
        <br />
        <span>Van Gogh’s Starry Night...</span><br>
        <span>replace... with cityscape</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/3s/sailor.jpg" alt="图片4" width="180" />
        <br />
        <span>A weathered sailor...</span><br>
        <span>blue eyes...</span>
    </div>
</div>
<div align="center">
    <p>4-Steps Sampling</p>
</div>
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/4s/guitar.jpg" alt="图片1" width="180" />
        <br />
        <span>A guitar,</span><br>
        <span>2d minimalistic icon...</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/4s/cat.jpg" alt="图片2" width="180" />
        <br/>
        <span>A cat</span><br>
        <span>near the window...</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/4s/rabbit.jpg" alt="图片3" width="180" />
        <br />
        <span>close up photo of a rabbit...</span><br>
        <span>forest in spring...</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/4s/blossom.jpg" alt="图片4" width="180" />
        <br />
        <span>...urban decay...</span><br>
        <span>...a vibrant cherry blossom...</span>
    </div>
</div>
<div align="center">
    <p>6-Steps Sampling</p>
</div>
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/6s/dog.jpg" alt="图片1" width="180" />
        <br />
        <span>A cute dog</span><br>
        <span>on the grass...</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/6s/tea.jpg" alt="图片2" width="180" />
        <br/>
        <span>...hot floral tea</span><br>
        <span>in glass kettle...</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/6s/bag.jpg" alt="图片3" width="180" />
        <br />
        <span>...a bag...</span><br>
        <span>luxury product style...</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/6s/cat.jpg" alt="图片4" width="180" />
        <br />
        <span>a master jedi cat...</span><br>
        <span>wearing a jedi cloak hood</span>
    </div>
</div>
<div align="center">
    <p>8-Steps Sampling</p>
</div>
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/8s/lion.jpg" alt="图片1" width="180" />
        <br />
        <span>A cute dog</span><br>
        <span>on the grass...</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/8s/tea.jpg" alt="图片2" width="180" />
        <br/>
        <span>...hot floral tea</span><br>
        <span>in glass kettle...</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/8s/bag.jpg" alt="图片3" width="180" />
        <br />
        <span>...a bag...</span><br>
        <span>luxury product style...</span>
    </div>
    <div style="text-align: center; margin: 10px;">
        <img src="assets/FLUX/8s/cat.jpg" alt="图片4" width="180" />
        <br />
        <span>a master jedi cat...</span><br>
        <span>wearing a jedi cloak hood</span>
    </div>
</div>
## Addition

We also provide the latent lpips model [here](https://huggingface.co/AIGCer-OPPO/TLCM). 
More details are presented in the paper.

## Citation

```
@article{xietlcm,
  title={TLCM: Training-efficient Latent Consistency Model for Image Generation with 2-8 Steps},
  author={Xie, Qingsong and Liao, Zhenyi and Chen, Chen and Deng, Zhijie and TANG, SHIXIANG and Lu, Haonan}
}
```