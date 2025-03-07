---
layout: post
title:  "Notes: playing with generative AI + Diffusion models"
date:   2025-03-06 08:49:21 -0800
categories: AI Notes
---

The notes in this post refer to code in this [gitHub repo](https://github.com/eanderson007/genAI).

## Getting Started

- install the Python requirements from the requirements.txt, noting that the torch version should be modified according to your system. Following the prompts from the torch website to get the installation command for your chosen environment management: [Install Torch](https://pytorch.org/get-started/locally/) 

- clone the diffusers repo locally
```
git clone https://github.com/huggingface/diffusers.git
```

- setup an accelerate configuration locally 
```
accelerate config
```

- setup your [huggingface](https://huggingface.co/docs/hub/en/security-tokens) login token. Login to the site, go to settings and setup a user access token. Copy the code for use with the following
```
huggingface-cli login
```

## Basic Data Generation

Create a Jupyter notebook. Import the necessary libraries 
```
import diffusers
import huggingface_hub
import transformers
import torch

diffusers.logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()

device = 'cuda' # or cpu etc as appropriate 
```

### Images

Load a diffusion model
```
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
).to('cuda')
```
Then prompt the model with text to generate an image
```
prompt = "game sprite viking"
pipe(prompt).images[0]
```

### Generating Text

Load a pipeline for text classification 
```
from transformers import pipeline
from transformers import set_seed

classifier = pipeline("text-classification", device='cuda')
classifier("This movie is disgustingly good !")

# Setting the seed -> get the same results every time run this code
set_seed(10)
```
Pass in text to generate more extensive text to follow
```
generator = pipeline("text-generation")
prompt = "It was a dark and stormy"
generator(prompt)[0]["generated_text"]
```

### Generating Audio

Use text to prompt an audio model to create sound output
```
pipe = pipeline("text-to-audio", model="facebook/musicgen-small", device='cuda')
data = pipe("celtic bapgpipes videogame music")
```

Listen to the audio file the model generated
```
import IPython.display as ipd

display(ipd.Audio(data["audio"][0], rate=data["sampling_rate"]))
```

Save a generated audio file 
```
import torchaudio
path = 'save_music_01.mp3'
audio = torch.from_numpy(data["audio"][0])

torchaudio.save(path, audio, data["sampling_rate"], format='mp3')
```


## Fine Tune a Diffusion Model

Fine tune a diffusion model using a dataset saved in huggingface, or your own local data
- Navigate to the local folder in the diffusions repo cloned earlier: diffusers/examples/text_to_image/

```
# replace "reach-vb/pokemon-blip-captions" with your own dataset
# another test data set: Supermaxman/esa-hubble
# replce "CompVis/stable-diffusion-v1-4" with desired model 
# set "sd-pokemon-model" to the desired output model

accelerate launch train_text_to_image.py 
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" 
  --dataset_name="reach-vb/pokemon-blip-captions" 
  --use_ema 
  --resolution=512 --center_crop --random_flip 
  --train_batch_size=1 
  --gradient_accumulation_steps=4 
  --gradient_checkpointing 
  --mixed_precision="fp16" 
  --max_train_steps=15000 
  --learning_rate=1e-05 
  --max_grad_norm=1 
  --lr_scheduler="constant" --lr_warmup_steps=0 
  --output_dir="sd-pokemon-model" 
  ```

Execute the model fine tuned that is saved locally
  ```
# testing the model
from diffusers import StableDiffusionPipeline
import torch

model_path = "sd-pokemon-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="vine moose").images[0]
  ```