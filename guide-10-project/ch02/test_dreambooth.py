import sys

import torch
from diffusers import StableDiffusionPipeline

sys.path.append("../../")
from wolf import get_my_gpu_device

# 이걸로 보내니 RTX 3090 장치는 1이 맞네...
device = get_my_gpu_device(1)
print("사용할 장치는", device)

model_id = "../../data/dreambooth/model/"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    device
)
prompt = "a photo of cute girl riding horse"
# 100번도 이상한데 500번은 그럴듯해진다...
image = pipe(prompt, num_inference_steps=500, guidance_scale=7.5).images[0]
image.save("../../data/photo/girl_ridding_horse.png")
