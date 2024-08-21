import cv2
import numpy as np
from PIL import Image

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
import time
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

PROMPT        = "Watercolor art, in the style of" 
WIDTH, HEIGHT = 320,320
TORCH_DEVICE  = 'cuda' 
TORCH_DTYPE   = torch.float16

def get_result_and_mask(frame, center_x, center_y, width, height):
    mask = np.zeros_like(frame)
    mask[center_y:center_y+height, center_x:center_x+width, :] = 255
    cutout = frame[center_y:center_y+height, center_x:center_x+width, :]
    return frame, cutout

model_loc = "SimianLuo/LCM_Dreamshaper_v7"
pipe = StableDiffusionPipeline.from_pretrained(model_loc).to(
    device=torch.device(TORCH_DEVICE),
    dtype=TORCH_DTYPE,
)

stream = StreamDiffusion(
    pipe,
    width = WIDTH,
    height= HEIGHT,
    t_index_list=[28, 35],
    torch_dtype=TORCH_DTYPE,
    do_add_noise=False,
)

stream.load_lcm_lora()
stream.fuse_lora()
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

# pipe.enable_xformers_memory_efficient_attention()

stream.prepare(prompt = PROMPT, 
        num_inference_steps=50,
        guidance_scale=0)

# if this fails, run trt.sh manually
stream = accelerate_with_tensorrt(
    stream, "engines", max_batch_size=2, engine_build_options={"engine_build_options":{"opt_image_height":HEIGHT, "opt_image_width":WIDTH}}
)

# stream.enable_similar_image_filter()

# cap = cv2.VideoCapture(0)
# Run the stream infinitely
for _ in range(100):
    ret, frame = True, np.random.randint(255, size=(768,512,3),dtype=np.uint8)
    # ret, frame = cap.read()
    frame = cv2.resize(frame, (768, 512))
    if not ret:
        print("Error: Failed to capture frame.")
        break

    center_x = (frame.shape[1] - WIDTH) // 2
    center_y = (frame.shape[0] - HEIGHT) // 2
    start = time.time()

    result_image, result_cutout = get_result_and_mask(frame, center_x, center_y, WIDTH, HEIGHT)
    result_cutout = Image.fromarray(cv2.cvtColor(result_cutout, cv2.COLOR_BGR2RGB)) 

    x_output = stream(result_cutout)
    print('frame time:', time.time() - start)
    rendered_image = postprocess_image(x_output, output_type="pil")[0]#.show()

    result_image[center_y:center_y+HEIGHT, center_x:center_x+WIDTH] = cv2.cvtColor(np.array(rendered_image), cv2.COLOR_RGB2BGR)

    # Display output
    # cv2.imshow("output", result_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
