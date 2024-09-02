from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import torch
import time
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

app = Flask(__name__)

PROMPT = "Watercolor art, in the style of"
WIDTH, HEIGHT = 384, 384
TORCH_DEVICE = 'cuda'
TORCH_DTYPE = torch.float16

# Load model and pipeline
model_loc = "SimianLuo/LCM_Dreamshaper_v7"
pipe = StableDiffusionPipeline.from_pretrained(model_loc).to(
    device=torch.device(TORCH_DEVICE),
    dtype=TORCH_DTYPE,
    safety_checker = True,
)

stream = StreamDiffusion(
    pipe,
    width=WIDTH,
    height=HEIGHT,
    t_index_list=[29, 39],
    torch_dtype=TORCH_DTYPE,
    do_add_noise=True,
)

stream.load_lcm_lora()
stream.fuse_lora()
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

stream.prepare(prompt=PROMPT, num_inference_steps=50, guidance_scale=0)

stream = accelerate_with_tensorrt(
    stream, "engines", max_batch_size=2, engine_build_options={"engine_build_options": {"opt_image_height": HEIGHT, "opt_image_width": WIDTH}}
)

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image'].read()
    image = Image.open(io.BytesIO(image_file)).convert("RGB")
    
    # Perform inference
    start = time.time()
    x_output = stream(image)
    rendered_image = postprocess_image(x_output, output_type="pil")[0]
    print('diff time:', time.time()-start)
    # Convert image to bytes and return as response
    img_byte_arr = io.BytesIO()
    rendered_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    return jsonify({"image": img_byte_arr.hex()})

@app.route('/set_prompt', methods=['POST'])
def set_prompt():
    data = request.json
    if 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400
    
    new_prompt = data['prompt']
    try:
        stream.update_prompt(new_prompt)
        return jsonify({"message": "Prompt updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
