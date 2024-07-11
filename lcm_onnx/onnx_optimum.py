from optimum.onnxruntime import ORTLatentConsistencyModelPipeline
import numpy as np
import cv2

# LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so 

pipe = ORTLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", provider = 'CUDAExecutionProvider', cache_dir='models/')
prompt = "sailing ship in storm by Leonardo da Vinci"
images = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=8.0, output_type=np.array)
img = np.squeeze(images[0])
cv2.imwrite('out.jpg', img[:,:,::-1]*255)

breakpoint()