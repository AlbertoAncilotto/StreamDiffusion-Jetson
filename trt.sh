WIDTH_RESOLUTION=320  # Example width resolution
HEIGHT_RESOLUTION=320  # Example height resolution
MAX_BATCH_SIZE=2  # Example max batch size (lcm step count)
ONNX_PATH="engines/onnx/unet.opt.onnx"  # Path to the ONNX model
ENGINE_PATH="engines/unet.engine"  # Path where the TensorRT engine will be saved

# Calculate the latent resolutions for width and height
LATENT_WIDTH_RES=$(($WIDTH_RESOLUTION / 8))
LATENT_HEIGHT_RES=$(($HEIGHT_RESOLUTION / 8))

# Execute the trtexec command with the calculated values
trtexec --onnx=$ONNX_PATH --saveEngine=$ENGINE_PATH --fp16 \
--minShapes=sample:1x4x${LATENT_HEIGHT_RES}x${LATENT_WIDTH_RES},timestep:1,encoder_hidden_states:1x77x768 \
--optShapes=sample:${MAX_BATCH_SIZE}x4x${LATENT_HEIGHT_RES}x${LATENT_WIDTH_RES},timestep:${MAX_BATCH_SIZE},encoder_hidden_states:${MAX_BATCH_SIZE}x77x768 \
--maxShapes=sample:${MAX_BATCH_SIZE}x4x${LATENT_HEIGHT_RES}x${LATENT_WIDTH_RES},timestep:${MAX_BATCH_SIZE},encoder_hidden_states:${MAX_BATCH_SIZE}x77x768 \
--workspace=14048 --verbose
