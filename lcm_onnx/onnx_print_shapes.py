import onnx

def print_model_shapes(onnx_model_path):
    model = onnx.load(onnx_model_path)
    graph = model.graph

    print("Model inputs:")
    for input_tensor in graph.input:
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"Input: {input_tensor.name}, Shape: {input_shape}")

    print("\nModel outputs:")
    for output_tensor in graph.output:
        output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"Output: {output_tensor.name}, Shape: {output_shape}")

    print("\nModel value info:")
    for value_info in graph.value_info:
        shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        print(f"Tensor: {value_info.name}, Shape: {shape}")

onnx_model_path = "unet.engine.onnx"
print_model_shapes(onnx_model_path)
