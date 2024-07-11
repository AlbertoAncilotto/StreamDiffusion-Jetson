import onnxruntime as ort

def list_available_providers():
    providers = ort.get_available_providers()
    return providers

if __name__ == "__main__":
    providers = list_available_providers()
    print("Available ONNX Runtime Providers:")
    for provider in providers:
        print(provider)
