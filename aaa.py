from collections import defaultdict

from transformers import GPT2LMHeadModel


def count_parameters(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layer_type(name):
    """Extract the layer type from parameter name"""
    if "wte" in name:
        return "Word Embeddings"
    elif "wpe" in name:
        return "Position Embeddings"
    elif "attn" in name:
        return "Attention"
    elif "mlp" in name:
        return "Feed Forward"
    elif "ln" in name:
        return "Layer Normalization"
    return "Other"


def main():
    # Download and load the GPT-2 model
    print("Downloading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Count parameters
    total_params = count_parameters(model)

    # Print model information
    print("\nModel Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size in MB: {total_params * 4 / (1024 * 1024):.2f}")

    # Group parameters by layer type
    layer_params = defaultdict(int)
    print("\nParameter Summary by Layer Type:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_type = get_layer_type(name)
            layer_params[layer_type] += param.numel()

    # Print summary
    print("\nParameter Distribution:")
    for layer_type, params in layer_params.items():
        percentage = (params / total_params) * 100
        print(f"{layer_type}: {params:,} parameters ({percentage:.2f}%)")

    # Print detailed layer-wise parameter count
    print("\nDetailed Layer-wise Parameter Count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} parameters")


main()
