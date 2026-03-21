import torch
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
import os
import sys
from model import GPTConfig, GPT
import argparse

def convert_custom_gpt_to_huggingface(
    input_pt_path: str,
    output_dir: str,
    tokenizer_name: str = "gpt2",
    model_config_overrides: dict = None
):
    """
    Converts a custom GPT model checkpoint (.pt file) to Hugging Face Transformers format.

    Args:
        input_pt_path (str): Path to your custom GPT model's .pt or .pth checkpoint file.
        output_dir (str): Directory where the Hugging Face model and tokenizer will be saved.
        tokenizer_name (str): Name of the Hugging Face tokenizer to use (e.g., "gpt2").
                              Ensure its vocabulary size matches your model's vocab_size.
        model_config_overrides (dict, optional): Dictionary to override default GPTConfig
                                                  parameters if your .pt file doesn't contain
                                                  them or they are different.
                                                  Example: {'n_layer': 12, 'n_head': 12, 'n_embd': 768}
    """
    print(f"Starting conversion of {input_pt_path} to Hugging Face format...")

    # 1. 加载自定义GPT模型的配置和状态字典
    print("Loading custom GPT model checkpoint...")
    checkpoint = torch.load(input_pt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        custom_gpt_state_dict = checkpoint['model']
        if 'model_args' in checkpoint:
            custom_gpt_config_args = checkpoint['model_args']
        else:
            print("Warning: 'model_args' not found in checkpoint. Using default GPTConfig.")
            custom_gpt_config_args = {}
    else:
        custom_gpt_state_dict = checkpoint
        print("Warning: Checkpoint is not a dictionary with 'model' key. Assuming it's a raw state_dict.")
        custom_gpt_config_args = {}

    # 移除可能由DDP或torch.compile添加的前缀
    unwanted_prefix = '_orig_mod.'
    custom_gpt_state_dict_cleaned = {}
    for k, v in custom_gpt_state_dict.items():
        if k.startswith(unwanted_prefix):
            custom_gpt_state_dict_cleaned[k[len(unwanted_prefix):]] = v
        else:
            custom_gpt_state_dict_cleaned[k] = v
    custom_gpt_state_dict = custom_gpt_state_dict_cleaned

    # 2. 初始化Hugging Face分词器以获取vocab_size
    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 3. 根据自定义模型的配置创建Hugging Face GPT2Config
    default_gpt_config = GPTConfig()

    # 确定HF模型的vocab_size，优先使用自定义模型的vocab_size
    model_vocab_size = custom_gpt_config_args.get('vocab_size', default_gpt_config.vocab_size)
    if model_vocab_size != tokenizer.vocab_size:
        print(f"Warning: Custom GPT model vocab_size ({model_vocab_size}) "
              f"does not match tokenizer vocab_size ({tokenizer.vocab_size}). "
              f"HF model will be initialized with vocab_size {model_vocab_size}. "
              f"You might need to adjust your tokenizer or handle new tokens if this is unexpected.")

    hf_config_params = {
        "vocab_size": model_vocab_size, # 使用模型的vocab_size来配置HF模型
        "n_positions": custom_gpt_config_args.get('block_size', default_gpt_config.block_size),
        "n_embd": custom_gpt_config_args.get('n_embd', default_gpt_config.n_embd),
        "n_layer": custom_gpt_config_args.get('n_layer', default_gpt_config.n_layer),
        "n_head": custom_gpt_config_args.get('n_head', default_gpt_config.n_head),
        "resid_pdrop": custom_gpt_config_args.get('dropout', default_gpt_config.dropout),
        "embd_pdrop": custom_gpt_config_args.get('dropout', default_gpt_config.dropout),
        "attn_pdrop": custom_gpt_config_args.get('dropout', default_gpt_config.dropout),
        "activation_function": "gelu_new", # GPT-2 uses "gelu_new"
        "layer_norm_epsilon": 1e-5, # GPT-2 default
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        # GPT2LMHeadModel doesn't have a global 'use_bias' parameter for all layers.
        # Biases are typically enabled by default in its Linear layers.
        # We will handle missing biases during state_dict loading.
    }

    # 应用用户提供的配置覆盖
    if model_config_overrides:
        print(f"Applying model config overrides: {model_config_overrides}")
        hf_config_params.update(model_config_overrides)

    hf_config = GPT2Config(**hf_config_params)
    print("Hugging Face GPT2Config created:")
    print(hf_config)

    # 4. 初始化Hugging Face GPT2LMHeadModel
    print("Initializing Hugging Face GPT2LMHeadModel...")
    hf_model = GPT2LMHeadModel(hf_config)

    # 如果HF模型的vocab_size与我们期望的不同，则调整其嵌入层大小
    if hf_model.config.vocab_size != model_vocab_size:
        print(f"Resizing HF model token embeddings from {hf_model.config.vocab_size} to {model_vocab_size}.")
        hf_model.resize_token_embeddings(model_vocab_size)
    
    # 5. 将自定义模型的权重映射并加载到Hugging Face模型中
    print("Mapping and loading weights from custom GPT to Hugging Face model...")
    
    # 定义需要转置的权重键
    # 这些是你的model.py中GPT.from_pretrained方法里提到的需要转置的键
    transposed_keys_template = [
        'transformer.h.{}.attn.c_attn.weight',
        'transformer.h.{}.attn.c_proj.weight',
        'transformer.h.{}.mlp.c_fc.weight',
        'transformer.h.{}.mlp.c_proj.weight',
    ]
    
    # 创建一个用于存储最终要加载到HF模型的state_dict
    hf_loadable_state_dict = {}

    # 遍历自定义模型的state_dict，处理转置和复制
    for k_custom, v_custom in custom_gpt_state_dict.items():
        k_hf = k_custom # 默认情况下键名相同

        # 检查是否是需要转置的键
        is_transposed = False
        for i in range(hf_config.n_layer):
            for template in transposed_keys_template:
                if k_custom == template.format(i):
                    is_transposed = True
                    break
            if is_transposed:
                break
        
        # 特殊处理 lm_head.weight，如果它与 wte.weight 共享
        # 你的model.py中是共享的，所以这里应该直接复制 wte.weight
        # lm_head.weight 在 HF 模型中是单独的，但在你的模型中是绑定的
        # 如果 lm_head.weight 在 custom_gpt_state_dict 中，且它就是 wte.weight
        # 那么我们直接复制 wte.weight，lm_head.weight 会自动绑定
        # 如果 lm_head.weight 报错，说明它不是绑定的，或者需要单独处理
        if k_custom == 'lm_head.weight' and 'transformer.wte.weight' in custom_gpt_state_dict and \
           torch.equal(v_custom, custom_gpt_state_dict['transformer.wte.weight']):
            # 如果lm_head.weight与wte.weight相同，则跳过，因为HF模型会处理权重绑定
            # 除非HF模型不绑定，那还是需要复制
            # 鉴于错误信息，我们还是尝试复制它
            pass # 继续让下面的逻辑处理

        if is_transposed:
            # 执行转置
            if v_custom.dim() == 2 and hf_model.state_dict()[k_hf].shape == v_custom.T.shape:
                hf_loadable_state_dict[k_hf] = v_custom.T
                print(f"Copied and transposed {k_custom} (shape {v_custom.shape}) to {k_hf} (expected {v_custom.T.shape}).")
            else:
                print(f"Warning: Transpose expected for {k_custom}, but shapes {v_custom.shape} and {hf_model.state_dict()[k_hf].shape} don't match for transpose. Skipping.")
        else:
            # 直接复制
            if k_hf in hf_model.state_dict() and hf_model.state_dict()[k_hf].shape == v_custom.shape:
                hf_loadable_state_dict[k_hf] = v_custom
                # print(f"Copied {k_custom} (shape {v_custom.shape}) to {k_hf}.")
            else:
                print(f"Warning: Shape mismatch or key not found for {k_custom} (custom: {v_custom.shape}, HF expected: {hf_model.state_dict()[k_hf].shape if k_hf in hf_model.state_dict() else 'N/A'}). Skipping direct copy.")

    # 尝试加载处理后的state_dict
    # 使用 strict=False 来允许缺失的偏置项（如果自定义模型没有偏置）
    print("Attempting to load state_dict with strict=False to handle mismatches and missing biases.")
    load_result = hf_model.load_state_dict(hf_loadable_state_dict, strict=False)
    
    if load_result.missing_keys:
        # 过滤掉由于 custom_gpt_config_args['bias'] = False 导致的缺失偏置项
        # 假设所有以 '.bias' 结尾的，且不在 custom_gpt_state_dict 中的键都是可以接受的缺失
        expected_missing_bias_keys = []
        if not custom_gpt_config_args.get('bias', default_gpt_config.bias):
            for missing_key in load_result.missing_keys:
                if missing_key.endswith('.bias'):
                    expected_missing_bias_keys.append(missing_key)
        
        actual_missing_keys = [k for k in load_result.missing_keys if k not in expected_missing_bias_keys]

        if actual_missing_keys:
            print(f"Warning: Missing key(s) in Hugging Face model state_dict (after filtering biases): {actual_missing_keys}")
        if expected_missing_bias_keys:
            print(f"Info: Expected missing bias key(s) (custom model trained without bias): {expected_missing_bias_keys}")

    if load_result.unexpected_keys:
        print(f"Warning: Unexpected key(s) in custom model state_dict: {load_result.unexpected_keys}")
    

    # 6. 保存Hugging Face模型和分词器
    print(f"Saving Hugging Face model and tokenizer to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Conversion complete!")
    print(f"You can now load your model using: AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"And tokenizer using: AutoTokenizer.from_pretrained('{output_dir}')")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert custom GPT checkpoint to Hugging Face format.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the custom GPT .pt checkpoint file.")
    parser.add_argument("--iteration_number", type=int, required=True, help="Iteration number of the checkpoint.")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory to save the Hugging Face model and tokenizer.")
    args = parser.parse_args()

    checkpoints_path = args.checkpoint_path
    iteration_number = args.iteration_number
    output_directory = args.output_directory
    
    input_pt_path = os.path.join(checkpoints_path, f"iter{iteration_number}_ckpt.pt")
    tokenizer_name = os.path.join(checkpoints_path, f"changer_iter{iteration_number}/tokenizer")
    example_model_config_overrides = {
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'block_size': 1024, # 确保与你模型训练时的block_size一致
        'dropout': 0.0,
        'bias': False, # 根据你的模型实际是否使用bias
        # 'vocab_size': 50304, # 如果你的模型训练时使用了这个vocab_size，但tokenizer是gpt2 (50257)
                                # 那么这里可能需要调整，或者确保tokenizer匹配
    }

    convert_custom_gpt_to_huggingface(
        input_pt_path=input_pt_path,
        output_dir=output_directory,
        tokenizer_name=tokenizer_name,
        model_config_overrides=example_model_config_overrides # 根据需要调整或移除
    )