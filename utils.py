import os


def parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return [float(item) for item in items]


def resolve_checkpoint_paths(lora_path):
    if os.path.isdir(lora_path):
        unet_weights_path = os.path.join(lora_path, "dual_lora_weights.pt")
        config_path = os.path.join(lora_path, "tlora_config.pt")
        text_encoder_weights_path = os.path.join(lora_path, "text_encoder_dual_lora_weights.pt")
        text_encoder_2_weights_path = os.path.join(lora_path, "text_encoder_2_dual_lora_weights.pt")
        return (
            unet_weights_path,
            config_path,
            text_encoder_weights_path,
            text_encoder_2_weights_path,
        )

    checkpoint_dir = os.path.dirname(lora_path)
    return (
        lora_path,
        os.path.join(checkpoint_dir, "tlora_config.pt"),
        os.path.join(checkpoint_dir, "text_encoder_dual_lora_weights.pt"),
        os.path.join(checkpoint_dir, "text_encoder_2_dual_lora_weights.pt"),
    )
