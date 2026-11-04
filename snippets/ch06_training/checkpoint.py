def save_checkpoint(path, model, model_config, experiment, step, validation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "model_config": model_config, "vocabulary": list(VOCAB),
                "experiment": experiment, "step": step,
                "validation": validation}, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["vocabulary"] != list(VOCAB):
        raise ValueError("checkpoint vocabulary does not match this example")
    model = Transformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), checkpoint
