@torch.no_grad()
def visibility_probe():
    original = torch.tensor([encode("the red car stopped")])
    changed = torch.tensor([encode("the red car moved")])
    torch.manual_seed(71)
    causal = TinyTokenModel(causal=True).double().eval()
    bidirectional = TinyTokenModel(causal=False).double().eval()
    bidirectional.load_state_dict(causal.state_dict())
    differences = {}
    for name, model in (("causal", causal), ("bidirectional", bidirectional)):
        before = model(original)[0, 1]   # vocabulary logits at "red"'s position
        after = model(changed)[0, 1]     # only a later token changed
        differences[name] = (after - before).abs().max().item()
    return differences
