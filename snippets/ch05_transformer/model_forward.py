def forward(self, source_ids, target_ids, source_valid=None,
            target_valid=None, return_weights=False):
    source_valid = self._valid_for(source_ids, source_valid, "source")
    target_valid = self._valid_for(target_ids, target_valid, "target")
    source_allowed, target_allowed = build_masks(source_valid, target_valid)
    memory = self._embed(source_ids, self.source_embedding)
    encoder_weights, decoder_weights, cross_weights = [], [], []
    for layer in self.encoder_layers:
        memory, weights = layer(memory, source_allowed)
        encoder_weights.append(weights)
    x = self._embed(target_ids, self.target_embedding)
    for layer in self.decoder_layers:
        x, self_weights, cross = layer(x, memory, target_allowed, source_allowed)
        decoder_weights.append(self_weights)
        cross_weights.append(cross)
    logits = self.vocabulary_projection(x)
    if return_weights:
        return logits, {"encoder": encoder_weights,
                        "decoder": decoder_weights, "cross": cross_weights}
    return logits
