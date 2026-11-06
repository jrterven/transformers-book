"""Component references and full-model information-flow checks for Chapter 5."""

import math
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from torch import nn
    from torch.nn import functional as F
    from examples.ch05_transformer.model import (
        DecoderLayer, EncoderLayer, MultiHeadAttention, PositionalEncoding,
        PositionwiseFeedForward, Transformer, build_masks,
    )
    from examples.ch05_transformer.run_examples import (
        check_prefix_and_padding, example_batch, full_model_example, toy_feed_forward, toy_multihead,
        vocabulary_readout_example,
    )


@unittest.skipUnless(torch is not None, "Chapter 5 needs requirements-ch05.txt")
class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def assertClose(self, actual, expected, atol=1e-12):
        torch.testing.assert_close(actual, expected, rtol=0, atol=atol)

    def test_sinusoidal_values_odd_width_and_input_scaling(self):
        position = PositionalEncoding(4, 3).double()
        self.assertClose(position.pe[0, 0], torch.tensor([0., 1., 0., 1.]).double())
        expected = torch.tensor([math.sin(1), math.cos(1), math.sin(.01), math.cos(.01)],
                                dtype=torch.float64)
        self.assertClose(position.pe[0, 1], expected)
        odd = PositionalEncoding(5, 3).double()
        self.assertEqual(tuple(odd.pe.shape), (1, 3, 5))
        x = torch.randn(2, 3, 5, dtype=torch.float64)
        self.assertClose(odd(x) - x, odd.pe.expand(2, -1, -1))
        model = Transformer(12, 12, dropout=0).double().eval()
        ids = torch.tensor([[1, 3, 4]])
        expected = model.source_embedding(ids) * 2 + model.position.pe[:, :3]
        self.assertClose(model._embed(ids, model.source_embedding), expected)
        with self.assertRaisesRegex(ValueError, "max_length"):
            position(torch.zeros(1, 4, 4, dtype=torch.float64))

    def test_split_merge_preserves_head_order(self):
        module = MultiHeadAttention(6, 3).double()
        x = torch.arange(2 * 4 * 6, dtype=torch.float64).reshape(2, 4, 6)
        split = module.split_heads(x)
        self.assertEqual(tuple(split.shape), (2, 3, 4, 2))
        self.assertClose(split[1, 2, 3], x[1, 3, 4:6])
        self.assertClose(module.merge_heads(split), x)
        with self.assertRaisesRegex(ValueError, "divisible"):
            MultiHeadAttention(5, 2)

    def test_two_head_hardcoded_reference(self):
        output, weights, combined = toy_multihead()
        expected = torch.tensor([
            [1.2033362780393578, 1.0961076756207493, 1.2033362780393578, .9972197683005353],
            [.7149289392509497, 1.6806420965822268, .5315260175211337, 1.556279776105515],
            [1.051188875402384, 1.401112092679786, .9747603681913526, 1.302224185359572],
        ], dtype=torch.float64)
        self.assertClose(output[0], expected, atol=2e-12)
        self.assertClose(weights.sum(-1), torch.ones(1, 2, 3, dtype=torch.float64))
        self.assertClose(combined[0, 0], torch.tensor(
            [.8022241853595719, .7966637219606423, .8022241853595719, .5988879073202141],
            dtype=torch.float64))
        self.assertGreater(abs(weights[0, 0, 1, 1] - weights[0, 1, 1, 1]), .3)

    def test_rectangular_attention_matches_torch_multihead_module(self):
        module = MultiHeadAttention(6, 3).double()
        query = torch.randn(2, 4, 6, dtype=torch.float64)
        memory = torch.randn(2, 5, 6, dtype=torch.float64)
        mask = torch.tensor([[True, False, True, True, False]])[None, None]
        actual, weights = module(query, memory, memory, mask)
        reference = nn.MultiheadAttention(6, 3, bias=False, dropout=0,
                                          batch_first=True).double()
        with torch.no_grad():
            reference.in_proj_weight.copy_(torch.cat([
                module.q_proj.weight, module.k_proj.weight, module.v_proj.weight,
            ], dim=0))
            reference.out_proj.weight.copy_(module.out_proj.weight)
        # nn.MultiheadAttention uses True for blocked entries, unlike our API.
        blocked = ~mask.expand(2, 3, 4, 5).reshape(2 * 3, 4, 5)
        expected, expected_weights = reference(
            query, memory, memory, attn_mask=blocked,
            need_weights=True, average_attn_weights=False,
        )
        self.assertClose(actual, expected)
        self.assertClose(weights, expected_weights)
        self.assertEqual(tuple(weights.shape), (2, 3, 4, 5))
        self.assertClose(weights[..., 1], torch.zeros(2, 3, 4, dtype=torch.float64))

    def test_attention_input_and_mask_checks(self):
        module = MultiHeadAttention(4, 2).double()
        x = torch.ones(2, 3, 4, dtype=torch.float64)
        cases = [
            (x[0], x, x, None, ValueError),
            (x, x[:1], x[:1], None, ValueError),
            (x, x, x[:, :2], None, ValueError),
            (x, x, x, torch.ones(3, 3), TypeError),
            (x, x, x, torch.ones(5, 3, dtype=torch.bool), ValueError),
            (x, x, x, torch.zeros(3, 3, dtype=torch.bool), ValueError),
        ]
        for q, k, v, mask, error in cases:
            with self.subTest(error=error):
                with self.assertRaises(error):
                    module(q, k, v, mask)
        bad = x.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            module(bad, x, x)

    def test_ffn_is_shared_and_positionwise(self):
        module = PositionwiseFeedForward(4, 8).double()
        x = torch.randn(2, 3, 4, dtype=torch.float64)
        actual = module(x)
        expected = torch.stack([torch.stack([module(row) for row in batch]) for batch in x])
        self.assertClose(actual, expected)
        changed = x.clone()
        changed[0, 1] += 20
        altered = module(changed)
        self.assertClose(altered[1], actual[1])
        self.assertClose(altered[0, [0, 2]], actual[0, [0, 2]])

    def test_feed_forward_hand_worked_values(self):
        hidden, output = toy_feed_forward()
        self.assertClose(hidden, torch.tensor([1., 0., 0., 3., 0., 2., 0., 0.]).double())
        self.assertClose(output, torch.tensor([1., 2., 0., 3.]).double())

    def test_vocabulary_readout_and_logit_loss(self):
        logits, probabilities = vocabulary_readout_example()
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertClose(logits, torch.tensor([[2., 1., 0.]], dtype=torch.float64))
        self.assertClose(probabilities, torch.tensor([
            [.6652409557748218, .2447284710547976, .0900305731703805],
        ], dtype=torch.float64))
        # With "coche" as the label, CE consumes the logits, not these probabilities.
        loss = F.cross_entropy(logits, torch.tensor([1]))
        self.assertAlmostEqual(loss.item(), 1.4076059644443804, places=12)

    def test_layer_norm_uses_last_axis_population_variance(self):
        norm = nn.LayerNorm(4, eps=1e-5).double()
        x = torch.tensor([[[1., 2., 3., 4.], [4., 2., 0., -2.]]], dtype=torch.float64)
        expected = (x - x.mean(-1, keepdim=True)) / torch.sqrt(
            x.var(-1, correction=0, keepdim=True) + 1e-5)
        self.assertClose(norm(x), expected)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1., 2., 3., 4.]))
            norm.bias.fill_(.5)
        self.assertClose(norm(x), expected * norm.weight + norm.bias)

    def test_encoder_is_post_norm(self):
        layer = EncoderLayer(4, 2, 8, dropout=0).double()
        x = torch.randn(2, 3, 4, dtype=torch.float64)
        with torch.no_grad():
            layer.self_attention.out_proj.weight.zero_()
            layer.feed_forward.contract.weight.zero_()
            layer.feed_forward.contract.bias.zero_()
        actual, _ = layer(x, torch.ones(2, 1, 1, 3, dtype=torch.bool))
        self.assertClose(actual, layer.norm2(layer.norm1(x)))

    def test_combined_masks_include_diagonal_and_exclude_padding(self):
        source_valid = torch.tensor([[True, True, False], [True, False, False]])
        target_valid = torch.tensor([[True, True, True], [True, True, False]])
        source, target = build_masks(source_valid, target_valid)
        self.assertEqual(tuple(source.shape), (2, 1, 1, 3))
        self.assertEqual(tuple(target.shape), (2, 1, 3, 3))
        self.assertTrue(torch.equal(target[0, 0], torch.ones(3, 3, dtype=torch.bool).tril()))
        self.assertTrue(torch.equal(target[1, 0, 2], torch.tensor([True, True, False])))
        for bad in (torch.tensor([[False, True]]), torch.tensor([[True, False, True]])):
            with self.assertRaisesRegex(ValueError, "right-padded"):
                build_masks(bad, bad)

    def test_full_model_prefix_and_padding_invariance(self):
        model = Transformer(12, 12, n_encoder_layers=2, n_decoder_layers=2).double()
        source, target, _ = example_batch()
        check_prefix_and_padding(model, source, target)
        model.eval()
        _, weights = model(source, target, return_weights=True)
        self.assertEqual(len(weights["encoder"]), 2)
        self.assertEqual(tuple(weights["cross"][0].shape), (2, 2, 4, 5))
        self.assertClose(weights["encoder"][1][0, :, :, 4], torch.zeros(2, 5, dtype=torch.float64))
        self.assertClose(weights["cross"][1][1, :, :, 3:], torch.zeros(2, 4, 2, dtype=torch.float64))

    def test_stacks_and_gradients_match_torch_reference(self):
        torch.manual_seed(1238)
        model = Transformer(16, 17, d_model=6, n_heads=3, d_ff=10,
                            n_encoder_layers=2, n_decoder_layers=2,
                            dropout=0).double().eval()
        encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(6, 3, 10, dropout=0, batch_first=True,
                                       norm_first=False).double(),
            2, norm=None, enable_nested_tensor=False,
        ).eval()
        decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(6, 3, 10, dropout=0, batch_first=True,
                                       norm_first=False).double(),
            2, norm=None,
        ).eval()
        projection = nn.Linear(6, 17).double()
        parameter_pairs = []

        def copy_module(reference, original):
            reference.load_state_dict(original.state_dict())
            for name, parameter in original.named_parameters():
                parameter_pairs.append((parameter, reference.get_parameter(name), None))

        def copy_attention(reference, original):
            projections = (original.q_proj, original.k_proj, original.v_proj)
            with torch.no_grad():
                reference.in_proj_weight.copy_(torch.cat([
                    part.weight for part in projections
                ]))
                reference.in_proj_bias.zero_()
                reference.out_proj.weight.copy_(original.out_proj.weight)
                reference.out_proj.bias.zero_()
            for index, part in enumerate(projections):
                parameter_pairs.append((part.weight, reference.in_proj_weight, index))
            parameter_pairs.append((original.out_proj.weight,
                                    reference.out_proj.weight, None))

        for reference, original in zip(encoder.layers, model.encoder_layers):
            copy_attention(reference.self_attn, original.self_attention)
            copy_module(reference.linear1, original.feed_forward.expand)
            copy_module(reference.linear2, original.feed_forward.contract)
            copy_module(reference.norm1, original.norm1)
            copy_module(reference.norm2, original.norm2)
        for reference, original in zip(decoder.layers, model.decoder_layers):
            copy_attention(reference.self_attn, original.self_attention)
            copy_attention(reference.multihead_attn, original.cross_attention)
            copy_module(reference.linear1, original.feed_forward.expand)
            copy_module(reference.linear2, original.feed_forward.contract)
            for name in ("norm1", "norm2", "norm3"):
                copy_module(getattr(reference, name), getattr(original, name))
        copy_module(projection, model.vocabulary_projection)

        source_ids = torch.tensor([[1, 3, 8, 2, 0], [1, 2, 0, 0, 0], [1, 5, 7, 6, 2]])
        target_ids = torch.tensor([[1, 6, 2, 0], [1, 7, 5, 2], [1, 0, 0, 0]])
        source_valid, target_valid = source_ids != 0, target_ids != 0
        source_allowed, target_allowed = build_masks(source_valid, target_valid)
        # Compare the complete stacks from their embedded/scaled/positioned inputs.
        # Embedding lookup gradients are covered by the full-model backward tests.
        source = model._embed(source_ids, model.source_embedding).detach().requires_grad_()
        target = model._embed(target_ids, model.target_embedding).detach().requires_grad_()
        source_reference = source.detach().clone().requires_grad_()
        target_reference = target.detach().clone().requires_grad_()
        memory = source
        for layer in model.encoder_layers:
            memory, _ = layer(memory, source_allowed)
        output = target
        for layer in model.decoder_layers:
            output, _, _ = layer(output, memory, target_allowed, source_allowed)
        actual = model.vocabulary_projection(output)
        memory_reference = encoder(source_reference, src_key_padding_mask=~source_valid)
        causal_blocked = ~torch.ones(4, 4, dtype=torch.bool).tril()
        expected = projection(decoder(
            target_reference, memory_reference, tgt_mask=causal_blocked,
            tgt_key_padding_mask=~target_valid, memory_key_padding_mask=~source_valid,
        ))
        self.assertClose(actual, expected, atol=1e-11)

        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        self.assertClose(source.grad, source_reference.grad, atol=1e-11)
        self.assertClose(target.grad, target_reference.grad, atol=1e-11)
        stack_parameters = [*model.encoder_layers.parameters(),
                            *model.decoder_layers.parameters(),
                            *model.vocabulary_projection.parameters()]
        self.assertEqual({id(part) for part in stack_parameters},
                         {id(part) for part, _, _ in parameter_pairs})
        for original, reference, index in parameter_pairs:
            with self.subTest(shape=tuple(original.shape), projection=index):
                self.assertIsNotNone(original.grad)
                self.assertIsNotNone(reference.grad)
                expected_gradient = reference.grad
                if index is not None:
                    expected_gradient = expected_gradient[index * 6:(index + 1) * 6]
                self.assertClose(original.grad, expected_gradient, atol=1e-11)

    def test_decoder_cross_attention_receives_memory(self):
        layer = DecoderLayer(4, 2, 8, dropout=0).double()
        x = torch.randn(1, 2, 4, dtype=torch.float64)
        memory = torch.randn(1, 5, 4, dtype=torch.float64)
        causal = torch.ones(2, 2, dtype=torch.bool).tril()
        result, self_weights, cross = layer(x, memory, causal, None)
        changed, _, _ = layer(x, memory * 2, causal, None)
        self.assertEqual(tuple(self_weights.shape), (1, 2, 2, 2))
        self.assertEqual(tuple(cross.shape), (1, 2, 2, 5))
        self.assertGreater((changed - result).abs().max().item(), 1e-5)

    def test_layer_parameters_are_independent(self):
        model = Transformer(12, 12, n_encoder_layers=2, n_decoder_layers=2)
        first = model.encoder_layers[0].self_attention.q_proj.weight
        second = model.encoder_layers[1].self_attention.q_proj.weight
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        original = second.detach().clone()
        with torch.no_grad():
            first.fill_(7)
        self.assertTrue(torch.equal(second, original))
        self.assertNotEqual(model.decoder_layers[0].norm1.weight.data_ptr(),
                            model.decoder_layers[1].norm1.weight.data_ptr())

    def test_eval_is_deterministic_and_keeps_autograd(self):
        model = Transformer(12, 12, dropout=.5).double().eval()
        source, target, labels = example_batch()
        a = model(source, target)
        b = model(source, target)
        self.assertClose(a, b, atol=0)
        self.assertTrue(a.requires_grad)
        F.cross_entropy(a.reshape(-1, 12), labels.reshape(-1), ignore_index=0).backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.source_embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.decoder_layers[0].cross_attention.q_proj.weight.grad.abs().sum().item(), 0)

    def test_dropout_changes_training_outputs(self):
        model = Transformer(12, 12, dropout=.5).double().train()
        source, target, _ = example_batch()
        a = model(source, target)
        b = model(source, target)
        self.assertFalse(torch.equal(a, b))
        model.eval()
        self.assertClose(model(source, target), model(source, target), atol=0)

    def test_shifted_targets_and_padding_aware_objective(self):
        _, target, labels = example_batch()
        self.assertTrue(torch.equal(target[0, 1:], labels[0, :-1]))
        self.assertEqual(labels[1, -1].item(), 0)
        logits = torch.randn(2, 4, 12, dtype=torch.float64)
        actual = F.cross_entropy(logits.reshape(-1, 12), labels.reshape(-1), ignore_index=0)
        expected = F.cross_entropy(logits[labels != 0], labels[labels != 0])
        self.assertClose(actual, expected)

    def test_output_projection_changes_only_mixed_features(self):
        original, weights, concatenated = toy_multihead()
        identity, new_weights, new_concat = toy_multihead(torch.eye(4))
        self.assertClose(new_weights, weights)
        self.assertClose(new_concat, concatenated)
        self.assertClose(identity, concatenated)
        self.assertFalse(torch.allclose(identity, original))
        # A nonsymmetric matrix detects a forgotten row-convention transpose.
        projection = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 8
        mixed, _, _ = toy_multihead(projection)
        self.assertClose(mixed, concatenated @ projection)
        for invalid in (torch.eye(3), torch.full((4, 4), float("nan"))):
            with self.assertRaises(ValueError):
                toy_multihead(invalid)

    def test_ignored_label_gradient_and_valid_mean(self):
        _, _, labels = example_batch()
        logits = torch.zeros(2, 4, 12, dtype=torch.float64, requires_grad=True)
        rows = F.cross_entropy(logits.reshape(-1, 12), labels.flatten(),
                               ignore_index=0, reduction="none").reshape(2, 4)
        loss = F.cross_entropy(logits.reshape(-1, 12), labels.flatten(),
                               ignore_index=0)
        self.assertEqual((labels != 0).sum().item(), 7)
        self.assertClose(loss, logits.new_tensor(math.log(12)))
        self.assertClose(rows.mean(), loss * (7 / 8))
        loss.backward()
        self.assertClose(logits.grad[labels == 0], torch.zeros_like(logits.grad[labels == 0]))
        # PAD remains a competing vocabulary class at every scored row.
        self.assertClose(logits.grad[..., 0][labels != 0], logits.new_full((7,), 1 / 84))

    def test_full_demo(self):
        logits, attention, loss = full_model_example()
        self.assertEqual(tuple(logits.shape), (2, 4, 12))
        self.assertEqual(tuple(attention["encoder"][0].shape), (2, 2, 5, 5))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
