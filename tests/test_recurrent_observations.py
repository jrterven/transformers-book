"""Check observed gates against an independent native LSTM computation."""

import unittest
import numpy as np
import torch

from examples.ch02_recurrence.inspect_trained import trace_lstm


class RecurrentObservationTests(unittest.TestCase):
    def test_observed_gates_reconstruct_every_native_state(self):
        torch.manual_seed(73)
        layer = torch.nn.LSTM(3, 5, batch_first=True).double().eval()
        embeddings = torch.randn(4, 3, dtype=torch.float64)
        trace = trace_lstm(layer, embeddings)
        with torch.no_grad():
            native, (_, final_cell) = layer(embeddings[None])
        np.testing.assert_allclose(trace['hidden'], native[0].numpy(), atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(trace['cell'][-1], final_cell[0, 0].numpy(), atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(trace['cell'], trace['forget'] * trace['old_cell']
                                   + trace['write'] * trace['candidate'], atol=1e-15)
        np.testing.assert_allclose(trace['hidden'], trace['expose'] * np.tanh(trace['cell']), atol=1e-15)
        np.testing.assert_array_equal(trace['old_cell'][0], np.zeros(5))
        np.testing.assert_array_equal(trace['old_cell'][1:], trace['cell'][:-1])
        for key in ('forget', 'write', 'expose'):
            self.assertTrue(((trace[key] > 0) & (trace[key] < 1)).all())


if __name__ == '__main__':
    unittest.main()
