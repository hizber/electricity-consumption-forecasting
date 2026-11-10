import unittest

import torch

from recurrent import GRUForecaster, LSTMForecaster
from temporal import TCNRegressor, TransformerRegressor


class ModelTests(unittest.TestCase):
    def test_models_return_one_finite_prediction_per_example(self):
        for model_class in (LSTMForecaster, GRUForecaster, TCNRegressor, TransformerRegressor):
            with self.subTest(model=model_class.__name__):
                model = model_class(context_size=8)
                predictions = model(torch.randn(3, 24, 1), torch.randn(3, 8))
                self.assertEqual(tuple(predictions.shape), (3,))
                self.assertTrue(torch.isfinite(predictions).all())
                predictions.mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_tcn_encoder_is_causal(self):
        model = TCNRegressor(context_size=0, dropout=0).eval()
        history = torch.randn(1, 24, 1)
        modified = history.clone()
        modified[:, 12:, :] += 100
        with torch.no_grad():
            first = model.blocks(model.project(history.transpose(1, 2)))
            second = model.blocks(model.project(modified.transpose(1, 2)))
        self.assertTrue(torch.allclose(first[:, :, :12], second[:, :, :12]))


if __name__ == '__main__':
    unittest.main()
