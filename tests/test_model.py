import unittest
import torch
from src.model import RULLSTM


class ModelTests(unittest.TestCase):
    def test_rullstm_initialization(self) -> None:
        model = RULLSTM(input_size=18, hidden_size=64, num_layers=2, dropout=0.15)
        self.assertEqual(model.lstm.input_size, 18)
        self.assertEqual(model.lstm.hidden_size, 64)
        self.assertEqual(model.lstm.num_layers, 2)
        self.assertEqual(model.lstm.dropout, 0.15)

    def test_rullstm_invalid_parameters_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            RULLSTM(input_size=0)
        with self.assertRaises(ValueError):
            RULLSTM(input_size=10, hidden_size=-1)
        with self.assertRaises(ValueError):
            RULLSTM(input_size=10, num_layers=0)
        with self.assertRaises(ValueError):
            RULLSTM(input_size=10, dropout=1.0)
        with self.assertRaises(ValueError):
            RULLSTM(input_size=10, dropout=-0.1)

    def test_rullstm_forward_shape_batch(self) -> None:
        model = RULLSTM(input_size=18, hidden_size=32, num_layers=1, dropout=0.0)
        # Batch of 16 sequences, each of length 30 with 18 features
        x = torch.randn(16, 30, 18)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([16]))

    def test_rullstm_forward_shape_single_sample(self) -> None:
        model = RULLSTM(input_size=18, hidden_size=32, num_layers=1, dropout=0.0)
        # Single sequence (batch size 1)
        x = torch.randn(1, 30, 18)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([1]))

    def test_gradient_flow(self) -> None:
        model = RULLSTM(input_size=5, hidden_size=16, num_layers=1)
        x = torch.randn(4, 10, 5)
        y = torch.tensor([50.0, 40.0, 30.0, 20.0])
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # Check that gradients were computed for parameters
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    unittest.main()
