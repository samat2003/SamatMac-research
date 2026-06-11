"""Training-data alignment checks for SamatNext-520M per SPEC.md."""

import unittest

from data.dataset import PythonCodeDataset
from data.fim import FIMSample
from model.config import SamatNextConfig


class _Padder:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def pad_to_length(self, samples, length):
        return [
            sample.tokens[:length]
            + [self.pad_id] * max(0, length - len(sample.tokens))
            for sample in samples
        ]


class DataAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.dataset = PythonCodeDataset.__new__(PythonCodeDataset)
        self.dataset.config = SamatNextConfig(max_seq_len=10)
        self.dataset.fim = _Padder()

    def test_non_fim_mask_excludes_padding(self):
        sample = FIMSample(
            tokens=[1, 2, 3, 4],
            is_fim=False,
            prefix_len=4,
            suffix_len=0,
            middle_len=0,
        )
        batch = self.dataset._make_batch([sample])

        self.assertEqual(batch.input_ids[0, :3].tolist(), [1, 2, 3])
        self.assertEqual(batch.targets[0, :3].tolist(), [2, 3, 4])
        self.assertEqual(batch.loss_mask[0].tolist(), [1, 1, 1, 0, 0, 0, 0, 0, 0])

    def test_fim_mask_marks_predictions_of_middle_tokens(self):
        # <prefix> p1 p2 <suffix> s1 s2 <middle> m1 m2 <eos>
        sample = FIMSample(
            tokens=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            is_fim=True,
            prefix_len=2,
            suffix_len=2,
            middle_len=2,
        )
        batch = self.dataset._make_batch([sample])

        self.assertEqual(batch.targets[0, 6:8].tolist(), [17, 18])
        self.assertEqual(batch.loss_mask[0].tolist(), [0, 0, 0, 0, 0, 0, 1, 1, 0])

    def test_only_final_four_gated_layers_write_memory(self):
        config = SamatNextConfig()
        writer_layers = [
            index
            for index in range(config.n_layers)
            if config.get_layer_type(index) == "gated_attention"
            and index >= config.memory_bus_start_layer
        ]
        self.assertEqual(writer_layers, [16, 18, 20, 22])


if __name__ == "__main__":
    unittest.main()
