from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "training dependency not installed")
class SparseModelTests(unittest.TestCase):
    def test_small_model_exposes_all_structured_heads(self) -> None:
        import torch
        from ubx.model import UBXModel, UBXModelConfig

        config = UBXModelConfig(
            width=64,
            depth=3,
            dense_blocks=1,
            heads=4,
            expert_hidden=96,
            routed_experts=4,
            top_k=2,
            roles=4,
            goals=4,
        )
        model = UBXModel(config).eval()
        grid = torch.zeros((1, 64, 64), dtype=torch.long)
        with torch.inference_mode():
            output = model(grid, grid)
        self.assertEqual(output["action_logits"].shape, (1, 8))
        self.assertEqual(output["click_logits"].shape, (1, 64, 64))
        self.assertEqual(output["next_action_logits"].shape, (1, 8, 8))
        self.assertEqual(output["next_latents"].shape, (1, 8, 64))
        self.assertEqual(output["routing_logits"].shape, (1, 2, 256, 4))

    def test_default_parameter_budget(self) -> None:
        from ubx.model import UBXModel

        model = UBXModel()
        report = model.parameter_report()
        self.assertGreaterEqual(report["total_parameters"], 170_000_000)
        self.assertLessEqual(report["total_parameters"], 250_000_000)
        self.assertGreaterEqual(report["estimated_active_parameters"], 50_000_000)
        self.assertLessEqual(report["estimated_active_parameters"], 80_000_000)
        self.assertLess(report["int8_megabytes"], 1000)


if __name__ == "__main__":
    unittest.main()
