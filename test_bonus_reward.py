"""Run with: python -m unittest -v test_bonus_reward.py"""
import unittest
from bonus_reward import reward_iou, trl_reward

GT = [{"label": "pothole", "bbox": [100, 100, 300, 300]}]
BOX = '{"label":"pothole","bbox_2d":[100,100,300,300]}'


class BonusRewardTests(unittest.TestCase):
    def test_perfect(self):
        self.assertAlmostEqual(reward_iou(f"[{BOX}]", GT), 1.0)

    def test_invalid_json(self):
        self.assertEqual(reward_iou("not a JSON list", GT), 0.0)

    def test_wrong_class(self):
        self.assertAlmostEqual(reward_iou('[{"label":"crack","bbox_2d":[100,100,300,300]}]', GT), 0.2)

    def test_extra_box(self):
        extra = '{"label":"pothole","bbox_2d":[600,600,800,800]}'
        self.assertAlmostEqual(reward_iou(f"[{BOX},{extra}]", GT), 0.6)

    def test_both_empty(self):
        self.assertAlmostEqual(reward_iou("[]", []), 1.0)

    def test_trl_batch(self):
        self.assertEqual(trl_reward([f"[{BOX}]", "bad"], [GT, GT]), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
