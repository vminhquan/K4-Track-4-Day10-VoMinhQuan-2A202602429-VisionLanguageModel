"""CPU checks for the student-owned label conversion."""
import unittest
from lab_workflow import answer_for, yolo_to_box


class GroundingLabelTests(unittest.TestCase):
    def test_yolo_to_norm1000(self):
        self.assertEqual(
            yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.4]),
            {"label": "pothole", "bbox_2d": [400, 300, 600, 700]},
        )

    def test_clipping(self):
        self.assertEqual(
            yolo_to_box("crack", [0.02, 0.5, 0.2, 0.2])["bbox_2d"],
            [0, 400, 120, 600],
        )

    def test_thin_box_remains_valid_after_rounding(self):
        self.assertEqual(
            yolo_to_box("crack", [0.639, 0.7863, 0.004, 0.0001])["bbox_2d"],
            [637, 786, 641, 787],
        )

    def test_order_and_empty(self):
        rows = [
            {"label": "crack", "yolo": [0.5, 0.5, 0.2, 0.2]},
            {"label": "pothole", "yolo": [0.1, 0.1, 0.1, 0.1]},
        ]
        self.assertEqual(answer_for([]), "[]")
        self.assertTrue(answer_for(rows).startswith('[{"label":"pothole"'))


if __name__ == "__main__":
    unittest.main()
