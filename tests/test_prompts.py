import json
import unittest

from pa_magcapposevlm.prompts import PROMPTS, VALID_CODES
from scripts.build_vlm_manifests import task_rows, with_prompt


class PromptTests(unittest.TestCase):
    def test_prompt_matrix_is_complete(self):
        self.assertEqual(set(PROMPTS), {"standard", "concise", "reordered"})
        self.assertTrue(
            all(set(tasks) == {"QC_ONLY", "CODE_ONLY", "JOINT"} for tasks in PROMPTS.values())
        )

    def test_legal_codes_are_three_bit_strings(self):
        self.assertEqual(VALID_CODES, tuple(f"{value:03b}" for value in range(8)))

    def test_canonical_answers_are_json_serializable(self):
        answers = [{"validity": "VALID"}] + [{"code": code} for code in VALID_CODES]
        for answer in answers:
            self.assertIsInstance(json.loads(json.dumps(answer)), dict)

    def test_task_expansion(self):
        accepted = {
            "example_id": "a",
            "partition": "train",
            "image_path": "a.png",
            "label": "ACCEPT",
            "true_code": "010",
        }
        rejected = {
            "example_id": "r",
            "partition": "train",
            "image_path": "r.png",
            "label": "REJECT",
            "true_code": None,
        }
        accepted_rows = task_rows(accepted)
        rejected_rows = task_rows(rejected)
        self.assertEqual({row["task"] for row in accepted_rows}, {"QC_ONLY", "CODE_ONLY", "JOINT"})
        self.assertEqual({row["task"] for row in rejected_rows}, {"QC_ONLY", "JOINT"})
        prompted = with_prompt(accepted_rows[0], "standard")
        self.assertEqual(prompted["source_example_id"], "a")
        self.assertIn("system_prompt", prompted)


if __name__ == "__main__":
    unittest.main()
