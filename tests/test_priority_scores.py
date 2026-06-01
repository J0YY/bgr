import unittest

from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord


class PriorityScoreTest(unittest.TestCase):
    def test_priority_downweights_clean_failing_state(self):
        scorer = BGRPriorityScorer()
        record = LevelRecord(
            id="x",
            domain="d",
            task_id="t",
            clean_success_hat=0.1,
            feasibility_hat=1.0,
            r_alpha_hat=0.4,
            sharpness_hat=1.0,
            uncertainty_hat=0.2,
        )
        self.assertEqual(scorer.score(record), scorer.min_priority)

    def test_priority_prefers_target_radius_over_saturated_radius(self):
        scorer = BGRPriorityScorer(target_radius=0.4)
        base = dict(
            domain="d",
            task_id="t",
            clean_success_hat=0.9,
            feasibility_hat=1.0,
            sharpness_hat=1.0,
            uncertainty_hat=0.2,
        )
        target = LevelRecord(id="target", r_alpha_hat=0.4, **base)
        saturated = LevelRecord(id="sat", r_alpha_hat=0.95, **base)
        self.assertGreater(scorer.score(target), scorer.score(saturated))
