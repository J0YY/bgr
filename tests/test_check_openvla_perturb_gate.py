import unittest

from scripts.check_openvla_perturb_gate import evaluate_gate


def _row(method: str, perturbation: str, successes: int, episodes: int = 100) -> dict[str, str]:
    return {
        "method": method,
        "perturbation": perturbation,
        "successes": str(successes),
        "episodes": str(episodes),
    }


def _clean_row(method: str, successes: int, episodes: int = 100) -> dict[str, str]:
    return {
        "method": method,
        "successes": str(successes),
        "episodes": str(episodes),
    }


class OpenVLAPerturbGateTest(unittest.TestCase):
    def test_passes_when_bgr_beats_both_comparators_and_preserves_identity(self) -> None:
        rows = [
            _row("bgr", "identity", 99),
            _row("official", "identity", 99),
            _row("random", "identity", 99),
        ]
        for perturbation in ["blur", "brightness", "occlusion", "shift"]:
            rows.append(_row("bgr", perturbation, 98))
            rows.append(_row("official", perturbation, 94))
            rows.append(_row("random", perturbation, 95))

        decision = evaluate_gate(rows)

        self.assertTrue(decision.complete)
        self.assertTrue(decision.passed)
        self.assertIn("episode_margin=12", decision.detail)
        self.assertIn("rate_margin=0.0300", decision.detail)

    def test_fails_against_best_comparator_even_if_official_is_lower(self) -> None:
        rows = [
            _row("bgr", "identity", 99),
            _row("official", "identity", 99),
            _row("random", "identity", 99),
        ]
        for perturbation in ["blur", "brightness", "occlusion", "shift"]:
            rows.append(_row("bgr", perturbation, 98))
            rows.append(_row("official", perturbation, 94))
            rows.append(_row("random", perturbation, 97))

        decision = evaluate_gate(rows)

        self.assertTrue(decision.complete)
        self.assertFalse(decision.passed)
        self.assertIn("best_comparator=random", decision.detail)
        self.assertIn("episode_margin=4", decision.detail)

    def test_reports_incomplete_until_all_non_identity_rows_exist(self) -> None:
        rows = [
            _row("bgr", "identity", 99),
            _row("official", "identity", 99),
            _row("random", "identity", 99),
            _row("bgr", "blur", 98),
            _row("official", "blur", 97),
            _row("random", "blur", 99),
        ]

        decision = evaluate_gate(rows)

        self.assertFalse(decision.complete)
        self.assertFalse(decision.passed)
        self.assertIn("missing bgr/brightness", decision.detail)

    def test_clean_summary_can_define_identity_floor(self) -> None:
        rows = [
            _row("bgr", "identity", 99),
            _row("official", "identity", 99),
            _row("random", "identity", 99),
        ]
        for perturbation in ["blur", "brightness", "occlusion", "shift"]:
            rows.append(_row("bgr", perturbation, 98))
            rows.append(_row("official", perturbation, 94))
            rows.append(_row("random", perturbation, 95))

        decision = evaluate_gate(
            rows,
            clean_rows=[
                _clean_row("bgr", 97),
                _clean_row("random", 99),
            ],
        )

        self.assertTrue(decision.complete)
        self.assertFalse(decision.passed)
        self.assertIn("identity_deficit=2", decision.detail)


if __name__ == "__main__":
    unittest.main()
