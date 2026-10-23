import unittest
import numpy as np
from conformal.prediction import calibrate, prediction_sets, evaluate


class ConformalTests(unittest.TestCase):
    def test_finite_sample_order_statistic(self):
        # n=4, alpha=.4 => rank=3, not an interpolated quantile.
        q, scores, rank = calibrate([.1, .2, .3, .4], [0, 0, 0, 0], .4)
        self.assertEqual(rank, 3)
        self.assertAlmostEqual(q, .3)
        np.testing.assert_allclose(scores, [.1, .2, .3, .4])

    def test_insufficient_calibration_returns_full_sets(self):
        q, _, rank = calibrate([.1, .9], [0, 1], .05)
        self.assertEqual(rank, 3)
        self.assertTrue(np.isinf(q))
        self.assertTrue(prediction_sets([0, .5, 1], q).all())

    def test_scores_use_true_class(self):
        _, scores, _ = calibrate([.2, .8], [0, 1], .5)
        np.testing.assert_allclose(scores, [.2, .2])

    def test_ties_empty_ambiguous_and_singleton(self):
        np.testing.assert_array_equal(prediction_sets([.25, .5, .75], .25),
                                      [[True, False], [False, False], [False, True]])
        np.testing.assert_array_equal(prediction_sets([.5], .5), [[True, True]])

    def test_abstentions_and_false_positive_denominators(self):
        p = [.9, .5, .1, .8]
        sets = prediction_sets(p, .3)
        metrics, decisions, covered = evaluate(p, [0, 0, 0, 1], sets)
        np.testing.assert_array_equal(decisions, [1, -1, 0, 1])
        c = metrics['conformal']
        self.assertEqual(c['abstention_rate'], .25)
        self.assertEqual(c['coverage'], .5)
        self.assertAlmostEqual(c['false_positive_rate_all_negatives'], 1/3)
        self.assertEqual(c['false_positive_rate_decided_negatives'], .5)

    def test_no_decisions_is_null_accuracy(self):
        m, _, _ = evaluate([.5], [0], [[True, True]])
        self.assertIsNone(m['conformal']['singleton_accuracy'])
        self.assertIsNone(m['conformal']['false_positive_rate_decided_negatives'])

    def test_invalid_inputs(self):
        for p, y, a in [([], [], .1), ([np.nan], [0], .1), ([1.1], [0], .1),
                        ([.5], [2], .1), ([.5], [], .1), ([.5], [0], 0)]:
            with self.assertRaises(ValueError):
                calibrate(p, y, a)


if __name__ == '__main__':
    unittest.main()
