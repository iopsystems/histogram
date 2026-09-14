import unittest
import run


class RunnerTests(unittest.TestCase):
    def test_missing_metadata_command_is_nonfatal(self):
        result = run.capture(["/definitely-not-installed/histogram-metadata"])
        self.assertIn("unavailable", result)

    def test_rejects_missing_duplicate_and_invalid_timing_rows(self):
        row = dict(width="32", gp="7", max_power="30", windows="8",
                   shape="clustered", report="false", method="fused",
                   iterations="10", elapsed_ns="100", ns_per_set="10.000")
        expected = {tuple(row[k] for k in run.KEYS)}
        run.validate_rows([row], expected)
        for bad in [[], [row, row], [dict(row, ns_per_set="nan")],
                    [dict(row, iterations="0")], [dict(row, ns_per_set="99")],
                    [dict(row, method="unknown")]]:
            with self.assertRaises(ValueError):
                run.validate_rows(bad, expected)

    def test_builds_summary_from_block_samples(self):
        rows = [dict(width="32", gp="7", max_power="30", windows="8",
                     shape="clustered", report="false", method="fused",
                     ns_per_set=str(ns), block=block)
                for block, ns in enumerate([10, 30, 20])]
        summary = run.summarize(rows, 3)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["median_ns"], 20)
        self.assertEqual(summary[0]["min_ns"], 10)
        self.assertEqual(summary[0]["max_ns"], 30)
        with self.assertRaises(ValueError):
            run.summarize(rows[:2], 3)
        with self.assertRaises(ValueError):
            run.summarize([rows[0], rows[0], rows[2]], 3)


if __name__ == "__main__":
    unittest.main()
