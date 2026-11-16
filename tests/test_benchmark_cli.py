from unittest.mock import patch

from scripts.run_benchmark import parse_args


def test_cli_requires_manifest():
    with patch("sys.argv", ["run_benchmark"]):
        try:
            parse_args()
            assert False, "Expected missing manifest to fail"
        except SystemExit as exc:
            assert exc.code == 2


def test_cli_accepts_manifest_and_pixel_budget():
    with patch(
        "sys.argv",
        [
            "run_benchmark",
            "--manifest",
            "data/manifest.jsonl",
            "--max-pixels",
            "401408",
        ],
    ):
        args = parse_args()

    assert str(args.manifest) == "data/manifest.jsonl"
    assert args.max_pixels == 401408


def test_cli_accepts_custom_results_directory():
    with patch(
        "sys.argv",
        [
            "run_benchmark",
            "--manifest",
            "data/manifest.jsonl",
            "--results-dir",
            "results/test",
        ],
    ):
        args = parse_args()

    assert str(args.results_dir) == "results/test"
