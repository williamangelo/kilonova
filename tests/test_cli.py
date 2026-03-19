"""Tests for CLI command error handling and validation."""

from click.testing import CliRunner

from kilonova.cli import cli


class TestTrainValidation:
    """tests for train command input validation"""

    def test_requires_data_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "gpt2-small"])
        assert result.exit_code != 0
        assert "Missing option '--data'" in result.output

    def test_rejects_unknown_architecture(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "unknown-arch", "--data", "test"])
        assert result.exit_code != 0

    def test_rejects_missing_dataset(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "train", "gpt2-small",
            "--data", "nonexistent",
        ])
        assert result.exit_code != 0
        assert "not preprocessed" in result.output or "Dataset" in result.output


class TestEvaluateValidation:
    """tests for evaluate command input validation"""

    def test_rejects_nonexistent_run(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "nonexistent-run"])
        assert result.exit_code != 0
        assert "not found" in result.output


class TestGenerateValidation:
    """tests for generate command input validation"""

    def test_rejects_nonexistent_model(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "nonexistent", "--prompt", "test"])
        assert result.exit_code != 0
        assert "not found" in result.output
