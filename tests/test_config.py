"""Tests for configuration file handling."""

from pathlib import Path

import pytest

from osmium.utils.config import load_yaml_config, save_yaml_config


class TestLoadYamlConfig:
    """tests for YAML config loading"""

    def test_loads_simple_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("epochs: 10\nbatch_size: 32\n")

        result = load_yaml_config(config_file)

        assert result["epochs"] == 10
        assert result["batch_size"] == 32

    def test_loads_nested_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
optimizer:
  lr: 0.001
  weight_decay: 0.1
model:
  layers: 12
""")
        result = load_yaml_config(config_file)

        assert result["optimizer"]["lr"] == 0.001
        assert result["optimizer"]["weight_decay"] == 0.1
        assert result["model"]["layers"] == 12

    def test_returns_empty_dict_for_empty_file(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        result = load_yaml_config(config_file)

        assert result == {}

    def test_raises_for_nonexistent_file(self, tmp_path):
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_yaml_config(nonexistent)

        assert "Config file not found" in str(exc_info.value)

    def test_handles_boolean_values(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("mixed_precision: true\ncompile: false\n")

        result = load_yaml_config(config_file)

        assert result["mixed_precision"] is True
        assert result["compile"] is False

    def test_handles_float_values(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        # note: YAML requires 3.0e-4 format, not 3e-4 which is parsed as string
        config_file.write_text("learning_rate: 0.0003\nmin_lr: 0.00001\n")

        result = load_yaml_config(config_file)

        assert result["learning_rate"] == 0.0003
        assert result["min_lr"] == 0.00001

    def test_handles_null_values(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("warmup_steps: null\nmax_tokens: ~\n")

        result = load_yaml_config(config_file)

        assert result["warmup_steps"] is None
        assert result["max_tokens"] is None


class TestSaveYamlConfig:
    """tests for YAML config saving"""

    def test_saves_simple_config(self, tmp_path):
        config = {"epochs": 10, "batch_size": 32}
        output_path = tmp_path / "output.yaml"

        save_yaml_config(config, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "epochs: 10" in content
        assert "batch_size: 32" in content

    def test_creates_parent_directories(self, tmp_path):
        config = {"key": "value"}
        output_path = tmp_path / "nested" / "dir" / "config.yaml"

        save_yaml_config(config, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_preserves_key_order(self, tmp_path):
        config = {"z_last": 1, "a_first": 2, "m_middle": 3}
        output_path = tmp_path / "output.yaml"

        save_yaml_config(config, output_path)

        content = output_path.read_text()
        z_pos = content.find("z_last")
        a_pos = content.find("a_first")
        m_pos = content.find("m_middle")
        # order should be preserved (z before a before m)
        assert z_pos < a_pos < m_pos

    def test_roundtrip_preserves_data(self, tmp_path):
        original = {
            "epochs": 10,
            "learning_rate": 3e-4,
            "mixed_precision": True,
            "warmup_steps": None,
        }
        config_path = tmp_path / "roundtrip.yaml"

        save_yaml_config(original, config_path)
        loaded = load_yaml_config(config_path)

        assert loaded == original
