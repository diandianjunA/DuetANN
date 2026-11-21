import shutil
import sys
import tempfile
from pathlib import Path

import configparser

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiment"))
from common.config import (
    ExperimentError, RESEARCH_KEYS, load_variant, required_mn_memory_gib,
    variant_options, write_config,
)


def snapshot(config):
    return {
        section: dict(config[section])
        for section in ("profile", "index", "transport", "gpu", "update", "benchmark")
    }


def main():
    config_dir = REPO / "experiment" / "sift100m"
    baseline = load_variant(config_dir / "baseline.conf")
    full = load_variant(config_dir / "full.conf")
    baseline_values = snapshot(baseline)
    full_values = snapshot(full)

    assert baseline["variant"]["name"] == "baseline"
    assert full["variant"]["name"] == "full"
    assert baseline["profile"]["profile"] == "baseline"
    assert full["profile"]["profile"] == "full"
    assert [baseline["profile"][key] for key in ("c1", "c2", "c3")] == \
        ["off", "off", "off"]
    assert [full["profile"][key] for key in ("c1", "c2", "c3")] == \
        ["on", "on", "on"]

    differences = set()
    for section in baseline_values:
        for key in set(baseline_values[section]) | set(full_values[section]):
            if baseline_values[section].get(key) != full_values[section].get(key):
                differences.add((section, key))
    assert differences == {("profile", key) for key in ("profile", "c1", "c2", "c3")}
    assert variant_options(baseline)["R"] == "96"
    assert variant_options(baseline)["profile"] == "baseline"
    assert variant_options(full)["c3"] == "on"

    contribution3_on = load_variant(config_dir / "full_contribution3_on.conf")
    contribution3_off = load_variant(config_dir / "full_contribution3_off.conf")
    on_values = snapshot(contribution3_on)
    off_values = snapshot(contribution3_off)
    ablation_differences = set()
    for section in on_values:
        for key in set(on_values[section]) | set(off_values[section]):
            if on_values[section].get(key) != off_values[section].get(key):
                ablation_differences.add((section, key))
    assert ablation_differences == {("profile", "c3")}
    for config in (contribution3_on, contribution3_off):
        assert config["profile"]["c1"] == "on"
        assert config["profile"]["c2"] == "off"

    motivation_dir = REPO / "experiment" / "motivation"
    c1 = load_variant(motivation_dir / "c1" / "baseline_coupled_insert.conf")
    assert c1["profile"]["c1"] == "off"
    assert c1["benchmark"]["workload"] == "insert"
    c2_configs = {
        path.stem: load_variant(path)
        for path in (motivation_dir / "c2").glob("*.conf")
    }
    assert c2_configs["baseline_fixed"]["profile"]["c2"] == "off"
    assert (c2_configs["baseline_header_neighbor"]["gpu"]["gpu-query-graph-read-policy"] ==
            "header-neighbor")
    assert c2_configs["baseline_live_extent"]["profile"]["c2"] == "on"
    for config in c2_configs.values():
        assert config["profile"]["c1"] == "on"
        assert config["benchmark"]["workload"] == "mixed"
        assert config["benchmark"]["write-insert-ratio"] == "0.5"
        assert config["benchmark"]["write-delete-ratio"] == "0.5"
        assert config["benchmark"]["mixed-delete-mode"] == "inserted"
        assert config["benchmark"]["mixed-mode"] == "write_rate_limited"
        assert config["benchmark"]["target-write-qps"] == "200"
        assert config["benchmark"]["write-threads"] == "16"
    highrate_configs = list(motivation_dir.glob(
        "*100m/c2/generated/*_highrate.conf"))
    highrate_configs += list((motivation_dir / "c2" / "generated").glob(
        "*_highrate.conf"))
    assert len(highrate_configs) == 9
    for path in highrate_configs:
        config = load_variant(path)
        if "live_extent" in path.stem:
            assert config["profile"]["c2"] == "on"
        assert config["benchmark"]["workload"] == "mixed"
        assert config["benchmark"]["write-insert-ratio"] == "1"
        assert config["benchmark"]["write-upsert-ratio"] == "0"
        assert config["benchmark"]["write-delete-ratio"] == "0"
        assert config["benchmark"]["mixed-delete-mode"] == "base"
        assert config["benchmark"]["mixed-mode"] == "fixed_threads"
        assert config["benchmark"]["target-write-qps"] == "0"
    c3 = load_variant(motivation_dir / "c3" / "baseline_high_expansion.conf")
    assert c3["profile"]["c3"] == "off"
    assert c3["gpu"]["gpu-max-expansions"] == "384"

    with tempfile.TemporaryDirectory(prefix="duetann-config-contract-") as temporary:
        temporary = Path(temporary)
        expanded_path = temporary / "full.conf"
        write_config(full, expanded_path)
        assert snapshot(load_variant(expanded_path)) == full_values

        shutil.copy2(config_dir / "runtime.conf", temporary / "runtime.conf")
        shutil.copy2(config_dir / "experiment.conf", temporary / "experiment.conf")
        overlay_path = temporary / "baseline.conf"
        overlay = configparser.ConfigParser(interpolation=None)
        overlay.optionxform = str
        overlay.read(config_dir / "baseline.conf")
        overlay.add_section("index")
        overlay["index"]["R"] = "64"
        with overlay_path.open("w", encoding="utf-8") as stream:
            overlay.write(stream)
        try:
            load_variant(overlay_path)
        except ExperimentError as error:
            assert "must remain in common.conf" in str(error)
        else:
            raise AssertionError("a non-research override escaped common.conf")

    metadata = {
        "dim": 128,
        "hot_graph_dynamic_record_bytes": 1040,
        "dynamic_node_base_offsets": [
            20_444_099_040, 21_095_426_384, 19_882_535_296,
            21_095_426_384, 19_882_535_296,
        ],
        "hot_graph_entry_counts": [
            19_964_936, 20_600_998, 19_416_534, 20_600_998, 19_416_534,
        ],
    }
    assert required_mn_memory_gib(metadata) == 24
    print("experiment config contract: PASS")


if __name__ == "__main__":
    main()
