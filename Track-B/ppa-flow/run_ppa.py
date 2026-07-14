#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


FLOW_ROOT = Path(__file__).resolve().parent
STDCELL_LOCK = FLOW_ROOT / "ASAP7_STDCELL_LOCK.sha256"
SRAM_LOCK = FLOW_ROOT.parent / "sram" / "ASAP7_SRAM_LOCK.sha256"
SRAM_AREAS = {
    "srambank_64x4x32_6t122": 415.2384,
    "srambank_128x4x32_6t122": 691.2,
    "srambank_256x4x32_6t122": 1311.0336,
    "srambank_64x4x64_6t122": 747.42912,
}
LIB_NAMES = {
    "ao": "asap7sc7p5t_AO_RVT_TT_nldm_211120.lib",
    "invbuf": "asap7sc7p5t_INVBUF_RVT_TT_nldm_220122.lib",
    "oa": "asap7sc7p5t_OA_RVT_TT_nldm_211120.lib",
    "seq": "asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib",
    "simple": "asap7sc7p5t_SIMPLE_RVT_TT_nldm_211120.lib",
}
OFFICIAL_DEFAULTS = {
    "PPA_ABC_DELAY_PS": "1000",
    "PPA_UNCERTAINTY_RATIO": "0.05",
    "PPA_IO_DELAY_RATIO": "0.10",
    "PPA_INPUT_DELAY_MIN_NS": "0.050",
    "PPA_INPUT_TRANSITION_NS": "0.010",
    "PPA_OUTPUT_LOAD_FF": "1.0",
    "PPA_MIN_PERIOD_NS": "0.050",
    "PPA_MAX_PERIOD_NS": "1000.000",
    "PPA_PERIOD_STEP_NS": "0.001",
    "PPA_RESET_STYLE": "async",
}
AEC_EVAL_TOP_PORTS = {
    "clk": ("input", 1),
    "rst_n": ("input", 1),
    "load_valid": ("input", 1),
    "load_ready": ("output", 1),
    "load_target": ("input", 3),
    "load_addr": ("input", 32),
    "load_data": ("input", 128),
    "load_strb": ("input", 16),
    "launch_valid": ("input", 1),
    "launch_ready": ("output", 1),
    "grid_x": ("input", 32),
    "grid_y": ("input", 32),
    "grid_z": ("input", 32),
    "block_x": ("input", 32),
    "block_y": ("input", 32),
    "block_z": ("input", 32),
    "program_instructions": ("input", 32),
    "result_valid": ("output", 1),
    "result_ready": ("input", 1),
    "result_status": ("output", 3),
    "result_cycles": ("output", 64),
    "read_valid": ("input", 1),
    "read_ready": ("output", 1),
    "read_addr": ("input", 32),
    "read_data_valid": ("output", 1),
    "read_data": ("output", 128),
    "mem_req_valid": ("output", 1),
    "mem_req_ready": ("input", 1),
    "mem_req_write": ("output", 1),
    "mem_req_space": ("output", 1),
    "mem_req_addr": ("output", 32),
    "mem_req_wdata": ("output", 1024),
    "mem_req_wstrb": ("output", 128),
    "mem_req_tag": ("output", 4),
    "mem_rsp_valid": ("input", 1),
    "mem_rsp_ready": ("output", 1),
    "mem_rsp_rdata": ("input", 1024),
    "mem_rsp_tag": ("input", 4),
    "mem_rsp_error": ("input", 1),
}


class FlowError(RuntimeError):
    pass


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise FlowError(f"missing environment variable {name}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lock(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        digest, rel = raw.split(maxsplit=1)
        records[rel.strip()] = digest
    return records


def verify_lock(root: Path, lock: Path) -> list[Path]:
    verified = []
    for rel, expected in read_lock(lock).items():
        path = root / rel
        if not path.is_file():
            raise FlowError(f"locked file missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise FlowError(f"checksum mismatch: {path}\nexpected {expected}\nactual   {actual}")
        verified.append(path)
    return verified


def run(command: list[str], log: Path, *, cwd: Path, extra_env: dict[str, str]) -> str:
    process_env = os.environ.copy()
    process_env.update(extra_env)
    completed = subprocess.run(command, cwd=cwd, env=process_env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise FlowError(f"command failed ({completed.returncode}): {' '.join(command)}\nlog: {log}")
    return completed.stdout


def tool_identity(path: Path, args: list[str]) -> dict[str, str]:
    completed = subprocess.run([str(path), *args], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise FlowError(f"cannot query tool identity: {path}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "version": lines[0] if lines else "unknown",
    }


def resolve_executable(value: str, name: str) -> Path:
    if "/" in value:
        path = Path(value).expanduser().resolve()
    else:
        found = shutil.which(value)
        if found is None:
            raise FlowError(f"{name} executable unavailable on PATH: {value}")
        path = Path(found).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FlowError(f"{name} executable unavailable: {path}")
    return path


def require_tool_revision(identity: dict[str, str], marker: str, name: str) -> None:
    if marker not in identity["version"]:
        raise FlowError(
            f"unsupported {name} revision; expected marker {marker!r}, "
            f"got {identity['version']!r}")


def parse_slack(text: str, kind: str) -> float:
    pattern = rf"worst slack\s+{re.escape(kind)}\s+(-?[0-9]+(?:\.[0-9]+)?)\s*$"
    matches = re.findall(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
    if not matches:
        raise FlowError(f"OpenSTA output did not contain {kind} worst slack")
    return float(matches[-1])


def validate_check_setup(text: str) -> None:
    match = re.search(
        r"PPA_CHECK_SETUP_BEGIN\s*(.*?)\s*PPA_CHECK_SETUP_END",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise FlowError("OpenSTA check_setup markers missing")
    report = match.group(1).strip()
    if report:
        raise FlowError(f"OpenSTA check_setup reported constraint problems:\n{report}")


def liberty_cell_areas(paths: list[Path]) -> dict[str, float]:
    areas: dict[str, float] = {}
    pattern = re.compile(
        r"\bcell\s*\(\s*([^\s)]+)\s*\)\s*\{.*?\barea\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*;",
        flags=re.DOTALL,
    )
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, value in pattern.findall(text):
            area = float(value)
            if name in areas and not math.isclose(areas[name], area):
                raise FlowError(f"conflicting Liberty area for cell {name}")
            areas[name] = area
    if not areas:
        raise FlowError("no standard-cell areas found in locked Liberty files")
    return areas


def liberty_cell_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    for match in re.finditer(r"(?m)^\s*cell\s*\(", text):
        start = match.start()
        opening = text.find("{", match.end())
        if opening < 0:
            raise FlowError("malformed Liberty cell declaration")
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(text[start:index + 1])
                    break
        else:
            raise FlowError("unterminated Liberty cell declaration")
    return blocks


def build_abc_liberty(base: Path, additions: list[Path], output: Path) -> None:
    text = base.read_text(encoding="utf-8", errors="replace")
    closing = text.rfind("}")
    if closing < 0:
        raise FlowError(f"malformed base Liberty: {base}")
    extra_blocks: list[str] = []
    for path in additions:
        blocks = liberty_cell_blocks(path.read_text(encoding="utf-8", errors="replace"))
        if not blocks:
            raise FlowError(f"no cells found in Liberty addition: {path}")
        extra_blocks.extend(blocks)
    merged = text[:closing] + "\n\n/* Locked cells merged for ABC mapping. */\n"
    merged += "\n\n".join(extra_blocks) + "\n" + text[closing:]
    output.write_text(merged, encoding="utf-8")


def mapped_inventory(netlist_json: Path, top: str,
                     cell_areas: dict[str, float]) -> tuple[dict[str, int], float, float]:
    payload = json.loads(netlist_json.read_text(encoding="utf-8"))
    modules = payload.get("modules", {})
    module = modules.get(top)
    if not isinstance(module, dict):
        raise FlowError(f"top module {top!r} missing from {netlist_json}")
    counts = {name: 0 for name in SRAM_AREAS}
    standard_area = 0.0

    def visit(module_name: str, active: frozenset[str]) -> None:
        nonlocal standard_area
        if module_name in active:
            raise FlowError(f"recursive module hierarchy at {module_name}")
        current = modules.get(module_name)
        if not isinstance(current, dict):
            raise FlowError(f"module {module_name!r} missing from mapped JSON")
        for cell in current.get("cells", {}).values():
            cell_type = cell.get("type")
            if cell_type in counts:
                counts[cell_type] += 1
            elif cell_type in cell_areas:
                standard_area += cell_areas[cell_type]
            elif cell_type in modules:
                visit(cell_type, active | {module_name})
            elif isinstance(cell_type, str) and cell_type.startswith("$"):
                raise FlowError(f"unmapped internal cell remains in netlist: {cell_type}")
            else:
                raise FlowError(f"cell has no locked area definition: {cell_type}")

    visit(top, frozenset())
    area = sum(counts[name] * SRAM_AREAS[name] for name in counts)
    return counts, standard_area, area


def validate_eval_top_ports(netlist_json: Path, top: str) -> None:
    if top != "aec_eval_top":
        return
    payload = json.loads(netlist_json.read_text(encoding="utf-8"))
    module = payload.get("modules", {}).get(top)
    if not isinstance(module, dict):
        raise FlowError(f"top module {top!r} missing from {netlist_json}")
    ports = module.get("ports", {})
    actual = {
        name: (entry.get("direction"), len(entry.get("bits", [])))
        for name, entry in ports.items()
    }
    missing = sorted(set(AEC_EVAL_TOP_PORTS) - set(actual))
    extra = sorted(set(actual) - set(AEC_EVAL_TOP_PORTS))
    mismatched = sorted(
        name for name in set(actual) & set(AEC_EVAL_TOP_PORTS)
        if actual[name] != AEC_EVAL_TOP_PORTS[name]
    )
    if missing or extra or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        for name in mismatched:
            details.append(
                f"{name}: expected={AEC_EVAL_TOP_PORTS[name]}, actual={actual[name]}")
        raise FlowError("aec_eval_top interface does not match spec plus ERRATA E-001: "
                        + "; ".join(details))


def main() -> int:
    parser = argparse.ArgumentParser(description="AEC ASAP7 synthesis, STA, and area flow")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filelist", type=Path, default=Path(env("PPA_FILELIST")))
    parser.add_argument("--top", default=env("PPA_TOP"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    yosys = resolve_executable(env("YOSYS"), "Yosys")
    opensta = resolve_executable(env("OPENSTA"), "OpenSTA")
    lib_dir = Path(env("ASAP7_LIB_DIR"))
    yosys_identity = tool_identity(yosys, ["-V"])
    opensta_identity = tool_identity(opensta, ["-no_init", "-exit"])
    require_tool_revision(yosys_identity, "78e05dfb0", "Yosys")
    require_tool_revision(opensta_identity, "66c29303", "OpenSTA")
    std_libs = {key: lib_dir / name for key, name in LIB_NAMES.items()}
    verified_std = verify_lock(lib_dir, STDCELL_LOCK)

    sram_root_value = os.environ.get("ASAP7_SRAM_ROOT", "")
    sram_libs: list[Path] = []
    sram_models: list[Path] = []
    verified_sram: list[Path] = []
    if sram_root_value:
        sram_root = Path(sram_root_value)
        verified_sram = verify_lock(sram_root, SRAM_LOCK)
        sram_libs = [sram_root / "generated" / "LIB" / f"{name}.lib" for name in SRAM_AREAS]
        sram_models = [sram_root / "generated" / "verilog" / f"{name}.v" for name in SRAM_AREAS]

    if args.check_only:
        print(json.dumps({
            "yosys": str(yosys),
            "opensta": str(opensta),
            "stdcell_libraries": [str(path) for path in std_libs.values()],
            "sram_libraries": [str(path) for path in sram_libs],
        }, indent=2))
        return 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_filelist = output / "resolved-files.f"
    original = args.filelist.resolve()
    base = original.parent
    lines = []
    for raw in original.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("+incdir+"):
            lines.append("+incdir+" + str((base / line[8:]).resolve()))
        else:
            source = (base / line).resolve()
            if not source.is_file():
                raise FlowError(f"RTL source missing: {source}")
            lines.append(str(source))
    if not lines:
        raise FlowError(f"filelist contains no RTL sources: {original}")
    resolved_filelist.write_text("\n".join(lines) + "\n", encoding="utf-8")

    netlist = output / "mapped.v"
    netlist_json = output / "mapped.json"
    synth_stat = output / "synth.stat"
    abc_liberty = output / "abc-map.lib"
    build_abc_liberty(std_libs["simple"], [std_libs["invbuf"]], abc_liberty)
    common = {
        "PPA_TOP": args.top,
        "PPA_FILELIST_ABS": str(resolved_filelist),
        "PPA_MAP_LIB": str(abc_liberty),
        "PPA_COMB_LIBS": ":".join(map(str, [std_libs["simple"], std_libs["invbuf"]])),
        "PPA_SEQ_LIB": str(std_libs["seq"]),
        "PPA_SRAM_MODELS": ":".join(map(str, sram_models)),
        "PPA_NETLIST": str(netlist),
        "PPA_NETLIST_JSON": str(netlist_json),
        "PPA_SYNTH_STAT": str(synth_stat),
        "PPA_ABC_DELAY_PS": env("PPA_ABC_DELAY_PS", "1000"),
    }
    run([str(yosys), "-c", str(FLOW_ROOT / "synth.tcl")], output / "yosys.log",
        cwd=FLOW_ROOT, extra_env=common)
    netlist_hash = sha256(netlist)
    validate_eval_top_ports(netlist_json, args.top)
    cell_areas = liberty_cell_areas(list(std_libs.values()))
    counts, standard_area, sram_area = mapped_inventory(
        netlist_json, args.top, cell_areas)

    sta_libs = list(std_libs.values()) + sram_libs
    sta_common = common | {
        "PPA_STA_LIBS": ":".join(map(str, sta_libs)),
        "PPA_SDC": str(FLOW_ROOT / "constraints.sdc.tcl"),
        "PPA_CLOCK_PORT": env("PPA_CLOCK_PORT", "clk"),
        "PPA_RESET_PORT": env("PPA_RESET_PORT", "rst_n"),
        "PPA_RESET_STYLE": env("PPA_RESET_STYLE", "async"),
        "PPA_UNCERTAINTY_RATIO": env("PPA_UNCERTAINTY_RATIO", "0.05"),
        "PPA_IO_DELAY_RATIO": env("PPA_IO_DELAY_RATIO", "0.10"),
        "PPA_INPUT_DELAY_MIN_NS": env("PPA_INPUT_DELAY_MIN_NS", "0.050"),
        "PPA_INPUT_TRANSITION_NS": env("PPA_INPUT_TRANSITION_NS", "0.010"),
        "PPA_OUTPUT_LOAD_FF": env("PPA_OUTPUT_LOAD_FF", "1.0"),
    }

    sta_runs = 0

    def sta(period: float, label: str) -> tuple[float, float, str]:
        nonlocal sta_runs
        sta_runs += 1
        text = run([str(opensta), "-no_init", "-exit", str(FLOW_ROOT / "sta.tcl")],
                   output / f"sta-{label}.log", cwd=FLOW_ROOT,
                   extra_env=sta_common | {"PPA_CLOCK_PERIOD_NS": f"{period:.9f}"})
        validate_check_setup(text)
        return parse_slack(text, "max"), parse_slack(text, "min"), text

    low = float(env("PPA_MIN_PERIOD_NS", "0.050"))
    high = float(env("PPA_MAX_PERIOD_NS", "1000.000"))
    step = float(env("PPA_PERIOD_STEP_NS", "0.001"))
    if not (0.0 < low < high and step > 0.0):
        raise FlowError("invalid period search bounds")
    high_setup, _, _ = sta(high, "upper-bound")
    if high_setup < 0.0:
        raise FlowError(f"maximum search period still fails setup: {high} ns, WNS {high_setup} ns")
    low_setup, _, _ = sta(low, "lower-bound")
    if low_setup >= 0.0:
        best = low
    else:
        while high - low > step:
            middle = (low + high) / 2.0
            setup, _, _ = sta(middle, f"search-{sta_runs:02d}")
            if setup >= 0.0:
                high = middle
            else:
                low = middle
        best = math.ceil(high / step) * step

    setup_wns, hold_wns, final_text = sta(best, "final")
    if setup_wns < 0.0:
        best += step
        setup_wns, hold_wns, final_text = sta(best, "final-adjusted")
    if setup_wns < 0.0 or hold_wns < 0.0:
        raise FlowError(f"final timing failed: setup WNS={setup_wns} ns, hold WNS={hold_wns} ns")
    if re.search(r"unconstrained endpoint", final_text, flags=re.IGNORECASE):
        raise FlowError("OpenSTA reported unconstrained endpoints")

    result = {
        "schema_version": 1,
        "status": "pass",
        "flow_scope": "synthesis_sta_area",
        "interface_contract": ("spec+ERRATA-E-001" if args.top == "aec_eval_top"
                               else "nonstandard-smoke-top"),
        "top": args.top,
        "mapped_netlist_sha256": netlist_hash,
        "resolved_filelist_sha256": sha256(resolved_filelist),
        "abc_map_liberty_sha256": sha256(abc_liberty),
        "abc_delay_ps": int(env("PPA_ABC_DELAY_PS", "1000")),
        "minimum_passing_period_ns": best,
        "fmax_mhz": 1000.0 / best,
        "setup_wns_ns": setup_wns,
        "hold_wns_ns": hold_wns,
        "standard_cell_area_um2": standard_area,
        "sram_area_um2": sram_area,
        "total_area_um2": standard_area + sram_area,
        "sram_instances": counts,
        "sta_runs": sta_runs,
        "power": {"status": "not_implemented"},
        "official_defaults": all(env(name, default) == default
                                 for name, default in OFFICIAL_DEFAULTS.items()),
        "configuration": {name: env(name, default)
                          for name, default in OFFICIAL_DEFAULTS.items()},
        "tools": {
            "yosys": yosys_identity,
            "opensta": opensta_identity,
        },
        "libraries": {
            "stdcell_lock_sha256": sha256(STDCELL_LOCK),
            "sram_lock_sha256": sha256(SRAM_LOCK) if sram_root_value else None,
            "files": {str(path): sha256(path)
                      for path in verified_std + verified_sram},
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FlowError, OSError, ValueError) as exc:
        print(f"ppa-flow: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
