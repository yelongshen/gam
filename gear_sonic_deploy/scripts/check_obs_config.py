#!/usr/bin/env python3
"""Check a deploy observation_config YAML against exported ONNX models.

    python check_obs_config.py <observation_config.yaml> <decoder.onnx> [<encoder.onnx>]

Sums the sizes of the YAML's policy (and encoder) observations, taken from the
runner's observation registry in src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp,
and compares them with the models' input sizes. Unknown names are reported too: the
runner would refuse to start with them. Needs pyyaml and onnx.
"""

import os
import re
import sys

import onnx
import yaml

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "g1", "g1_deploy_onnx_ref",
                      "src", "g1_deploy_onnx_ref.cpp")
# Registry sizes that are expressions rather than literals.
SYMBOLS = {"terrain_scan_.points.size()": 363, "terrain_scan_.valid.size()": 121,
           "terrain_scan::kTorsoGridCells": 441}


def registry(token_dim):
    src = open(RUNNER).read()
    body = src[src.index("std::vector<ObservationRegistry> GetObservationRegistry()"):]
    body = body[:body.index("void InitializeObservationFunctions()")]
    sizes = {}
    for name, dim in re.findall(r'\{"([A-Za-z0-9_]+)",\s*([^,]+),\s*\[this\]', body):
        dim = dim.strip()
        sizes[name] = token_dim if dim == "token_dim" else SYMBOLS.get(dim, None) if not dim.isdigit() else int(dim)
    return sizes


def onnx_input_size(path):
    m = onnx.load(path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    (inp,) = [i for i in m.graph.input if i.name not in inits]
    return inp.type.tensor_type.shape.dim[-1].dim_value


def total(names, sizes, what):
    ok, n = True, 0
    for name in names:
        if sizes.get(name) is None:
            print(f"  {what}: '{name}' is not in the runner's registry (or has a computed size)")
            ok = False
        else:
            n += sizes[name]
    return n, ok


def main(cfg_path, decoder, encoder=None):
    cfg = yaml.safe_load(open(cfg_path))
    enc = cfg.get("encoder") or {}
    sizes = registry(int(enc.get("dimension", 64)))
    good = True
    pol, ok = total([o["name"] for o in cfg["observations"] if o.get("enabled", True)], sizes, "policy")
    want = onnx_input_size(decoder)
    print(f"policy observations: {pol} values; {os.path.basename(decoder)} input: {want}")
    good &= ok and pol == want
    if encoder:
        names = [o["name"] for o in enc.get("encoder_observations", []) if o.get("enabled", True)]
        n, ok = total(names, sizes, "encoder")
        want = onnx_input_size(encoder)
        print(f"encoder observations: {n} values; {os.path.basename(encoder)} input: {want}")
        good &= ok and n == want
        for mode in enc.get("encoder_modes", []):
            missing = [r for r in mode.get("required_observations", []) if r not in names]
            if missing:
                print(f"  mode '{mode['name']}' requires observations not listed: {missing}")
                good = False
    print("OK" if good else "MISMATCH")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
