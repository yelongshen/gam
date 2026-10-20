#!/usr/bin/env python3
"""Add zero-weight height-map inputs to an exported SONIC decoder ONNX (no retraining).

    python widen_decoder_for_height_map.py <model_decoder.onnx> <out_decoder.onnx> [--extra 484]

The decoder takes obs_dict = [token | proprioception] and starts with two Slices of it, a
Concat and a MatMul by a (D, H) weight. This appends `extra` inputs (height_map_flat 363 +
height_map_valid_flat 121 by default) at the END of obs_dict: the slice bounds are rewritten as
absolute indices (exports often write them relative to the end), the slice that reached the old
end is extended to the new one, and the weight gets `extra` zero rows. The result computes exactly the same actions
as the original for any values of the new inputs (checked on random inputs before saving), so a
deployed policy can run with the height-map observations wired in before any model is trained
on them. Pair it with the original encoder and the policy's observation_config.yaml plus the
two height_map entries appended to `observations`.
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper


def widen(model, extra):
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    (inp,) = [i for i in g.input if i.name not in inits]
    dim = inp.type.tensor_type.shape.dim[-1]
    d = dim.dim_value
    consts = {}
    for n in g.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = n
    slices = [n for n in g.node if n.op_type == "Slice" and n.input[0] == inp.name]
    if len(slices) != 2:
        raise RuntimeError(f"expected 2 Slices of {inp.name}, found {len(slices)}")

    def const_val(name):
        if name in inits:
            return numpy_helper.to_array(inits[name])
        return numpy_helper.to_array(consts[name].attribute[0].t)

    def set_const(name, value):
        new = numpy_helper.from_array(np.array([value], dtype=const_val(name).dtype), name=name)
        if name in inits:
            inits[name].CopyFrom(new)
        else:
            consts[name].attribute[0].t.CopyFrom(new)

    # Exports write these bounds relative to the end ("the last 930", "all but the last 930"),
    # which would silently pick up the new inputs. Resolve every bound against the original width
    # and write it back as an absolute index; the slice that reached the old end is extended to
    # the new end (it is the one the zero rows below line up with).
    def absolute(v):
        v = int(v)
        return max(0, d + v) if v < 0 else min(v, d)

    reached_end = 0
    for n in slices:
        start = absolute(const_val(n.input[1]).reshape(-1)[0])
        end = absolute(const_val(n.input[2]).reshape(-1)[0])
        if end == d:
            end = d + extra
            reached_end += 1
        set_const(n.input[1], start)
        set_const(n.input[2], end)
    if reached_end != 1:
        raise RuntimeError(f"expected exactly one slice to reach the end of the input, found {reached_end}")

    # First MatMul whose weight has d input rows.
    mm = [n for n in g.node if n.op_type == "MatMul" and n.input[1] in inits and inits[n.input[1]].dims[0] == d]
    if len(mm) != 1:
        raise RuntimeError(f"expected one MatMul with a ({d}, H) weight, found {len(mm)}")
    w = numpy_helper.to_array(inits[mm[0].input[1]])
    w2 = np.concatenate([w, np.zeros((extra, w.shape[1]), dtype=w.dtype)], axis=0)
    inits[mm[0].input[1]].CopyFrom(numpy_helper.from_array(w2, name=mm[0].input[1]))
    dim.dim_value = d + extra
    return d


def check(orig_path, new_path, d, extra, trials=50):
    import onnxruntime as ort

    a = ort.InferenceSession(orig_path, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(new_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for k in range(trials):
        x = rng.normal(0, 1, (1, d)).astype(np.float32)
        hm = (rng.normal(0, 1, (1, extra)) * (k % 5)).astype(np.float32)
        ya = a.run(None, {a.get_inputs()[0].name: x})[0]
        yb = b.run(None, {b.get_inputs()[0].name: np.concatenate([x, hm], 1)})[0]
        worst = max(worst, float(np.abs(ya - yb).max()))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--extra", type=int, default=363 + 121)
    args = ap.parse_args()
    model = onnx.load(args.src)
    d = widen(model, args.extra)
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    worst = check(args.src, args.dst, d, args.extra)
    print(f"input {d} -> {d + args.extra}; max |action difference| vs original over 50 random inputs "
          f"(random height-map values): {worst:.3g}")
    if worst > 1e-5:
        raise SystemExit("MISMATCH: widened decoder does not reproduce the original")


if __name__ == "__main__":
    main()
