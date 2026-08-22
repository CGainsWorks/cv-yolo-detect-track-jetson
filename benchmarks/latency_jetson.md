# Jetson latency and throughput

**Empty on purpose.  No measurement has been taken on a Jetson.**

There is no board.  Every latency figure in this repository came from a desktop
with an RTX 4060 and a Ryzen 7 5700X, and `benchmarks/results.csv` carries a
`device_class` column that reads `desktop` on every row.
`scripts/benchmark_latency.py` decides that column from the presence of
`/etc/nv_tegra_release` rather than from a flag, so it cannot be set wrongly,
and CI fails if a published results file ever claims `jetson`.

This file exists so the gap is a visible hole rather than a missing section.

## The table this will become

| variant | precision | power mode | inference p50 | inference p95 | end to end p50 | FPS | mAP@0.5:0.95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tensorrt-fp32` | fp32 | MAXN | | | | | |
| `tensorrt-fp16` | fp16 | MAXN | | | | | |
| `tensorrt-int8` | int8 | MAXN | | | | | |
| `tensorrt-fp32` | fp32 | 15 W | | | | | |
| `tensorrt-fp16` | fp16 | 15 W | | | | | |
| `tensorrt-int8` | int8 | 15 W | | | | | |

Two power modes rather than one, because a Jetson at 7 W and the same board at
MAXN differ by more than the difference between two precisions.  A single row
would invite the reader to attribute a power-budget effect to a quantisation
choice.

## Filling it in

```bash
sh scripts/bench_jetson.sh
```

That builds the three engines on the board, sweeps the power modes with the
clocks pinned and writes `results/latency_jetson_mode*.json`.  Copy those into
`results/published/` and run `make docs-fix`.

`docs/jetson.md` has the full procedure, the version matrix for Jetpack 5 and
6, and an honest account of which conclusions in `docs/results.md` would
survive the measurement and which would not.

## What to expect, stated in advance

Writing the prediction down before the measurement is the only way it can be
wrong in public.

* **TensorRT will beat eager PyTorch by much more than the precision changes
  do.**  `results/published/launch_bound.json` shows this network is kernel
  launch bound at batch 1 on a desktop GPU: eight times the work costs 1.237
  times the wall clock.  Layer fusion and CUDA graph replay attack exactly that,
  so the FP32 engine should be several times faster than the 8.93 ms PyTorch
  row despite the Orin Nano being far smaller silicon.
* **INT8 should finally show a real speedup.**  On the CPU it bought 1.03 times
  because the bottleneck was per-operator overhead.  On tensor cores with a
  fused engine the arithmetic is the bottleneck again and the usual figure is
  between 1.5 and 2.5 times over FP16.
* **The accuracy column should not move much.**  mAP is a property of the
  arithmetic and TensorRT INT8 is a different calibrator rather than a
  different idea, so a drop within about a point of the 0.01005 measured here
  would be unsurprising.  A much larger drop would point at the detection head
  being quantised, which is the failure documented in `docs/results.md`.
* **The tracker may stop being free.**  It is 3 percent of the frame budget on
  a desktop.  On a board perhaps a third the speed, with the appearance encoder
  running once per detection rather than once per frame, that share could grow
  enough to matter.  The benchmark reports the stages separately so the answer
  will be readable rather than inferred.
