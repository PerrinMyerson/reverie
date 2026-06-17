# Jana vs Reverie Benchmark Summary

Generated at `2026-06-12T02:14:24.254891+00:00` with `5` runs, `1` warmup, min speedup 1.25x.

Workloads: `41`
Speedup range: `2.40x` to `10.93x` (minimum required `2.00x`)
Median speedup: `3.79x` (required `3.00x`)
Geometric mean speedup: `3.93x` (required `3.00x`)
Median peak RSS ratio (Jana/Reverie): `3.45x`
Directions: `19` forward, `12` reverse, `10` roundtrip
Weakest workload: `jana_matrixmult_v1_roundtrip` at `2.40x`
Strongest workload: `janus_schroedinger_n16_t100_roundtrip` at `10.93x`
Gate: `PASS`

| workload | Jana median | Jana peak RSS | Reverie median | Reverie peak RSS | speedup | RSS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `jana_fib_recursive_direct` | 17.760 ms | 13.95 MiB | 4.668 ms | 3.95 MiB | 3.80x | 3.53x |
| `jana_sqrt_direct` | 18.416 ms | 14.00 MiB | 4.743 ms | 4.12 MiB | 3.88x | 3.39x |
| `jana_stack_operations_direct` | 17.655 ms | 13.95 MiB | 4.640 ms | 4.12 MiB | 3.81x | 3.38x |
| `janus_stack_reverse_cleanup` | 17.580 ms | 13.98 MiB | 4.691 ms | 4.23 MiB | 3.75x | 3.30x |
| `bit_reversal_n8` | 18.236 ms | 13.92 MiB | 4.794 ms | 3.98 MiB | 3.80x | 3.49x |
| `bit_reversal_n8_reverse` | 18.711 ms | 13.94 MiB | 4.576 ms | 4.00 MiB | 4.09x | 3.48x |
| `bit_reversal_n8_roundtrip` | 37.472 ms | 13.94 MiB | 9.321 ms | 4.02 MiB | 4.02x | 3.47x |
| `jana_matrixmult_v1_direct` | 18.787 ms | 14.56 MiB | 7.286 ms | 4.97 MiB | 2.58x | 2.93x |
| `jana_matrixmult_v1_reverse` | 18.299 ms | 14.59 MiB | 7.300 ms | 5.14 MiB | 2.51x | 2.84x |
| `jana_matrixmult_v1_roundtrip` | 35.053 ms | 14.58 MiB | 14.575 ms | 5.16 MiB | 2.40x | 2.83x |
| `matrix_transpose_3x3` | 17.968 ms | 13.88 MiB | 6.006 ms | 4.00 MiB | 2.99x | 3.47x |
| `matrix_transpose_3x3_reverse` | 17.300 ms | 13.94 MiB | 5.978 ms | 4.05 MiB | 2.89x | 3.44x |
| `matrix_transpose_3x3_roundtrip` | 36.678 ms | 13.92 MiB | 9.437 ms | 4.03 MiB | 3.89x | 3.45x |
| `jana_factor_direct` | 17.646 ms | 13.95 MiB | 5.969 ms | 4.22 MiB | 2.96x | 3.31x |
| `jana_perm_to_code_direct` | 17.534 ms | 13.92 MiB | 4.723 ms | 4.17 MiB | 3.71x | 3.34x |
| `janus_perm_to_code_reverse` | 18.639 ms | 13.94 MiB | 4.825 ms | 4.08 MiB | 3.86x | 3.42x |
| `janus_perm_to_code_roundtrip` | 36.820 ms | 13.92 MiB | 9.510 ms | 4.09 MiB | 3.87x | 3.40x |
| `jana_run_length_enc_direct` | 17.599 ms | 13.94 MiB | 5.764 ms | 4.16 MiB | 3.05x | 3.35x |
| `jana_run_length_enc_stack_direct` | 17.459 ms | 13.89 MiB | 4.675 ms | 4.14 MiB | 3.73x | 3.35x |
| `rle_compression_n8` | 18.610 ms | 13.95 MiB | 4.907 ms | 4.05 MiB | 3.79x | 3.45x |
| `rle_compression_n8_reverse` | 18.306 ms | 14.02 MiB | 4.868 ms | 4.05 MiB | 3.76x | 3.46x |
| `rle_compression_n8_roundtrip` | 37.707 ms | 14.02 MiB | 9.557 ms | 4.05 MiB | 3.95x | 3.46x |
| `fib_loop_n1000` | 19.583 ms | 14.47 MiB | 4.790 ms | 3.81 MiB | 4.09x | 3.80x |
| `fib_loop_n1000_reverse` | 18.827 ms | 14.47 MiB | 4.698 ms | 3.84 MiB | 4.01x | 3.76x |
| `fib_loop_n1000_roundtrip` | 35.871 ms | 14.48 MiB | 9.492 ms | 3.84 MiB | 3.78x | 3.77x |
| `procedure_call_n1000` | 18.434 ms | 14.48 MiB | 4.715 ms | 3.89 MiB | 3.91x | 3.72x |
| `procedure_call_n1000_reverse` | 17.795 ms | 14.48 MiB | 4.835 ms | 3.91 MiB | 3.68x | 3.71x |
| `procedure_call_n1000_roundtrip` | 35.584 ms | 14.48 MiB | 9.997 ms | 3.91 MiB | 3.56x | 3.71x |
| `janus_root_66` | 17.349 ms | 14.00 MiB | 4.889 ms | 4.05 MiB | 3.55x | 3.46x |
| `janus_root_66_reverse` | 17.149 ms | 13.98 MiB | 5.004 ms | 4.14 MiB | 3.43x | 3.38x |
| `janus_factor_840` | 18.469 ms | 13.95 MiB | 4.822 ms | 3.77 MiB | 3.83x | 3.71x |
| `janus_factor_840_reverse` | 17.696 ms | 14.56 MiB | 4.855 ms | 3.80 MiB | 3.65x | 3.84x |
| `janus_sort_n50_reverse_order` | 40.901 ms | 14.44 MiB | 5.962 ms | 4.23 MiB | 6.86x | 3.41x |
| `janus_sort_n50_reverse_order_reverse` | 42.066 ms | 14.56 MiB | 6.149 ms | 4.34 MiB | 6.84x | 3.35x |
| `janus_sort_n50_reverse_order_roundtrip` | 81.779 ms | 14.55 MiB | 11.545 ms | 4.31 MiB | 7.08x | 3.37x |
| `janus_schroedinger_n16_t100` | 65.708 ms | 17.58 MiB | 7.027 ms | 4.36 MiB | 9.35x | 4.03x |
| `janus_turing_binary_inc` | 17.601 ms | 14.56 MiB | 7.136 ms | 5.12 MiB | 2.47x | 2.84x |
| `janus_turing_binary_inc_reverse` | 18.526 ms | 14.56 MiB | 7.160 ms | 5.14 MiB | 2.59x | 2.83x |
| `janus_turing_binary_inc_roundtrip` | 36.162 ms | 14.58 MiB | 14.353 ms | 5.12 MiB | 2.52x | 2.84x |
| `janus_schroedinger_n16_t100_reverse` | 64.601 ms | 17.59 MiB | 6.002 ms | 4.38 MiB | 10.76x | 4.02x |
| `janus_schroedinger_n16_t100_roundtrip` | 129.529 ms | 17.59 MiB | 11.848 ms | 4.36 MiB | 10.93x | 4.04x |
