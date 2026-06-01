# Jana vs Reverie Benchmark Summary

Generated at `2026-05-30T06:48:19.690441+00:00` with `5` runs, `1` warmup, min speedup 1.25x.

Workloads: `41`
Speedup range: `3.62x` to `99.43x` (minimum required `2.00x`)
Median speedup: `5.41x` (required `4.00x`)
Geometric mean speedup: `8.97x` (required `4.00x`)
Directions: `19` forward, `12` reverse, `10` roundtrip
Weakest workload: `janus_factor_840` at `3.62x`
Strongest workload: `janus_schroedinger_n16_t100_reverse` at `99.43x`
Gate: `PASS`

| workload | Jana median | Reverie median | speedup |
| --- | ---: | ---: | ---: |
| `jana_fib_recursive_direct` | 18.120 ms | 3.222 ms | 5.62x |
| `jana_sqrt_direct` | 18.145 ms | 3.664 ms | 4.95x |
| `jana_stack_operations_direct` | 18.686 ms | 3.413 ms | 5.48x |
| `janus_stack_reverse_cleanup` | 20.006 ms | 3.819 ms | 5.24x |
| `bit_reversal_n8` | 17.769 ms | 3.053 ms | 5.82x |
| `bit_reversal_n8_reverse` | 21.283 ms | 4.073 ms | 5.23x |
| `bit_reversal_n8_roundtrip` | 36.594 ms | 7.223 ms | 5.07x |
| `jana_matrixmult_v1_direct` | 26.340 ms | 6.053 ms | 4.35x |
| `jana_matrixmult_v1_reverse` | 26.871 ms | 6.167 ms | 4.36x |
| `jana_matrixmult_v1_roundtrip` | 53.652 ms | 11.908 ms | 4.51x |
| `matrix_transpose_3x3` | 18.915 ms | 3.368 ms | 5.62x |
| `matrix_transpose_3x3_reverse` | 19.147 ms | 4.089 ms | 4.68x |
| `matrix_transpose_3x3_roundtrip` | 37.994 ms | 7.335 ms | 5.18x |
| `jana_factor_direct` | 19.151 ms | 4.139 ms | 4.63x |
| `jana_perm_to_code_direct` | 18.486 ms | 3.413 ms | 5.42x |
| `janus_perm_to_code_reverse` | 17.343 ms | 2.884 ms | 6.01x |
| `janus_perm_to_code_roundtrip` | 29.738 ms | 5.132 ms | 5.79x |
| `jana_run_length_enc_direct` | 16.162 ms | 3.080 ms | 5.25x |
| `jana_run_length_enc_stack_direct` | 14.245 ms | 2.673 ms | 5.33x |
| `rle_compression_n8` | 14.506 ms | 2.502 ms | 5.80x |
| `rle_compression_n8_reverse` | 15.806 ms | 2.922 ms | 5.41x |
| `rle_compression_n8_roundtrip` | 30.150 ms | 5.655 ms | 5.33x |
| `fib_loop_n1000` | 44.308 ms | 2.662 ms | 16.64x |
| `fib_loop_n1000_reverse` | 44.068 ms | 3.318 ms | 13.28x |
| `fib_loop_n1000_roundtrip` | 89.760 ms | 5.154 ms | 17.42x |
| `procedure_call_n1000` | 63.646 ms | 3.472 ms | 18.33x |
| `procedure_call_n1000_reverse` | 62.407 ms | 2.649 ms | 23.56x |
| `procedure_call_n1000_roundtrip` | 131.207 ms | 7.006 ms | 18.73x |
| `janus_root_66` | 17.598 ms | 3.469 ms | 5.07x |
| `janus_root_66_reverse` | 18.354 ms | 4.554 ms | 4.03x |
| `janus_factor_840` | 18.743 ms | 5.180 ms | 3.62x |
| `janus_factor_840_reverse` | 16.919 ms | 3.598 ms | 4.70x |
| `janus_sort_n50_reverse_order` | 229.176 ms | 3.540 ms | 64.74x |
| `janus_sort_n50_reverse_order_reverse` | 219.539 ms | 3.260 ms | 67.34x |
| `janus_sort_n50_reverse_order_roundtrip` | 443.016 ms | 6.071 ms | 72.97x |
| `janus_schroedinger_n16_t100` | 443.189 ms | 4.838 ms | 91.60x |
| `janus_turing_binary_inc` | 21.561 ms | 4.847 ms | 4.45x |
| `janus_turing_binary_inc_reverse` | 18.970 ms | 4.869 ms | 3.90x |
| `janus_turing_binary_inc_roundtrip` | 38.595 ms | 9.790 ms | 3.94x |
| `janus_schroedinger_n16_t100_reverse` | 432.332 ms | 4.348 ms | 99.43x |
| `janus_schroedinger_n16_t100_roundtrip` | 876.085 ms | 9.377 ms | 93.43x |
