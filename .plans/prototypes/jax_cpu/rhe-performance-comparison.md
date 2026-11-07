# Kodama RHE runtime comparison

Measured 2026-11-06 through the installed CLI implementation, using all 35 real 1KG blocks (4,245,055 variants, 3,202 diploid individuals). Four CPU workers, default XNysTrace estimator with 100 probes, normal sampler, seed 42, intercept plus one covariate. Phenotypes are synthetic independent normal noise plus 0.3 times the covariate; this is a runtime/parity test, not heritability recovery.

Three fresh processes per backend and phenotype count, interleaved in alternating order. OS file cache was not flushed; persistent JAX compilation caching was disabled. Process runtime includes imports, loading, estimation, output and cleanup. RHE phase includes first-use JIT and deferred Cython worker readiness; it is not a warm-only measurement.

| Phenotypes | Cython total s | JAX total s | Total speedup | Cython RHE phase s | JAX RHE phase s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 17.93 | 14.34 | 1.25x | 16.88 | 7.83 |
| 8 | 20.28 | 15.11 | 1.34x | 19.25 | 8.59 |

The RHE phase is about 2.2x faster, but end-to-end speedup is only 1.25–1.34x because the current JAX loading/setup path costs more. Trial timings, output estimates, seeds and conditions are retained in `rhe-cli-comparison.json`; reproduce with `benchmark_rhe_cli.py`.

## Correctness limitation discovered

Variance-component and heritability point estimates agreed (all s2g/s2e/h2g checks passed rtol=2e-4, atol=2e-4; largest absolute h2g difference below 6.1e-8). Standard errors did not agree: JAX returned s2g.se=0 and h2g.se=NaN, while the Cython one-trait reference returned s2g.se=68.37 and h2g.se=0.02135. This recurred across repetitions and trait counts.

The existing JAX RHE uncertainty formula squares a large unnormalized-GRM denominator in float32. An independent dense 10x10 reproduction, with no native graph traversal, produced a finite denominator of 8.25e20 but an infinite square in float32, yielding zero genetic variance. Running that same algebra in float64 produced denominator squared 6.80625e41 and finite nonzero uncertainty. See `diagnose_rhe_precision.py` and `rhe-precision-diagnostic.json`.

This is an unresolved RHE uncertainty-calculation issue; timing improvements do not establish a correct drop-in RHE replacement. Production code was not changed during this performance comparison. A follow-up should stabilize/promote uncertainty calculations and add a regression using a realistically scaled GRM before repeating end-to-end parity and timing checks.
