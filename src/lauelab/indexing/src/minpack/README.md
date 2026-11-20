# MINPACK (from cminpack)

The peak fits in `peaksearch/src/fitToFunction.c` and `fitToGaussian.c` use
the Levenberg-Marquardt least-squares method of MINPACK's `lmder` with
internal variable scaling (mode 1). This folder contains the MINPACK routines it
needs, from the C translation cminpack, and `lmsder.c`, which runs `lmder`
one outer iteration at a time.

## Source

- Project: cminpack, C/C++ Minpack by Manolis Lourakis and Frederic Devernay,
  <https://github.com/devernay/cminpack>
- Release: v1.3.11 (13 September 2024), tag commit
  `17dab75c6160d2ee42a3c95ea55e94738d7e559d`,
  <https://github.com/devernay/cminpack/archive/refs/tags/v1.3.11.tar.gz>
- SHA-256 of that archive:
  `45675fac0a721a1c7600a91a9842fe1ab313069db163538f2923eaeddb0f46de`
- Licence: the MINPACK copyright notice of the University of Chicago (1999),
  in `CopyrightMINPACK.txt`. The cminpack documentation (`docs/index.html` in
  the archive) states that cminpack "is distributed under the original
  Minpack license". Its third condition asks for the acknowledgment "This
  product includes software developed by the University of Chicago, as
  Operator of Argonne National Laboratory." in end-user documentation or in
  the software.

## Files

| File | Origin | Change |
| --- | --- | --- |
| `CopyrightMINPACK.txt` | cminpack 1.3.11 | blank line at the end removed |
| `cminpack.h`, `cminpackP.h`, `minpackP.h` | cminpack 1.3.11 | trailing whitespace removed |
| `dpmpar.c`, `lmpar.c`, `qrfac.c`, `qrsolv.c` | cminpack 1.3.11 | trailing whitespace and the blank lines at the end removed |
| `enorm.c` | cminpack 1.3.11 | the same, and the two range tests in the main loop are marked as unlikely with `__builtin_expect` |
| `lmsder.c` | derived from cminpack 1.3.11 `lmder.c` | restructured, see below |
| `lmsder.h` | new | |

`diff -b -B` between each file of the first three rows and the file of the
same name in the archive prints nothing; for `enorm.c` it prints the two
marked lines. The marks change the order of the machine code, not the
arithmetic: with the branches of the common case laid out in line, the run
time of a fit no longer depends on where the function is placed in memory.
The files compile in cminpack's default configuration: double precision,
without BLAS or LAPACK.

`lmsder.c` keeps the arithmetic of one outer iteration of `lmder` with
`mode = 1` and `factor = 100`: the QR factorization of the Jacobian, the
initial scaling and step bound, `qtf`, the scaled gradient norm, the inner
loop of trial steps with `lmpar`, the actual and predicted reductions, and the
update of the step bound. It differs from `lmder` in what ends an iteration
and the fit:

- `lmsder_iterate()` returns after the first trial step that is accepted (the
  ratio of the actual to the predicted reduction of the sum of squares is at
  least 0.0001), after 10 trial steps that are not (`LMSDER_ENOPROG`), or when a
  tolerance test of `lmder` at machine precision holds (`LMSDER_ETOLF`,
  `LMSDER_ETOLX`, `LMSDER_ETOLG`). A further call continues the trial steps
  with the step bound and Levenberg-Marquardt parameter that the previous
  call left, as the inner loop of `lmder` does; the scaling and the initial
  step bound are set once after `lmsder_set()`.
- It returns at once, without a step, when the residuals are already zero.
- The tests on `ftol`, `xtol`, `gtol` and `maxfev` are not used; the caller
  stops on the step size with `lmsder_test_delta()` and on an iteration count.
- The caller's Jacobian is kept: the QR factorization works on a copy.

`lmsder_alloc()` returns NULL when `p` is 0 or `n < p`. When it cannot allocate the solver, it returns NULL in
`liblaue.so`, whose callers report an out-of-memory status; in the
`peaksearch` program it prints a message and calls `abort()`.
