# Predictor-Corrector Interior Point Method for QP

A MATLAB project for MIE1621 (Convex Optimization), Winter 2025. I implemented a predictor-corrector primal-dual interior point method from scratch to solve quadratic programs, and compared it against MATLAB's built-in `quadprog`.

## The problem

The method solves QPs of the form:

```
minimize    (1/2) x'Qx + c'x
subject to  Ax = b, x >= 0
```

I derived the algorithm by applying Newton's method to the barrier KKT conditions, then reduced the resulting linear system (eliminating the dual slack and dual variables) so it's cheap to solve even for larger problems. Each iteration does a predictor step (ignoring the centering term) followed by a corrector step (adding it back in, with a centering parameter picked from how well the predictor step did).

Instead of solving one big sparse KKT system directly, I reduce it down to a smaller system in the dual variable, which is much faster.

## What's tested

**Part 1 — Financial optimization (3 assets).** A classic mean-variance portfolio problem: maximize expected return minus a risk penalty, with weights summing to 1 and non-negative. Solved with my solver and with `quadprog` — both land on the same optimal weights (about 39% / 17% / 44%) and the same objective value.

**Part 2 — Large random QPs.** Randomly generated positive-definite covariance matrices at sizes n = 5, 10, 20, 100, and 10000, solved both ways and compared on objective value and runtime. Objective values match quadprog closely at every size. My solver is a bit slower on the largest case (about 69s vs 13s for quadprog at n = 10000), which makes sense since quadprog is a mature, heavily optimized implementation, but it still gets there.

## How to run it

Open `MIE1621_Project_Hamid_Nakhaei.m` in MATLAB and hit Run. It takes about a minute. It runs Part 1 first, then Part 2, printing iteration tables and comparison results as it goes. Requires the Optimization Toolbox (for `quadprog`).

## Status

Course project, not under active development. The report has the full math if you want to see how the reduced system is derived.
