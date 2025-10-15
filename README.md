# AI & Data Science Roadmap
A weekly, cumulative, and measurable plan to become a data specialist who understands the theory, can explain “why”, and delivers in practice.

What’s new in this version
- Plain-language definitions and mini-examples added wherever terms could be unfamiliar (e.g., normalize_minmax, tokenize_basic, safe_div, idempotent, Frobenius norm, WAIC/LOO, Hausman, etc.).
- Assignment & Pass and Self-check sections rewritten so anyone can follow them step by step.
- Each week states clear outcomes: “By the end of this week, you will be able to…”
- All main sources remain fully completed by the end.

Key Principles
- Order (prerequisites): Setup/Programming → Mathematics (MML) → Probability & Statistics (frequentist) → Bayesian Statistics (Statistical Rethinking) → Econometrics & Causal Inference → Classical ML → Time Series → Deep Learning → MLOps → Data Engineering → LLMs → Capstone.
- Main sources: Authoritative, internationally recognized, and completed (every week has clickable links).
- Weekly structure:
  - Sources (Primary → Alternatives)
  - What you’ll learn
  - Role in ML/AI
  - Time & Load
  - By the end of this week, you will be able to…
  - Key terms explained (plain words + tiny examples if needed)
  - Assignment & Pass (simple steps + numeric acceptance criteria)
  - Self-check (how to run tests, pass = ≥ 80% unless stricter noted)

Gap Week (if you don’t pass)
- Spend 6–10 hours only on the missing concept(s).
- Read the exact sections again + do 2–3 small exercises + re-run tests.
- Write a 1-page note “What I fixed and how”.
- Proceed only after passing.

Testing & Hygiene (every week)
- Keep tests/ at repo root; run tests with: `pytest -q` (or in a Notebook cell: `!pytest -q`).
- Pass threshold: ≥ 80% tests pass (unless a stricter threshold is stated).
- Reproducibility: environment.yml or requirements.txt, fixed random seeds.
- Filenames: `Week-XX-Topic.ipynb` (or .py) and short `Week-XX-RESULTS.md`.

Visual Legend
- Primary = must complete
- Alternatives = optional reinforcement
- ✅ = passed this week
- ⚠️ = add a Gap Week before continuing

---

## PHASE 0 — Setup & Acceleration (2 weeks)

### Week 1 — Python + Git Basics
- Sources  
  - Primary: [Python Tutorial for Beginners (Full Course)](https://www.youtube.com/watch?v=rfscVS0vtbw)  
  - Alternatives: [Kevin Sheppard — Python Introduction Notes (PDF)](https://www.kevinsheppard.com/files/teaching/python/notes/python_introduction_2021.pdf) · [SOGA-PY: Introduction to Python](https://www.geo.fu-berlin.de/en/v/soga-py/Introduction-to-Python/index.html) · [Git & GitHub — Getting Started](https://docs.github.com/en/get-started/using-git)
- What you’ll learn: Python syntax, types, functions, modules; Git flow (branch/PR).
- Role in ML/AI: Development backbone for all analysis/modeling.
- Time & Load: 8–10h
- By the end of this week, you will be able to…  
  - Write simple Python functions, read/write files, and use Git branches and PRs.

- Key terms explained
  - normalize_minmax: Rescale numbers to [0, 1] so the smallest value becomes 0 and the largest becomes 1.
    ```python
    def normalize_minmax(xs):
        lo, hi = min(xs), max(xs)
        return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]
    ```
  - tokenize_basic: Split a sentence into simple word tokens by removing punctuation and splitting on spaces.
    ```python
    import re
    def tokenize_basic(text):
        return re.findall(r"[A-Za-z0-9]+", text)
    # "Hello, world!" -> ["Hello","world","1"] if numbers exist
    ```
  - safe_div: Division that doesn’t crash on division-by-zero (returns None or a message instead).
    ```python
    def safe_div(a, b):
        return a / b if b != 0 else None
    ```

- Assignment & Pass (step-by-step)
  1) Create a GitHub repo “ai-ds-journey”.  
  2) Add `Week-01-Python.ipynb` implementing `normalize_minmax`, `tokenize_basic`, `safe_div`, and simple file I/O (read a small text file; count lines).  
  3) Add `README.md` with environment and run steps.  
  4) Create a branch, open a PR, merge it.  
  Pass: Notebook runs fully; PR merged; README clear.

- Self-check (how to test)
  - Create `tests/test_week01.py` with ≥ 10 tiny tests (mean, normalize, tokenize, unique set, join paths OS-safe, safe_div handles 1/0, strip+lower, sum list, read text line count, sorting).
  - Run `pytest -q`. Pass if ≥ 8 tests are green (≥ 80%). If not, fix code and re-run.

---

### Week 2 — Pandas for IO/Cleaning + Mini EDA
- Sources  
  - Primary: [Python for Data Analysis (online book)](https://wesmckinney.com/book/)  
  - Alternatives: [PFDA (PDF)](https://ix.cs.uoregon.edu/~norris/cis407/books/python_for_data_analysis.pdf) · [Data Science and Analytics with Python (PDF)](https://mathstat.dal.ca/~brown/sound/python/P1-Data_Science_and_Analytics_with_Python_2b29.pdf)
- What you’ll learn: Read CSV/Parquet, handle missing/outliers, reshape/join, basic plotting.
- Role: Data quality and EDA foundation.
- Time & Load: 8–10h
- By the end of this week, you will be able to…  
  - Load, clean, join, and visualize a dataset; write a short findings summary.

- Key terms explained
  - Missing values (NA): Cells with no data; you can drop or impute (fill) them.
  - Outliers: Unusually large/small points; simple rule: IQR (Q3−Q1). Points < Q1−1.5·IQR or > Q3+1.5·IQR are “outliers”.
  - Join/Merge: Combine tables by a common key (e.g., user_id).

- Assignment & Pass
  - Do: `Week-02-EDA.ipynb` on an open dataset; NA/outlier handling, type fixes, joins; ≥ 4 plots (hist, box, scatter, heatmap).  
  - Write `Week-02-EDA.md` with 5 findings + 3 recommendations.  
  - Provide `requirements.txt` or `environment.yml`.  
  - Pass: Reproducible notebook; labeled plots; findings match visuals.

- Self-check
  - `tests/test_week02.py`: missing rate function, IQR outlier count (on a tiny fixture), merge shape correctness.  
  - Run `pytest -q`; pass if ≥ 80%.

---

## PHASE 1 — Mathematics (MML) with soft transitions (11 weeks) — complete MML
Primary: [Mathematics for Machine Learning (MML)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)

### Week 3 — Linear Algebra I: Vectors, Matrices, Linear Maps
- Sources: Primary: [MML Ch.2](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) · Alt: [MIT 18.06 L1–L3](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) · [3Blue1Brown (LA intro)](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- What you’ll learn: Matrix/vector ops, rank, linear transforms.
- Role: Algebraic foundation for features/parameters.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Compute with matrices/vectors and visualize simple 2D transforms.

- Key terms explained
  - Rank: How many independent columns (or rows) a matrix has—capacity to transform space.
  - Linear map: A function like y = A x that preserves addition and scalar multiplication.

- Assignment & Pass  
  - Do: `Week-03-LA.ipynb` — compare `matmul/solve/pinv`; visualize rotations/scaling.  
  - Pass: Relative solve error < 1e−6 on 100 random systems; mini-quiz ≥ 80%.

- Self-check: `tests/test_week03.py` (rank, rotation orthonormality, identity solve). Run `pytest -q`.

---

### Week 4 — Analytic Geometry: Norms, Projections (Idempotency)
- Sources: Primary: [MML Ch.3](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) · Alt: [3Blue1Brown (projections)](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- What you’ll learn: L1/L2/∞ norms, projections, orthogonality.
- Role: Geometric least squares and regularization intuition.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Choose appropriate norms and verify projection behavior programmatically.

- Key terms explained
  - Norms: Ways to measure vector size. L1 = sum |xᵢ|; L2 = sqrt(sum xᵢ²); L∞ = max |xᵢ|.
  - Projection matrix (P): Maps any vector to the “closest point” in a subspace (like dropping a shadow on a plane).
  - Idempotent (for P): Applying P twice is same as once (P·P = P).
  - Frobenius norm: Size of a matrix like a vector of all its entries: `np.linalg.norm(M, 'fro')`.

- Assignment & Pass
  - Do: Visualize k-NN boundaries with L1/L2/∞ and explain differences.  
  - Do: Projection matrix idempotency test: pass if `np.linalg.norm(P @ P - P, ord='fro') < 1e-8`.  
  - Pass: 1–2 page “which norm when, and why?” note + idempotency test passes.

- Self-check: `tests/test_week04.py` (idempotency; ||x||∞ ≤ ||x||2 ≤ ||x||1 by examples). Run `pytest -q`.

---

### Week 5 — Matrix Decompositions I: QR, Orthogonality, SVD (intuition)
- Sources: Primary: [MML Ch.4 (first half)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) · Alt: [Gantmacher (PDF)](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/gantmacher1.pdf)
- What you’ll learn: QR factorization, orthogonal matrices, SVD intuition.
- Role: Numerical stability, high-quality least squares.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Solve least squares with QR and check SVD’s orthonormal factors.

- Key terms explained
  - QR: A = Q·R with Q orthonormal (columns ⟂ and unit length), R upper-triangular—great for stable solves.
  - SVD: A = U Σ Vᵀ; like rotating, scaling, rotating—reveals directions of most variance.

- Assignment & Pass  
  - Do: QR-based LS vs `np.linalg.lstsq`; L2 difference < 1e−8. Verify U/V orthonormality.  
  - Pass: All checks pass; code is clean and commented.

- Self-check: `tests/test_week05.py`.

---

### Week 6 — Matrix Decompositions II: SVD→PCA, Low-Rank Approximation
- Sources: Primary: [MML Ch.4 (PCA section)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) · Alt: [3Blue1Brown (PCA)](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- What you’ll learn: PCA from SVD, reconstruction error, elbow.
- Role: Dimensionality reduction and denoising.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Implement PCA from scratch and justify chosen components.

- Key terms explained
  - Reconstruction error: How much detail you lose when you keep only top-k components.
  - Elbow method: Choose k at the “bend” of error curve (diminishing returns after that point).

- Assignment & Pass  
  - Do: PCA from scratch + sklearn; per-component explained variance diff < 1e−3; error vs rank plot + elbow justification.  
  - Pass: Matching results + clear elbow rationale.

- Self-check: `tests/test_week06.py`.

---

### Week 7 — Vector Calculus I: Multivariate Derivatives & Chain Rule
- Sources: Primary: [MML Ch.5 (basics)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) · Alt: [3Blue1Brown (Calculus 1–12)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNPOjrT6KVlfJu6NsY6v3v)
- What you’ll learn: Gradient, Jacobian, Hessian; chain rule.
- Role: Math behind backprop and optimization.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Derive gradients and verify them numerically.

- Key terms explained
  - Gradient: Vector of partial derivatives—direction of steepest ascent.
  - Jacobian: Matrix of first derivatives for vector-valued functions.
  - Hessian: Matrix of second derivatives—curvature.

- Assignment & Pass  
  - Do: Gradient checks (analytical vs finite differences); relative error < 1e−5.  
  - Pass: All functions meet the threshold.

- Self-check: `tests/test_week07.py`.

---

### Week 8 — Vector Calculus II: Bridge to Autodiff
- Sources: Primary: [MML Ch.5 (advanced)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) · Alt: [PyTorch Autograd](https://pytorch.org/docs/stable/autograd.html) · [JAX](https://jax.readthedocs.io/en/latest/)
- What you’ll learn: Manual derivatives vs autograd; stability notes.
- Role: Trusting your training loop.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Verify autograd output matches your manual gradients.

- Key terms explained
  - Autograd: Library feature computing derivatives automatically via chain rule.

- Assignment & Pass  
  - Do: For log-loss, softmax CE, L2 regularization—max difference (manual vs autograd) < 1e−6.  
  - Pass: Thresholds met; short commentary.

- Self-check: `tests/test_week08.py`.

---

### Week 9 — Probability & Distributions (bridge to stats)
- Sources: Primary: [MML Ch.6](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) · Alt: [STAT 414 Units 1–2](https://online.stat.psu.edu/stat414/)
- What you’ll learn: Basic distributions, expectation/variance, independence, Bayes intuition.
- Role: Warm start for stats.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Simulate CLT and explain it with simple plots.

- Key terms explained
  - CLT (Central Limit Theorem): Averages of many samples tend to be normally distributed.

- Assignment & Pass  
  - Do: CLT simulations; convergence plots + 1-page plain-language interpretation.  
  - Pass: Correct interpretation; reproducible code.

- Self-check: `tests/test_week09.py`.

---

### Week 10 — Continuous Optimization I: Convexity, Gradient Descent
- Sources: Primary: [MML Ch.7 (basics)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) · Alt: [Boyd & Vandenberghe Ch.1–2](https://web.stanford.edu/~boyd/cvxbook/)
- What you’ll learn: Convexity/strong convexity; GD convergence.
- Role: Training dynamics and LR selection.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Train logistic regression from scratch and compare LR schedules.

- Key terms explained
  - Convex function: Any line segment between two points on the graph lies above the graph—no “local” traps.
  - Learning rate (LR) schedule: How LR changes over time (constant/decay/cosine).

- Assignment & Pass  
  - Do: Train with constant/decay/cosine LR; best ROC-AUC improves ≥ +0.03 vs worst.  
  - Pass: Plots + short report.

- Self-check: `tests/test_week10.py`.

---

### Week 11 — Continuous Optimization II: Momentum, Adam(W)
- Sources: Primary: [MML Ch.7 (advanced)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) · Alt: [Deep Learning (Optimization)](https://www.deeplearningbook.org/)
- What you’ll learn: Momentum, Nesterov, Adam/AdamW; conditioning, scaling.
- Role: Stable/faster training.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Show statistically significant optimizer improvements across seeds.

- Key terms explained
  - Momentum: Adds fraction of previous update to smooth descent.
  - AdamW: Adaptive learning with decoupled weight decay—robust default.

- Assignment & Pass  
  - Do: Compare GD vs Momentum vs AdamW (5 seeds); best optimizer’s improvement has t-test p<0.05.  
  - Pass: Significance shown + brief methods note.

- Self-check: `tests/test_week11.py`.

---

### Week 12 — Math Mini-Project & Gate to Stats
- Sources: Primary: [MML (relevant sections)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- What you’ll learn: Standardize → PCA → Logistic → Optimizer comparisons.
- Role: Confirms stats readiness.
- Time & Load: 8–10h
- By the end of this week, you will be able to…  
  - Deliver a reproducible pipeline connecting math to ML.

- Key terms explained
  - Standardization: Make each feature mean 0, std 1.
  - ROC-AUC: Probability the model ranks a random positive above a random negative.

- Assignment & Pass (all required)  
  - Do: ROC-AUC ≥ 0.80; pass 20-question quiz (rank/QR/SVD/CLT/convexity) with ≥ 80%; README documents data/steps/results.  
  - Pass: ✅ All criteria met; otherwise ⚠️ Gap Week.

- Self-check: `tests/test_week12.py`.

Status: MML completed ✅

---

## PHASE 2 — Probability & Statistics (frequentist, 12 weeks) — complete All of Statistics
Primary: [All of Statistics (AoS)](https://link.springer.com/book/10.1007/978-0-387-21736-9)

### Week 13 — Probability Basics (AoS Ch.1–2)
- Sources: Primary: [AoS Ch.1–2](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [Casella & Berger Ch.1–2 (PDF)](https://pages.stat.wisc.edu/~shao/stat610/Casella_Berger_Statistical_Inference.pdf) · [STAT 414 (Units 1–2)](https://online.stat.psu.edu/stat414/)
- What you’ll learn: Probability spaces, conditional probability, Bayes, independence.
- Role: Core uncertainty calculus.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Compute conditional probabilities and apply Bayes’ rule.

- Key terms explained
  - Bayes’ rule: Update belief with new evidence: Posterior ∝ Likelihood × Prior.

- Assignment & Pass: Simulate Bayes updating; analytical vs simulation difference < 0.01.  
- Self-check: `tests/test_week13.py`.

---

### Week 14 — Distributions & Expectations (AoS Ch.3–4)
- Sources: Primary: [AoS Ch.3–4](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [Think Stats Ch.2–4 (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- What you’ll learn: Discrete/continuous distributions, moments, transforms.
- Role: Foundations for losses/metrics.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Simulate common distributions and verify their moments.

- Key terms explained
  - Expectation/Variance: Average value; average squared deviation from the mean.

- Assignment & Pass: Moment validation via simulation (±0.02).  
- Self-check: `tests/test_week14.py`.

---

### Week 15 — Multivariate Distributions & Dependence (AoS Ch.5–6)
- Sources: Primary: [AoS Ch.5–6](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [STAT 414 (Units 3–4)](https://online.stat.psu.edu/stat414/)
- What you’ll learn: Joint/conditional distributions, covariance/correlation.
- Role: Dependence structure for features.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Build a correlation heatmap and interpret it.

- Key terms explained
  - Covariance/Correlation: How two variables move together; correlation is scaled to [-1, 1].

- Assignment & Pass: Multivariate simulation + heatmap + interpretation.  
- Self-check: `tests/test_week15.py`.

---

### Week 16 — Sampling, LLN, CLT (AoS Ch.7–8)
- Sources: Primary: [AoS Ch.7–8](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [C&B (asymptotics) (PDF)](https://pages.stat.wisc.edu/~shao/stat610/Casella_Berger_Statistical_Inference.pdf)
- What you’ll learn: Sampling distributions, LLN, CLT.
- Role: Basis for CIs and tests.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Show empirically that sample means converge to normality.

- Key terms explained
  - LLN: With more samples, sample mean approaches true mean.

- Assignment & Pass: CLT convergence; KS test p>0.05 at adequate n.  
- Self-check: `tests/test_week16.py`.

---

### Week 17 — Estimation I: Unbiasedness, Sufficiency, UMVU (AoS Ch.9–10)
- Sources: Primary: [AoS Ch.9–10](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [STAT 415 (Estimation)](https://online.stat.psu.edu/stat415/)
- What you’ll learn: Unbiasedness, efficiency, sufficiency, UMVU.
- Role: Evaluating estimators.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Compare estimators via simulated bias and MSE tables.

- Key terms explained
  - Unbiased: On average, hits the true value.  
  - UMVU: Best (smallest variance) among unbiased estimators.

- Assignment & Pass: MSE comparisons + “which/why” justification.  
- Self-check: `tests/test_week17.py`.

---

### Week 18 — Estimation II: MLE & Asymptotics (AoS Ch.11–12)
- Sources: Primary: [AoS Ch.11–12](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [Think Stats (MLE examples)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- What you’ll learn: MLE, regularity, asymptotic normality.
- Role: Likelihood-based inference and CI building.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Fit MLEs numerically and compare with closed forms.

- Key terms explained
  - MLE: Parameter values that maximize the likelihood of observed data.

- Assignment & Pass: `scipy.optimize` MLE; analytical vs numeric difference < 1e−2.  
- Self-check: `tests/test_week18.py`.

---

### Week 19 — Hypothesis Testing & Power (AoS Ch.13–14)
- Sources: Primary: [AoS Ch.13–14](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [STAT 415 (Testing)](https://online.stat.psu.edu/stat415/)
- What you’ll learn: Neyman–Pearson, power, Type I/II errors.
- Role: Experiment design and model comparisons.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Draw power curves and compute sample sizes.

- Key terms explained
  - Power: Probability of detecting a real effect.  
  - Type I/II: False positive / false negative.

- Assignment & Pass: Power curves + sample size plan.  
- Self-check: `tests/test_week19.py`.

---

### Week 20 — Linear Regression (statistical framing) (AoS Ch.18)
- Sources: Primary: [AoS Ch.18](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [STAT 500 (Regression)](https://online.stat.psu.edu/stat500/) · [statsmodels (Linear Models)](https://www.statsmodels.org/dev/stats.html)
- What you’ll learn: OLS estimation, assumptions, diagnostics.
- Role: Foundation of supervised learning.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Fit OLS and interpret QQ, VIF, residual plots.

- Key terms explained
  - QQ-plot: Checks normality of residuals.  
  - VIF: Detects multicollinearity; high VIF → redundant predictors.

- Assignment & Pass: OLS + diagnostics report.  
- Self-check: `tests/test_week20.py`.

---

### Week 21 — ANOVA & Experimental Design (AoS relevant)
- Sources: Primary: [AoS (ANOVA/Design sections)](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [STAT 502 (ANOVA & DOE)](https://online.stat.psu.edu/stat502/) · [A/B Testing Guide](https://vkteam.medium.com/practitioners-guide-to-statistical-tests-ed2d580ef04f#1e3b)
- What you’ll learn: ANOVA logic, blocking, power.
- Role: Product/policy experiments.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Design an A/B test with justified sample size and error control.

- Key terms explained
  - ANOVA: Tests if group means differ.  
  - Blocking: Group similar units to reduce noise.

- Assignment & Pass: A/B design + power + false positive control notes.  
- Self-check: `tests/test_week21.py`.

---

### Week 22 — Logistic Regression & Categorical Data (GLM bridge)
- Sources: Primary: [AoS (Logistic/GLM sections)](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [STAT 504 (Categorical)](https://online.stat.psu.edu/stat504/)
- What you’ll learn: Logistic regression, odds ratios, calibration.
- Role: Binary classification foundation.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Fit/assess a logistic model (ROC/PR-AUC, calibration).

- Key terms explained
  - Odds ratio: How odds change per unit increase in a predictor.  
  - Calibration: Do predicted probabilities match observed frequencies?

- Assignment & Pass: Logistic model + calibration curve + ROC/PR-AUC.  
- Self-check: `tests/test_week22.py`.

---

### Week 23 — Multivariate Analysis (PCA)
- Sources: Primary: [AoS (PCA/multivariate sections)](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [STAT 505 (Multivariate)](https://online.stat.psu.edu/stat505/)
- What you’ll learn: PCA, covariance structure, visualization.
- Role: Dimensionality reduction for EDA/modeling.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Run PCA and interpret loadings/plots.

- Key terms explained
  - Loadings: How strongly each original variable influences a component.

- Assignment & Pass: PCA plots + short interpretation.  
- Self-check: `tests/test_week23.py`.

---

### Week 24 — Sampling & Frequentist Mini-Project
- Sources: Primary: [AoS (Sampling sections)](https://link.springer.com/book/10.1007/978-0-387-21736-9) · Alt: [STAT 506 (Sampling Theory)](https://online.stat.psu.edu/stat506/)
- What you’ll learn: Sampling designs; bias; weighting intuition.
- Role: Data collection bias control.
- Time & Load: 8–10h
- By the end of this week, you will be able to…  
  - Deliver a complete frequentist analysis with clear assumptions.

- Key terms explained
  - Sampling weights: Adjust for unequal sampling probabilities.

- Assignment & Pass (gate)  
  - Do: 4–6 page report + code—experiment/analysis/CI/test integrated; explain regression assumptions/diagnostics.  
  - Pass: Reproducible and sound; else ⚠️ Gap Week.

- Self-check: `tests/test_week24.py`.

Status: All of Statistics completed ✅

---

## PHASE 2B — Bayesian Statistics (6 weeks) — complete Statistical Rethinking (2e)
Primary: [Statistical Rethinking (book site)](https://xcelab.net/rm/statistical-rethinking/) · [Lectures](https://www.youtube.com/@rmcelreath)  
Tooling: R+Stan (brms/rstanarm) or Python+PyMC ([PyMC docs](https://www.pymc.io/projects/docs/en/stable/)). Optional: [Think Bayes](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple)

### Week 25 — Bayesian Foundations (SR Ch.1–2)
- What you’ll learn: Bayesian vs frequentist; priors/posteriors; generative mindset.
- Role: Foundation for Bayesian modeling.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Compute simple posteriors and compare priors.

- Key terms explained
  - Prior/Posterior: Belief before/after seeing data.
  - Conjugate prior: Prior that keeps posterior in the same family (easy math).

- Assignment & Pass: Conjugate + grid approx (binomial); ≥ 3 priors; match analytic/hi-precision refs (explain tolerance).  
- Self-check: `tests/test_week25.py`.

---

### Week 26 — Bayesian Regression & Posterior Predictive Checks (SR Ch.3–4)
- What you’ll learn: Bayesian linear regression; priors on β/σ; PPC.
- Role: Regression under uncertainty.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Fit BLR, run PPCs, and contrast with OLS.

- Key terms explained
  - Posterior Predictive Check (PPC): Simulate from the model and compare to observed data.

- Assignment & Pass: Fit BLR; PPC plots; BLR vs OLS comparison with uncertainty.  
- Self-check: `tests/test_week26.py`.

---

### Week 27 — Regularization & Model Comparison (SR Ch.5–6)
- What you’ll learn: Priors as regularizers; WAIC/LOO; overfitting control.
- Role: Principled model selection.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Compare ≥ 3 models via WAIC/LOO and justify choice.

- Key terms explained
  - WAIC/LOO: Bayesian criteria to compare models by out-of-sample fit.

- Assignment & Pass: WAIC/LOO table + PPC sanity; short justification.  
- Self-check: `tests/test_week27.py`.

---

### Week 28 — Categorical Outcomes (SR Ch.7–8)
- What you’ll learn: (Ordered) logistic models; priors; calibration; credible intervals.
- Role: Bayesian classification.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Fit Bayesian logistic and report calibrated probabilities.

- Key terms explained
  - Credible interval: Bayesian interval for the parameter with, say, 95% probability mass.

- Assignment & Pass: Fit model; calibration curve with credible intervals; posterior odds ratios.  
- Self-check: `tests/test_week28.py`.

---

### Week 29 — Multilevel/Hierarchical Models (SR Ch.9–10)
- What you’ll learn: Partial pooling; varying intercepts/slopes; shrinkage.
- Role: Robust grouped estimates.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Build a hierarchical model and show shrinkage benefits.

- Key terms explained
  - Partial pooling: Groups borrow strength; extreme estimates shrink toward overall mean.

- Assignment & Pass: Compare no pooling vs multilevel; interpret group posteriors.  
- Self-check: `tests/test_week29.py`.

---

### Week 30 — Causal Graphs & Bayesian Causal Thinking (SR Ch.11)
- What you’ll learn: DAGs, d-separation, backdoor/frontdoor.
- Role: Bridge to econometric identification.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Draw DAGs, find adjustment sets, and simulate bias removal.

- Key terms explained
  - DAG: Directed Acyclic Graph representing causal assumptions.
  - Backdoor criterion: Which variables to adjust for to block confounding paths.

- Assignment & Pass: DAG + adjustment set(s); simulate confounding vs adjusted.  
- Self-check: `tests/test_week30.py`.

Status: Statistical Rethinking completed ✅

---

## PHASE 3 — Econometrics & Causal Inference (16 weeks) — complete Stock & Watson
Primary: [Introduction to Econometrics (Stock & Watson)](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-econometrics/P200000003546/9780136647991)  
Alternatives: [Wooldridge](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge/) · [Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf) · [Greene (PDF)](https://www.ctanujit.org/uploads/2/5/3/9/25393293/_econometric_analysis_by_greence.pdf) · [Mostly Harmless Econometrics (MHE)](https://press.princeton.edu/books/hardcover/9780691120355/mostly-harmless-econometrics) · [Mixtape](https://mixtape.scunning.com/)

### Week 31 — OLS & Gauss–Markov (bridge from stats)
- What you’ll learn: BLUE, assumptions, interpretation.
- Role: Parametric inference backbone.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Fit/interpret OLS with uncertainty (CI/tests).

- Key terms explained
  - Gauss–Markov: Under assumptions, OLS is the “best” linear unbiased estimator.

- Assignment & Pass: OLS + CI/tests + effect sizes; clear narrative.  
- Self-check: `tests/test_week31.py`.

---

### Week 32 — Diagnostics: Heteroskedasticity, Autocorrelation, Robust SE
- What you’ll learn: HC-robust; autocorrelation tests/fixes.
- Role: Valid inference under violations.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Use robust/clustered SE and explain differences.

- Key terms explained
  - Heteroskedasticity: Error variance changes with predictors.  
  - Robust/Clustered SE: Fix SE estimates under such issues.

- Assignment & Pass: Refit with HC/cluster-robust; comparison note.  
- Self-check: `tests/test_week32.py`.

---

### Week 33 — Specification: Multicollinearity, OVB, Interactions
- What you’ll learn: VIF, omitted variable bias, interactions/dummies.
- Role: Correct model specification and interpretation.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Detect/fix misspecification and justify changes.

- Key terms explained
  - Omitted Variable Bias (OVB): Leaving out a relevant variable biases estimates.

- Assignment & Pass: Spec table + diagnostics-based decision.  
- Self-check: `tests/test_week33.py`.

---

### Week 34 — GLM Bridge Refresh (soften transition)
- What you’ll learn: Exponential family, links, MLE recap.
- Role: Gentle entry to Logit/Probit.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Choose a link and explain why.

- Key terms explained
  - Link function: Connects linear predictors to non-linear outcomes (e.g., logit).

- Assignment & Pass: Small GLMs; short link selection note.  
- Self-check: `tests/test_week34.py`.

---

### Week 35 — Binary Response: Logit/Probit
- What you’ll learn: MLE, marginal effects, calibration; ROC/PR-AUC.
- Role: Econometric classification.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Fit Logit/Probit; report marginal effects and calibration.

- Key terms explained
  - Marginal effect: Change in probability for a small change in a predictor.

- Assignment & Pass: Fit both; marginal effects table; calibration report.  
- Self-check: `tests/test_week35.py`.

---

### Week 36 — OLS Asymptotics & Consistency; Motivation for IV
- What you’ll learn: Endogeneity, OVB, consistency.
- Role: Why instruments are needed.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Simulate OVB and show bias; motivate IV.

- Key terms explained
  - Endogeneity: Predictor correlates with error (e.g., omitted confounder).

- Assignment & Pass: OVB simulation + written motivation for IV.  
- Self-check: `tests/test_week36.py`.

---

### Weeks 37–38 — Instrumental Variables & 2SLS
- What you’ll learn: Instrument validity/relevance; 2SLS; weak IV tests.
- Role: Identification with endogeneity.
- Time & Load: 2×12–14h
- By the end of these weeks, you will be able to…  
  - Estimate 2SLS and assess instrument strength.

- Key terms explained
  - First-stage F-stat (≥ 10 rule-of-thumb): Detects weak instruments.  
  - Over-identification test: Checks if multiple instruments agree.

- Assignment & Pass: First-stage F ≥ 10; over-ID test + interpretation; discuss validity.  
- Self-check: `tests/test_week37.py`, `tests/test_week38.py`.

---

### Weeks 39–40 — Panel Data I–II: FE/RE, Hausman, Cluster-Robust
- What you’ll learn: FE/RE assumptions; Hausman; clustered SE.
- Role: Control unobserved heterogeneity.
- Time & Load: 2×12–14h
- By the end of these weeks, you will be able to…  
  - Choose FE/RE (Hausman), justify clustered SE.

- Key terms explained
  - FE (Fixed Effects): Controls unit-specific constants; RE (Random Effects): Assumes random unit effects uncorrelated with regressors.  
  - Hausman test: Decides between FE and RE.

- Assignment & Pass: FE vs RE + Hausman + cluster-robust; decision note.  
- Self-check: `tests/test_week39.py`, `tests/test_week40.py`.

---

### Week 41 — Difference-in-Differences & Event Studies
- What you’ll learn: Parallel trends; event-study plots.
- Role: Natural experiments.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Test pre-trends and present an event-study figure.

- Key terms explained
  - Parallel trends: Treated vs control would track similarly absent treatment.

- Assignment & Pass: Pre-trend test; sensitivity variants; event-study plot.  
- Self-check: `tests/test_week41.py`.

---

### Week 42 — Regression Discontinuity
- What you’ll learn: Cutoffs; local polynomials; bandwidth sensitivity.
- Role: Strong quasi-experiment.
- Time & Load: 12–14h
- By the end of this week, you will be able to…  
  - Fit local polynomials and report bandwidth sensitivity.

- Key terms explained
  - Bandwidth: How close to the cutoff you look; smaller → less bias but more noise.

- Assignment & Pass: Sensitivity table/plot; robustness checks.  
- Self-check: `tests/test_week42.py`.

---

### Week 43 — Time-Series Bridge (econometric POV)
- What you’ll learn: Stationarity; AC/PAC; ARMA families (intro).
- Role: Bridge to forecasting.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Identify ARMA orders and transform to stationarity.

- Key terms explained
  - Stationarity: Distribution does not change over time (roughly constant mean/variance).

- Assignment & Pass: ARMA identification + stationarity transform.  
- Self-check: `tests/test_week43.py`.

---

### Weeks 44–46 — Replication + Mini Project
- What you’ll learn: Full pipeline for IV/DiD/RD/Panel with robustness culture.
- Role: Realistic causal analysis and reporting.
- Time & Load: 3×10–12h
- By the end of these weeks, you will be able to…  
  - Reproduce/execute a mini causal study with assumptions stated and tested.

- Key terms explained
  - Robustness checks: Try alternative specs/samples to see if result holds.

- Assignment & Pass: Full replication/study; clear identification; robustness/sensitivity; 4–6 page report + code.  
- Self-check: `tests/test_week44.py`, `tests/test_week45.py`, `tests/test_week46.py`.

Status: S&W core topics completed ✅ (Add 2–3 reading weeks if you want to fully read MHE chapters.)

Gate to ML  
- ≥ 2 methods among IV/DiD/RD/Panel completed with robust reports.  
- Assumptions/diagnostics correct; reproducible code + README.

---

## PHASE 4 — Classical Machine Learning (8 weeks) — complete ISLR v2
Primary: [An Introduction to Statistical Learning (ISLR v2)](https://www.statlearning.com/)  
Alternatives: [scikit-learn](https://scikit-learn.org/stable/index.html) · [ESL (PDF)](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf) · [Interpretable ML](https://christophm.github.io/interpretable-ml-book/) · [FIMD](https://stefvanbuuren.name/fimd/)

### Week 47 — Linear Regression & Validation (ISLR Ch.3/5)
- What you’ll learn: Train/val/test, K-fold, leakage avoidance.
- Role: Reliable evaluation.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Set up clean validation and report baselines.

- Key terms explained
  - Data leakage: Using information from validation/test in training by mistake.

- Assignment & Pass: Baseline + CV scheme; brief report.  
- Self-check: `tests/test_week47.py`.

---

### Week 48 — Classification, Metrics, Calibration (ISLR Ch.4)
- What you’ll learn: ROC/PR-AUC; calibration; thresholding.
- Role: Metric literacy.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Plot calibration and choose thresholds for business goals.

- Key terms explained
  - PR-AUC: Area under precision–recall—better for imbalanced data.

- Assignment & Pass: Calibration curves + threshold analysis.  
- Self-check: `tests/test_week48.py`.

---

### Week 49 — Regularization: Ridge/Lasso/ElasticNet (ISLR Ch.6)
- What you’ll learn: Bias–variance; sparsity; HPO.
- Role: Generalization improvements.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Tune regularization and justify choice.

- Key terms explained
  - Lasso: Encourages zeros (feature selection).  
  - ElasticNet: Mix of Ridge and Lasso.

- Assignment & Pass: HPO table + rationale.  
- Self-check: `tests/test_week49.py`.

---

### Week 50 — Trees & Ensembles (RF/GBM) (ISLR Ch.8)
- What you’ll learn: Trees; feature importance; interactions.
- Role: Strong tabular baselines.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Compare RF/GBM and provide basic explanations.

- Key terms explained
  - SHAP/LIME: Techniques to explain feature contributions locally.

- Assignment & Pass: RF/GBM comparison + SHAP/LIME intro.  
- Self-check: `tests/test_week50.py`.

---

### Week 51 — SVM & Kernels (ISLR Ch.9)
- What you’ll learn: Margin; C/γ tuning; kernels.
- Role: Powerful non-linear classifiers.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Run a kernel sweep and choose appropriately.

- Key terms explained
  - Kernel: Function enabling non-linear decision boundaries in a linear model.

- Assignment & Pass: Kernel sweep plots + selection rationale.  
- Self-check: `tests/test_week51.py`.

---

### Week 52 — Dimensionality Reduction & Clustering (ISLR Ch.10)
- What you’ll learn: PCA; KMeans/DBSCAN; visualization.
- Role: Segmentation & EDA.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Build PCA+KMeans with silhouette score.

- Key terms explained
  - Silhouette score: How well clusters are separated (−1 to 1, higher is better).

- Assignment & Pass: PCA+KMeans; silhouette report.  
- Self-check: `tests/test_week52.py`.

---

### Week 53 — Missing Data & Imputation
- What you’ll learn: MCAR/MAR/MNAR; multiple imputation; validation.
- Role: Data pipeline integrity/fairness.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Implement multiple imputation and assess impact.

- Key terms explained
  - Multiple imputation: Fill NAs several times; combine results for uncertainty.

- Assignment & Pass: Imputation pipeline + quality metrics.  
- Self-check: `tests/test_week53.py`.

---

### Week 54 — Classical ML Mini-Project
- What you’ll learn: End-to-end ML + explainability.
- Role: Production-leaning habits.
- Time & Load: 8–10h
- By the end of this week, you will be able to…  
  - Deliver a reproducible pipeline and model card.

- Key terms explained
  - Model card: Short document describing purpose, data, metrics, caveats.

- Assignment & Pass: Full pipeline + explainability report.  
- Self-check: `tests/test_week54.py`.

Status: ISLR completed ✅

---

## PHASE 5 — Time Series & Forecasting (6 weeks) — complete FPP3
Primary: [FPP3 (R)](https://otexts.com/fpp3/) · Alt: [FPP — Pythonic Way](https://otexts.com/fpppy/) · Advanced optional: [Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)

### Week 55 — Decomposition, Seasonality, ETS (FPP3 Ch.2–7)
- What you’ll learn: STL; ETS; seasonal patterns.
- Role: Forecasting basics.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Decompose and fit ETS models.

- Key terms explained
  - ETS: Error, Trend, Seasonality components in exponential smoothing.

- Assignment & Pass: ETS comparisons + short report.  
- Self-check: `tests/test_week55.py`.

---

### Weeks 56–57 — ARIMA/SARIMA (FPP3 Ch.8–9)
- What you’ll learn: Stationarity; ACF/PACF; model selection; diagnostics.
- Role: Classic forecasting workhorse.
- Time & Load: 2×12–14h
- By the end of these weeks, you will be able to…  
  - Fit SARIMA with justified orders and good diagnostics.

- Key terms explained
  - ACF/PACF: Autocorrelation/partial autocorrelation—help choose AR/MA orders.

- Assignment & Pass: SARIMA with full diagnostics.  
- Self-check: `tests/test_week56.py`, `tests/test_week57.py`.

---

### Week 58 — Time Series CV & Metrics
- What you’ll learn: Rolling-origin CV; MASE/SMAPE.
- Role: Reliable forecast comparison.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Compare models with tsCV and defend a choice.

- Key terms explained
  - MASE/SMAPE: Scale-free errors used in forecasting comparisons.

- Assignment & Pass: tsCV comparison table + selection rationale.  
- Self-check: `tests/test_week58.py`.

---

### Week 59 — ARIMAX/XREG, Multiple Series & Hierarchies
- What you’ll learn: Exogenous regressors; multiple series; hierarchical reconciliation.
- Role: Real-world forecasting at scale.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Improve ARIMA with XREG by ≥ 5% MASE.

- Key terms explained
  - XREG: External regressors (e.g., promotions, weather) to explain variation.

- Assignment & Pass: XREG improves MASE ≥ 5% vs ARIMA baseline.  
- Self-check: `tests/test_week59.py`.

---

### Week 60 — Time Series Mini-Project
- What you’ll learn: End-to-end forecasting pipeline.
- Role: Production-ready thinking.
- Time & Load: 8–10h
- By the end of this week, you will be able to…  
  - Deliver forecasts with clear metrics and choices.

- Key terms explained
  - Hierarchical forecasts: Consistent totals across levels (e.g., product → category).

- Assignment & Pass: Project + report (MASE/SMAPE).  
- Self-check: `tests/test_week60.py`.

Status: FPP3 completed ✅

---

## PHASE 6 — Deep Learning (8 weeks) — complete D2L
Primary: [Dive into Deep Learning (D2L)](https://d2l.ai) · Alternatives: [Deep Learning (Goodfellow et al.)](https://www.deeplearningbook.org/) · [Applied ML Practices](https://github.com/eugeneyan/applied-ml)

### Week 61 — Autograd & Training Loop
- What you’ll learn: Tensors; autograd; training loop.
- Role: Core DL training.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Build and verify a training loop.

- Key terms explained
  - Mini-batch: Small subset of data per step—stabilizes and speeds training.

- Assignment & Pass: Training loop + gradient verification.  
- Self-check: `tests/test_week61.py`.

---

### Week 62 — MLP, Regularization, Optimization
- What you’ll learn: Weight decay; dropout; LR scheduling.
- Role: Generalization/stability.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Show significant improvement via HPO (p<0.05).

- Key terms explained
  - Dropout: Randomly drop units at train time to reduce overfitting.

- Assignment & Pass: HPO with significance.  
- Self-check: `tests/test_week62.py`.

---

### Week 63 — CNN & Transfer Learning
- What you’ll learn: Convolutions; augmentation; TL.
- Role: Standard for vision tasks.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Improve top-1 accuracy by ≥ 5pp via transfer learning.

- Key terms explained
  - Transfer Learning: Start from a pretrained model and fine-tune.

- Assignment & Pass: TL improvement ≥ 5pp vs scratch.  
- Self-check: `tests/test_week63.py`.

---

### Week 64 — Sequence Models (RNN/GRU/LSTM)
- What you’ll learn: Sequence modeling; regularization.
- Role: NLP/time-series DL.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Build a text classification pipeline with metrics.

- Key terms explained
  - LSTM/GRU: RNN variants that remember long-term dependencies.

- Assignment & Pass: Working pipeline + evaluation.  
- Self-check: `tests/test_week64.py`.

---

### Week 65 — Attention & Transformers
- What you’ll learn: Self-attention; encoder/decoder basics.
- Role: Modern NLP/CV backbone.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Fine-tune a small HF model and report results.

- Key terms explained
  - Self-attention: Each token attends to others to gather context.

- Assignment & Pass: Fine-tune + metrics.  
- Self-check: `tests/test_week65.py`.

---

### Week 66 — Performance & Training Tricks
- What you’ll learn: LR schedules; early stopping; mixed precision.
- Role: Efficient, stable training.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Produce ablations with clear gains.

- Key terms explained
  - Mixed precision: Use float16 where safe to speed up training.

- Assignment & Pass: Ablation + performance report.  
- Self-check: `tests/test_week66.py`.

---

### Weeks 67–68 — DL Mini-Project (2 weeks)
- What you’ll learn: CV or NLP PoC near production.
- Role: From prototype to reliable system.
- Time & Load: 2×8–10h
- By the end of these weeks, you will be able to…  
  - Deliver a reproducible DL project with metrics and lessons.

- Key terms explained
  - Early stopping: Stop when validation stops improving to avoid overfit.

- Assignment & Pass: Project + report (metrics, ablation).  
- Self-check: `tests/test_week67.py`, `tests/test_week68.py`.

Status: D2L completed ✅

---

## PHASE 7 — MLOps (5 weeks) — complete MLOps Zoomcamp
Primary: [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) · Alt: [Machine Learning Systems](https://mlsysbook.ai)

### Week 69 — Experiment Tracking & Model Registry
- What you’ll learn: MLflow runs; model registry; model cards; versioning basics.
- Role: Reproducibility and governance.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Track experiments, register models, write a model card.

- Key terms explained
  - Model registry: Central store for versioned models ready for deployment.

- Assignment & Pass: MLflow runs + registry + model card.  
- Self-check: `tests/test_week69.py`.

---

### Week 70 — Data/Model Pipelines
- What you’ll learn: DAGs; feature store; data contracts.
- Role: Reliability at scale.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Build and document an end-to-end pipeline.

- Key terms explained
  - Data contract: Agreement on schema/fields (prevents breaking changes).

- Assignment & Pass: Pipeline + documentation.  
- Self-check: `tests/test_week70.py`.

---

### Week 71 — Deployment (FastAPI) & Docker
- What you’ll learn: REST inference service; containerization; basic CI.
- Role: Make models usable.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Serve a model and call it with curl from README.

- Key terms explained
  - Container image: Pack code + dependencies to run anywhere.

- Assignment & Pass: Docker image + live endpoint (curl example).  
- Self-check: `tests/test_week71.py`.

---

### Week 72 — Monitoring & Drift
- What you’ll learn: Data/concept drift; thresholds; alerts; dashboards.
- Role: Keep models healthy after deploy.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Create drift reports and set alerts.

- Key terms explained
  - Concept drift: Relationship between inputs and output changes over time.

- Assignment & Pass: Drift report + dashboard screenshots + alert config notes.  
- Self-check: `tests/test_week72.py`.

---

### Week 73 — Mini MLOps PoC
- What you’ll learn: Small end-to-end production-like setup.
- Role: Ops mindset.
- Time & Load: 8–10h
- By the end of this week, you will be able to…  
  - Present a minimal production workflow with a runbook.

- Key terms explained
  - Runbook: Step-by-step “how to run/operate” instructions.

- Assignment & Pass: PoC diagrams + README runbook.  
- Self-check: `tests/test_week73.py`.

Status: MLOps Zoomcamp completed ✅

---

## PHASE 8 — Data Engineering (5 weeks) — complete DE Zoomcamp
Primary: [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp) · Alt: [Data Mining: Concepts & Techniques (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)

### Week 74 — Orchestration & Data Quality
- What you’ll learn: Airflow; schema contracts; Great Expectations/dbt checks.
- Role: Trustworthy data feeds.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Orchestrate a DAG and validate data quality.

- Key terms explained
  - DAG (workflow): Directed steps with dependencies (e.g., extract → transform → load).

- Assignment & Pass: Airflow DAG + quality report.  
- Self-check: `tests/test_week74.py`.

---

### Week 75 — Batch ETL & Storage
- What you’ll learn: File formats; partitioning; cost/scale.
- Role: Efficient pipelines.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Build a batch ETL and reason about storage/cost.

- Key terms explained
  - Partitioning: Split large tables by date/key for faster queries and cheaper storage.

- Assignment & Pass: Batch ETL + cost notes.  
- Self-check: `tests/test_week75.py`.

---

### Week 76 — Streaming
- What you’ll learn: Messaging; latency; exactly-once patterns.
- Role: Real-time use cases.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Implement a minimal streaming demo and explain guarantees.

- Key terms explained
  - Exactly-once: Each event is processed once despite failures/dedup steps.

- Assignment & Pass: Simple streaming pipeline + notes.  
- Self-check: `tests/test_week76.py`.

---

### Weeks 77–78 — Mini Data Platform Project
- What you’ll learn: Storage → ETL → Serving end-to-end.
- Role: Connect DE with MLOps.
- Time & Load: 2×8–10h
- By the end of these weeks, you will be able to…  
  - Deliver a small platform with diagrams and run steps.

- Key terms explained
  - Serving layer: Where cleaned data or model outputs are exposed to apps/users.

- Assignment & Pass: README + diagrams + run instructions; small demo dataset.  
- Self-check: `tests/test_week77.py`, `tests/test_week78.py`.

Status: DE Zoomcamp completed ✅

---

## PHASE 9 — LLMs (3 weeks) — complete Hugging Face Course
Primary: [Hugging Face Course](https://huggingface.co/course/chapter1) · Alts: [HF Docs (Pipelines/Trainer)](https://huggingface.co/docs) · [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

### Week 79 — HF Core Modules
- What you’ll learn: Transformers; Datasets; Tokenizers; Pipelines.
- Role: Modern NLP tooling.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Fine-tune a small model and report metrics.

- Key terms explained
  - Tokenizer: Splits text into tokens (sub-words) expected by models.

- Assignment & Pass: Simple fine-tune + metrics report.  
- Self-check: `tests/test_week79.py`.

---

### Week 80 — Tasks: Classification / QA / Summarization
- What you’ll learn: Training; evaluation; stability checks.
- Role: Practical NLP.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Reach a target score on at least one NLP task.

- Key terms explained
  - QA (Question Answering): Predict answer spans or generate answers from context.

- Assignment & Pass: Achieve target score; brief error analysis.  
- Self-check: `tests/test_week80.py`.

---

### Week 81 — Agents (Optional) & Integration
- What you’ll learn: Agent chains; evaluation; safety/guardrails.
- Role: Autonomous flows.
- Time & Load: 8–10h
- By the end of this week, you will be able to…  
  - Build a small agent demo and document risks/guardrails.

- Key terms explained
  - Guardrails: Rules/sanitizers to keep outputs safe and on-policy.

- Assignment & Pass: Agent demo + risk/guardrail notes.  
- Self-check: `tests/test_week81.py`.

Status: HF Course completed ✅

---

## PHASE 10 — Capstone (2–3 weeks)

### Week 82 — Problem, Data, and Architecture
- Sources: Primary: your domain data + earlier artifacts · Alt: [MLOps Zoomcamp (Design)](https://github.com/DataTalksClub/mlops-zoomcamp)
- What you’ll learn: Requirements; KPIs; data contracts; architecture diagrams.
- Role: Production planning.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Present a complete design doc ready for execution.

- Key terms explained
  - KPI: Metric that reflects business/mission impact (e.g., conversion rate).

- Assignment & Pass: Design doc reviewed/approved.  
- Self-check: `tests/test_week82.py` (lint/format/CI smoke).

---

### Week 83 — Modeling, Service, Monitoring
- Sources: Primary: scikit-learn/TF/PyTorch + FastAPI + MLflow · Alt: [mlsysbook](https://mlsysbook.ai)
- What you’ll learn: Model(s); API; monitoring dashboard; ethics/fairness note.
- Role: End-to-end delivery.
- Time & Load: 10–12h
- By the end of this week, you will be able to…  
  - Ship a working system with report and slides.

- Key terms explained
  - Fairness note: How you checked bias and mitigations considered.

- Assignment & Pass: Working system + 6–10 page report + slides.  
- Self-check: `tests/test_week83.py`.

---

### Week 84 (optional) — Presentation & Feedback Loop
- Sources: Primary: project artifacts
- What you’ll learn: Decision narrative; ROI/impact; iteration planning.
- Role: Stakeholder communication.
- Time & Load: 6–8h
- By the end of this week, you will be able to…  
  - Deliver a clear presentation and define next steps.

- Key terms explained
  - ROI: Return on investment—benefit vs cost.

- Assignment & Pass: Presentation delivered; iteration plan captured.  
- Self-check: Presentation checklist (scope, metrics, risks, ethics).

---

## Why this is the right roadmap to become a “data specialist”
- Complete primary sources: MML, All of Statistics, Statistical Rethinking, Stock & Watson, ISLR, FPP3, D2L, MLOps & DE Zoomcamps, HF Course—each is explicitly covered and finished.
- Dual inference then causality: Frequentist + Bayesian foundations, then identification (IV/DiD/RD/Panel) with DAG intuition.
- Measurable progress: Weekly numeric thresholds and tests; Gap Weeks ensure no gaps remain.
- End-to-end skills: Data pipelines (DE), modeling (classical/DL/TS), serving/monitoring (MLOps), LLM tasks—culminating in a production-flavored capstone.
- Plain-language clarity: Unknown terms are defined where they appear, with tiny examples when helpful.

If any main source needs more time, insert a Gap Week to finish remaining chapters/units and add a brief “What I finished” note before continuing.

Good luck—and always explain not only what works, but why it works, under which assumptions, and how you verified it.
