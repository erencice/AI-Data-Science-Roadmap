# 🎓 Data Science & Artificial Intelligence Mastery Roadmap

---

> **A Professional, Execution-Oriented Syllabus for Serious Practitioners**

> **Study Pace:** 12–15 hours per week (minimum)

> **Total Estimated Duration:** ~192 weeks (~3.7 years)

> **Format:** Self-paced, resource-integrated, project-driven

---

## 📋 Roadmap Overview

| Phase | Name | Duration | Status |
|-------|------|----------|--------|
| 1 | Data Analysis Foundations | 8 weeks | ✅ Completed |
| 2 | SQL | 4 weeks | 🚧 In Progress |
| 3 | Introduction to Probability & Linear Algebra | 16 weeks | ⏳ Not Started |
| 4 | Statistics Fundamentals | 6 weeks | ⏳ Not Started |
| 5 | Mathematical Statistics | 12 weeks | ⏳ Not Started |
| 6 | Applied Multivariate Statistics | 8 weeks | ⏳ Not Started |
| 7 | Bayesian Statistics & Missing Data | 9 weeks | ⏳ Not Started |
| 8 | Statistical Learning with Python (ISLP) | 12 weeks | ⏳ Not Started |
| 9 | Data Mining | 10 weeks | ⏳ Not Started |
| 10 | Classical Machine Learning | 16 weeks | ⏳ Not Started |
| 11 | Elements of Statistical Learning | 14 weeks | ⏳ Not Started |
| 12 | Deep Learning | 20 weeks | ⏳ Not Started |
| 13 | R for Data Science | 7 weeks | ⏳ Not Started |
| 14 | Econometrics, Time Series & Financial Econometrics | 16 weeks | ⏳ Not Started |
| 15 | Causal Inference | 10 weeks | ⏳ Not Started |
| 16 | MLOps & Data Engineering | 16 weeks | ⏳ Not Started |
| 17 | Flow Matching & Diffusion Models | 8 weeks | ⏳ Not Started |
| | **TOTAL** | **~192 weeks** | |

---

## 🧭 How to Use This Roadmap

- **Core Principle:** Starting a resource is irrelevant. Finishing it completely is what matters.
- **Finished means all of it.** All chapters read, all lectures watched, all exercises solved, all labs completed, all checkpoints passed, all projects implemented.
- **Never skip a phase.** Each phase is a hard prerequisite for the next.
- **Never skip exercises.** Superficial familiarity is not accepted.
- **Track your hours.** Aim for 12–15 hours per week minimum.
- **Run weekly completion audits.** Unfinished items roll forward before any new content.
- **Enforce phase gates.** You cannot enter the next phase until deliverables and mastery criteria are fully met.
- **Use monthly cumulative review blocks.** Every 4th week block includes review to prevent shallow progression.

---

---

<details>
<summary><h2>Phase 1: Data Analysis Foundations</h2></summary>

**Duration:** 8 Weeks
**Resource:** [Python for Data Analysis, 3rd Ed. — Wes McKinney](https://wesmckinney.com/book/)
**Depth Assessment:** ~550 pages. Practical, code-heavy, moderate difficulty. The definitive reference for pandas and NumPy. At 12–15 hrs/week this translates to disciplined chapter completion plus full labs and capstone hardening.
**Why this resource matters:** Core Python/NumPy/pandas implementation fluency for everything downstream.
**Weekly Structure (Strict Completion):** Original Week 1–6 sequence + 2 completion buffer weeks.
**Phase Deliverables:** All weekly notebooks + end-to-end capstone (clean, reproducible, tested).
**Phase Mastery Criteria:** Every chapter, exercise, lab, checkpoint, and capstone step completed.

---

### Week 1: Python Environment, NumPy & Data Structures

**Study:**
- Read Chapters 1–3 (Preliminaries, Python Language Basics, Built-in Data Structures)
- Set up your full environment: Python 3.11+, Jupyter Lab, conda environments

**Practice:**
- Write 30+ Python one-liners covering list comprehensions, dict/set operations, generators, and lambda functions
- Implement a simple custom class (e.g., `DataRecord`) with `__repr__`, `__len__`, and iteration support

**Checkpoint:**
> Create a Jupyter notebook titled `week01_python_foundations.ipynb`. It must demonstrate mastery of all built-in data structures, implement at least one generator function, and pass 10 self-written `assert` unit tests.

---

### Week 2: NumPy — Arrays, Vectorization & Broadcasting

**Study:**
- Read Chapter 4 (NumPy Basics: Arrays and Vectorized Computation) in full
- Supplement: Skim the official NumPy documentation on broadcasting rules

**Practice:**
- Implement matrix multiplication **from scratch** using only NumPy array indexing (no `np.matmul`)
- Benchmark your manual implementation vs. `np.dot` and record the speedup

**Checkpoint:**
> Submit `week02_numpy.ipynb` containing: a visual explanation of broadcasting with annotated examples, a performance benchmark table comparing pure Python loops vs. vectorized NumPy for 5 operations, and your from-scratch matrix multiply function.

---

### Week 3: pandas I — Loading, Indexing & Cleaning

**Study:**
- Read Chapters 5–6 (Getting Started with pandas; Data Loading, Storage & File Formats)
- Focus: Series, DataFrame, indexing (`.loc`, `.iloc`, boolean indexing), I/O (CSV, JSON, Excel, SQL)

**Practice:**
- Download a real-world messy dataset (e.g., [NYC 311 Service Requests](https://data.cityofnewyork.us/)) with at least 100,000 rows
- Write a complete data cleaning pipeline: handle missing values, fix dtypes, rename columns, parse dates

**Checkpoint:**
> Produce `week03_pandas_cleaning.ipynb`. It must: load the dataset, generate a "data quality report" (% nulls, dtype summary, unique value counts per column), and output a fully cleaned DataFrame saved to Parquet.

---

### Week 4: pandas II — Transformation, GroupBy & Merging

**Study:**
- Read Chapters 7–8 (Data Cleaning and Preparation; Data Wrangling: Join, Combine, Reshape)
- Supplement Chapter 10 (Data Aggregation and Group Operations)

**Practice:**
- Using the dataset from Week 3, write a full GroupBy analysis: compute min, max, mean, median, and a custom aggregation function in a single `.agg()` call
- Practice `merge`, `join`, and `concat` by combining two related real datasets

**Checkpoint:**
> `week04_groupby_merge.ipynb` must include: at least 3 non-trivial GroupBy analyses with custom `agg` functions, a multi-key merge demonstrating all join types (inner, left, right, outer), and a pivot table with proper formatting.

---

### Week 5: Time Series, Visualization & Advanced Features

**Study:**
- Read Chapters 11–12 (Time Series; Advanced pandas)
- Read Chapter 9 (Plotting and Visualization) — focus on matplotlib integration

**Practice:**
- Download historical stock price data (e.g., via `yfinance`) for 5 tickers
- Build a time-series analysis notebook: resample to weekly/monthly, compute rolling statistics, handle timezone localization, plot with annotated events

**Checkpoint:**
> `week05_timeseries.ipynb` must: demonstrate `resample`, `rolling`, `shift`, and `ewm`; produce at least 4 publication-quality matplotlib plots with titles, axis labels, and legends; and compute a 20-day/50-day moving average crossover signal.

---

### Week 6: Capstone — End-to-End Data Analysis Project

**Study:**
- Review Chapters 13–14 (Introduction to Modeling Libraries; Data Analysis Examples)
- Re-read any sections where you felt weakest during Weeks 1–5

**Practice:**
- Full end-to-end analysis on a chosen dataset (e.g., [Titanic](https://www.kaggle.com/c/titanic/data), [AirBnb NYC](http://insideairbnb.com/), or similar)
- Pipeline must cover: ingestion → cleaning → feature engineering → GroupBy insights → visualization → written narrative conclusions

**Checkpoint:**
> Deliver `phase1_capstone.ipynb` with a structured narrative (using Markdown cells), at least 8 visualizations, a summary table of key findings, and a "Limitations & Next Steps" section. Must be reproducible end-to-end with a single `Run All` command.

---

---

</details>

---

<details>
<summary><h2>Phase 2: SQL</h2></summary>

**Duration:** 4 Weeks
**Resource:** [SQL Roadmap — GeeksforGeeks](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
**Supplementary Platform:** [SQLZoo](https://sqlzoo.net/), [pgexercises.com](https://pgexercises.com/), [LeetCode SQL](https://leetcode.com/problemset/database/)
**Depth Assessment:** The GfG roadmap is a structured guide, not a textbook. The core SQL curriculum (~60 topics) at this pace covers 4 weeks with daily practice and a dedicated advanced drill week.
**Why this resource matters:** Query thinking, relational design, and analytics-grade data retrieval.
**Weekly Structure (Strict Completion):** Existing Week 7–9 topics + 1 dedicated advanced drill week.
**Phase Deliverables:** Schema DDL, seed scripts, solved SQL sets, advanced optimization notebook.
**Phase Mastery Criteria:** All roadmap topics solved across platforms with no unresolved query class gaps.

---

### Week 7: SQL Foundations — DDL, DML & Basic Queries

**Study:**
- GfG Roadmap: "SQL Basics" → "DDL Commands" → "DML Commands" → "SELECT Queries"
- Topics: `CREATE`, `ALTER`, `DROP`, `INSERT`, `UPDATE`, `DELETE`, `WHERE`, `ORDER BY`, `LIMIT`
- Install PostgreSQL locally; use pgAdmin or DBeaver as your IDE

**Practice:**
- Design and create a normalized 3-table schema (e.g., `customers`, `orders`, `products`)
- Populate it with 1000+ rows using a script
- Write 20 SELECT queries of increasing complexity

**Checkpoint:**
> Submit a `.sql` file containing your schema DDL, a seed script, and 20 documented queries. Every query must have a comment explaining its business question. Run all on PostgreSQL with no errors.

---

### Week 8: Intermediate SQL — Joins, Aggregations & Subqueries

**Study:**
- GfG Roadmap: "Joins" (all types) → "Aggregate Functions" → "GROUP BY / HAVING" → "Subqueries"
- Topics: `INNER`, `LEFT`, `RIGHT`, `FULL OUTER`, `CROSS`, `SELF` joins; `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`; correlated subqueries; `EXISTS` / `IN`

**Practice:**
- Solve 15 problems on [pgexercises.com](https://pgexercises.com/) (Joins + Aggregates sections)
- Solve 5 LeetCode SQL medium problems (e.g., "Consecutive Numbers," "Department Top 3 Salaries")

**Checkpoint:**
> A documented solution notebook (Markdown + SQL blocks) for all 20 problems. For each problem: state the question, write the query, and explain the join/subquery logic in one sentence.

---

### Week 9: Advanced SQL — Window Functions, CTEs, Indexes & Optimization

**Study:**
- GfG Roadmap: "Window Functions" → "CTEs" → "Stored Procedures & Functions" → "Indexes & Query Optimization"
- Topics: `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`, `PARTITION BY`; `WITH` CTEs; `EXPLAIN ANALYZE`; B-tree indexes

**Practice:**
- Rewrite 5 of your Week 8 subquery solutions using CTEs — compare readability
- Use `EXPLAIN ANALYZE` to profile 3 queries; add appropriate indexes and document the speedup
- Implement a running total and a 3-period moving average using window functions on your orders table

**Checkpoint:**
> `phase2_sql_advanced.sql`: contains all window function solutions, CTE rewrites with inline commentary, and a performance report (before/after `EXPLAIN ANALYZE` output) showing measurable query plan improvement.

---

---

</details>

---

<details>
<summary><h2>Phase 3: Introduction to Probability & Linear Algebra</h2></summary>

**Duration:** 16 Weeks
**Resources (Interwoven):**
- [Statistics 110 — Harvard (YouTube, 34 lectures)](https://youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo)
- [Introduction to Probability — Blitzstein & Hwang (probabilitybook.net)](http://probabilitybook.net)
- [Essence of Linear Algebra — 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Khan Academy Linear Algebra](https://www.khanacademy.org/math/linear-algebra)

**Depth Assessment:** Stat 110 = 34 × ~70 min = ~40 hrs. Blitzstein book = ~600 pages, rigorous. 3B1B = ~3 hrs total. Khan = ~20 hrs. Linear algebra is interwoven in the second half of this phase.
**Why this resource matters:** Mathematical bedrock for statistics/ML proofs and modeling intuition.
**Weekly Structure (Strict Completion):** Keep original Week 10–20 order + 5 proof/problem completion weeks.
**Phase Deliverables:** Simulation notebooks, derivation writeups, and linear algebra computation portfolio.
**Phase Mastery Criteria:** All lectures watched, all assigned problems solved, derivations reproducible without notes.

---

### Week 10: Sample Spaces, Events & Axioms of Probability

**Study:**
- Watch Stat 110 Lectures 1–3 (Sample Spaces, Probability Axioms, Birthday Problem)
- Read Blitzstein Ch. 1 (Probability and Counting)
- Complete all Ch. 1 practice problems

**Practice:**
- Simulate the Birthday Problem in Python for group sizes 1–80; plot the probability curve
- Code a Monte Carlo estimator for the probability that at least 2 people share a birthday — verify it converges to the analytic formula

**Checkpoint:**
> `week10_probability.ipynb`: simulation code, convergence plot (simulation vs. analytic), and written derivation of the Birthday Problem formula in LaTeX (Markdown math mode).

---

### Week 11: Conditional Probability & Bayes' Theorem

**Study:**
- Watch Stat 110 Lectures 4–6 (Conditional Probability, Bayes' Theorem, Independence)
- Read Blitzstein Ch. 2 (Conditional Probability)

**Practice:**
- Implement Bayes' Theorem for a medical test scenario (sensitivity, specificity, prevalence) as a Python function
- Simulate the Monty Hall Problem (N=100,000 trials) and compare "switch" vs. "stay" win rates

**Checkpoint:**
> `week11_bayes.ipynb`: Monty Hall simulation with bar chart, a generalized Bayes' calculator function, and a written answer explaining why counterintuitive results arise.

---

### Week 12: Discrete Random Variables & Distributions

**Study:**
- Watch Stat 110 Lectures 7–10 (Discrete RVs, Binomial, Hypergeometric, Geometric)
- Read Blitzstein Ch. 3–4 (Random Variables and Their Distributions; Expectation)

**Practice:**
- Implement PMF and CDF from scratch for Binomial and Poisson distributions (no `scipy.stats`)
- Compare your implementations against `scipy.stats` — verify they match

**Checkpoint:**
> `week12_discrete_rvs.ipynb`: from-scratch PMF/CDF implementations, side-by-side comparison plots vs. `scipy.stats`, and a short proof (in LaTeX) that the Poisson distribution is the limit of Binomial as n→∞, λ=np fixed.

---

### Week 13: Continuous Distributions & the Normal Distribution

**Study:**
- Watch Stat 110 Lectures 11–14 (Continuous RVs, Uniform, Normal, Exponential)
- Read Blitzstein Ch. 5 (Continuous Random Variables)

**Practice:**
- Plot PDFs and CDFs for Uniform, Exponential, and Normal distributions with varied parameters
- Implement the Box-Muller transform to generate Normal samples from Uniform random numbers — verify normality with a Q-Q plot

**Checkpoint:**
> `week13_continuous.ipynb`: Box-Muller implementation, Q-Q plot vs. theoretical normal, and distribution parameter sensitivity plots.

---

### Week 14: Expectation, Variance & Moment Generating Functions

**Study:**
- Watch Stat 110 Lectures 15–17 (Expectation, Variance, MGFs)
- Read Blitzstein Ch. 6 (Moments)

**Practice:**
- Derive (on paper) the expectation and variance of the Binomial, Poisson, and Exponential distributions
- Implement a general `moment(k, distribution)` function using numerical integration (`scipy.integrate.quad`)

**Checkpoint:**
> Scan/photograph your paper derivations and include in a notebook with the numerical moment function. The function must correctly compute the first 4 moments for at least 3 distributions.

---

### Week 15: Joint Distributions, Covariance & Correlation

**Study:**
- Watch Stat 110 Lectures 18–20 (Joint Distributions, Covariance, Correlation)
- Read Blitzstein Ch. 7 (Joint Distributions)
- **Begin Linear Algebra:** Watch 3Blue1Brown Episodes 1–5 (Vectors, Linear Combinations, Matrix Transformations, Determinants)

**Practice:**
- Simulate a 2D multivariate normal with a specified covariance matrix; plot the joint density as a contour map
- Animate (using matplotlib) how a 2×2 matrix transformation distorts the unit circle

**Checkpoint:**
> `week15_joint_linalg.ipynb`: joint density contour plot, animated matrix transformation (saved as GIF), and a written explanation of what the determinant geometrically represents.

---

### Week 16: Law of Large Numbers, CLT & Linear Algebra Foundations

**Study:**
- Watch Stat 110 Lectures 21–23 (LLN, CLT, Chi-Squared)
- Read Blitzstein Ch. 10 (Inequalities and Limit Theorems)
- Khan Academy Linear Algebra: "Vectors and Spaces" unit + "Matrix Transformations" unit

**Practice:**
- Simulate LLN convergence for 5 different distributions — plot mean estimates vs. sample size on log-scale
- Demonstrate CLT: sample means of a highly skewed distribution approach normality — animate with increasing n

**Checkpoint:**
> `week16_lln_clt.ipynb`: LLN convergence plots (log-scale), CLT animation for a skewed distribution, and a written statement of both theorems with the key conditions.

---

### Week 17: Matrix Operations — Eigenvectors, Eigenvalues & Decompositions

**Study:**
- Watch 3Blue1Brown Episodes 6–15 (Dot Products, Cross Products, Eigenvectors, Change of Basis, Abstract Vector Spaces)
- Khan Academy: "Eigenvalues and Eigenvectors" unit
- Supplement: NumPy `linalg` documentation

**Practice:**
- Implement Power Iteration from scratch to find the dominant eigenvector of a matrix
- Apply eigendecomposition to PCA by hand on a 2D dataset — compare to `sklearn.decomposition.PCA`

**Checkpoint:**
> `week17_eigens.ipynb`: Power Iteration implementation with convergence tracking, visual comparison of manual PCA vs. sklearn PCA on a 2D scatter plot, with written explanation of what each eigenvector represents.

---

### Week 18: SVD, Projections & Linear Systems

**Study:**
- Khan Academy: "Alternate coordinate systems" + "Orthogonality" units
- Stat 110 Lectures 24–25 (Order Statistics, Conditional Expectation)
- Read Blitzstein Ch. 9 (Conditional Expectation)

**Practice:**
- Implement image compression using SVD — show the reconstruction at k=5, 20, 50, 100 singular values
- Solve a linear regression as a linear system (`Ax = b`) using both SVD and normal equations — compare solutions

**Checkpoint:**
> `week18_svd.ipynb`: image compression at multiple ranks with PSNR metric table, linear regression solved via SVD vs. normal equations with numerical comparison.

---

### Week 19: Markov Chains & Generating Functions

**Study:**
- Watch Stat 110 Lectures 26–30 (Markov Chains, Stationary Distribution, Gambler's Ruin)
- Read Blitzstein Ch. 11 (Markov Chains)

**Practice:**
- Implement a Markov Chain from scratch with a custom transition matrix
- Simulate the Gambler's Ruin problem for multiple starting capitals and compute empirical vs. theoretical ruin probabilities

**Checkpoint:**
> `week19_markov.ipynb`: Markov Chain class with `.simulate()`, `.stationary_distribution()` methods; Gambler's Ruin comparison table; and a state-transition diagram (using `networkx`).

---

### Week 20: Phase 3 Review & Capstone Problem Set

**Study:**
- Watch Stat 110 Lectures 31–34 (Poisson Processes, Beta, Dirichlet, Final Review)
- Read Blitzstein Ch. 12–13 (Markov Chains continued, Poisson Processes)

**Practice:**
- Solve 10 problems from the Blitzstein book's Strategic Practice sets (Chapters 1–12)
- Implement a simple Poisson Process simulator and verify inter-arrival times are exponentially distributed

**Checkpoint:**
> `phase3_capstone_problems.ipynb`: 10 solved practice problems with written mathematical derivations + code verification. Must include at least one proof by induction and one moment generating function derivation.

---

---

</details>

---

<details>
<summary><h2>Phase 4: Statistics Fundamentals</h2></summary>

**Duration:** 6 Weeks
**Resource:** [Think Stats, 2nd Ed. — Allen B. Downey](https://allendowney.github.io/ThinkStats/)
**Depth Assessment:** ~300 pages, Python-based, moderate difficulty. Bridges probability theory with practical statistical analysis. Strong overlap with Phase 3, so the pace is faster.
**Why this resource matters:** Bridge from probability theory to practical data-driven inference.
**Weekly Structure (Strict Completion):** Week 21–24 core + 2 full reproduction/extension weeks.
**Phase Deliverables:** Full chapter analysis notebooks + phase capstone.
**Phase Mastery Criteria:** All chapter exercises/labs completed; every figure/result reproducible and interpretable.

---

### Week 21: Exploratory Statistics — Distributions & PMFs

**Study:**
- Read Think Stats Chapters 1–3 (Histograms, PMFs, CDFs)
- Contrast with your Phase 1 pandas EDA work

**Practice:**
- Download the NSFG (National Survey of Family Growth) dataset used in the book
- Reproduce and extend all Chapter 1–3 figures with additional annotations and commentary

**Checkpoint:**
> `week21_exploratory_stats.ipynb`: full reproduction of NSFG analysis + 3 original questions answered with new visualizations. Each figure must have a caption explaining the statistical insight.

---

### Week 22: Probability Distributions & Hypothesis Testing

**Study:**
- Read Think Stats Chapters 4–6 (Distributions, Probability, Operations on Distributions)

**Practice:**
- Fit Normal, Exponential, and Pareto distributions to real data using MLE
- Implement a Kernel Density Estimator (KDE) from scratch using a Gaussian kernel — compare to `scipy.stats.gaussian_kde`

**Checkpoint:**
> `week22_distributions.ipynb`: MLE fitting for 3 distribution families, Q-Q plots for goodness-of-fit, and your KDE implementation vs. scipy with a visual comparison.

---

### Week 23: Hypothesis Testing, p-values & Effect Sizes

**Study:**
- Read Think Stats Chapters 7–9 (Relationships Between Variables, Estimation, Hypothesis Testing)

**Practice:**
- Implement a permutation test from scratch for two-sample comparison
- Implement bootstrap confidence intervals for the mean, median, and variance of a sample
- Compare your results to `scipy.stats.ttest_ind` and comment on differences

**Checkpoint:**
> `week23_hypothesis.ipynb`: permutation test vs. t-test comparison on real data, bootstrap CI for 3 statistics with 95% CI plots, and a written discussion of when permutation tests are preferable.

---

### Week 24: Regression, Correlation & Phase 4 Capstone

**Study:**
- Read Think Stats Chapters 10–12 (Linear Least Squares, Regression, Time Series)

**Practice:**
- Implement Ordinary Least Squares (OLS) regression from scratch using matrix algebra (no sklearn)
- Analyze the residuals for heteroskedasticity and autocorrelation using diagnostic plots

**Checkpoint:**
> `phase4_capstone.ipynb`: from-scratch OLS implementation, coefficient comparison with `statsmodels.OLS`, residual diagnostic plots (residuals vs. fitted, Q-Q of residuals, scale-location), and a written interpretation of model assumptions.

---

---

</details>

---

<details>
<summary><h2>Phase 5: Mathematical Statistics</h2></summary>

**Duration:** 12 Weeks
**Resources:**
- [John E. Freund's Mathematical Statistics with Applications, 8th Ed.](https://archive.org/details/johnefreundsmath0008mill)
- [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)

**Depth Assessment:** ~650 pages, rigorous mathematical statistics textbook. Heavy on proofs and derivations. Solutions manual is essential for self-study. Hardest phase so far — budget the full 12 hrs/week.
**Why this resource matters:** Formal estimation/testing theory and proof discipline.
**Weekly Structure (Strict Completion):** Week 25–32 topics + 4 end-of-chapter closure weeks.
**Phase Deliverables:** Typed/scanned proof archive + solved problem sets + capstone set.
**Phase Mastery Criteria:** Target chapter problems solved with full derivations; checkpoint thresholds met.

---

### Week 25: Probability & Mathematical Foundations (Chapters 1–2)

**Study:**
- Freund Chapters 1–2 (Introduction to Probability, Probability Distributions)
- Focus: formal definitions of sample space, sigma-algebras, probability measure

**Practice:**
- Solve 15 end-of-chapter problems from Chapters 1–2 without consulting solutions
- Verify answers against the Solutions Manual

**Checkpoint:**
> Paper problem set (scanned or typed in LaTeX): 15 solved problems, each with a complete proof and not just a numerical answer. Must score 12+/15 verified against solution manual.

---

### Week 26: Discrete & Continuous Distributions (Chapters 3–4)

**Study:**
- Freund Chapters 3–4 (Mathematical Expectation, Special Probability Distributions)
- Focus: Binomial, Multinomial, Hypergeometric, Negative Binomial, Poisson; Uniform, Gamma, Beta, Normal

**Practice:**
- Derive the MGF of the Gamma distribution from first principles
- Use MGFs to prove that the sum of independent Poisson RVs is Poisson

**Checkpoint:**
> LaTeX document with 2 full derivations (Gamma MGF + Poisson sum proof) and 10 solved textbook problems from Chapters 3–4.

---

### Week 27: Functions of Random Variables & Sampling Distributions (Chapters 5–6)

**Study:**
- Freund Chapters 5–6 (Functions of Random Variables, Sampling Distributions)
- Focus: Jacobian transformation method, Chi-squared, t, and F distributions

**Practice:**
- Derive the PDF of Y = X² where X ~ N(0,1) using the Jacobian method
- Simulate and verify the t-distribution as a ratio of Normal and Chi-squared variables

**Checkpoint:**
> `week27_transformations.ipynb`: Jacobian derivation (LaTeX), simulation verification of t-distribution with histogram overlay of theoretical PDF, and 8 solved problems from Chapters 5–6.

---

### Week 28: Point Estimation (Chapter 7)

**Study:**
- Freund Chapter 7 (Point Estimation)
- Focus: MLE, Method of Moments, UMVUE, Cramér-Rao Lower Bound, sufficiency

**Practice:**
- Derive MLE estimators for Normal (μ, σ²) and Exponential (λ) distributions
- Compute the Fisher Information and Cramér-Rao bound for the Bernoulli distribution; verify MLE achieves it

**Checkpoint:**
> LaTeX problem set: MLE derivations for 3 distributions, Fisher Information computation, and a written explanation of what "UMVUE" means and why it matters.

---

### Week 29: Interval Estimation (Chapter 8)

**Study:**
- Freund Chapter 8 (Interval Estimation)
- Focus: confidence intervals for mean, difference of means, variance; bootstrap CIs

**Practice:**
- Implement CIs for μ (known σ), μ (unknown σ), proportion, and difference of two means from scratch
- Demonstrate coverage probability: simulate 1,000 CIs and show 95% of them contain the true parameter

**Checkpoint:**
> `week29_confidence_intervals.ipynb`: all CI functions from scratch, coverage simulation with a visualization of 1,000 CI intervals plotted as horizontal lines (true parameter marked).

---

### Week 30: Hypothesis Testing I (Chapter 9)

**Study:**
- Freund Chapter 9 (Tests of Hypotheses)
- Focus: Neyman-Pearson lemma, UMP tests, likelihood ratio tests, Type I/II errors, power functions

**Practice:**
- Derive the Neyman-Pearson most powerful test for a simple hypothesis for Normal data
- Plot a power curve as a function of the true mean for a one-sample z-test

**Checkpoint:**
> LaTeX proof of the Neyman-Pearson lemma for a specific case, plus `week30_power_curve.ipynb` with power curves for multiple significance levels (α = 0.01, 0.05, 0.10) plotted together.

---

### Week 31: Hypothesis Testing II & Nonparametric Tests (Chapters 10–11)

**Study:**
- Freund Chapters 10–11 (Nonparametric Tests, Analysis of Variance)
- Focus: Wilcoxon, Mann-Whitney U, Kruskal-Wallis, one-way ANOVA

**Practice:**
- Implement the Wilcoxon signed-rank test from scratch; verify against `scipy.stats.wilcoxon`
- Run a one-way ANOVA on a real dataset; follow up with Tukey's HSD post-hoc test

**Checkpoint:**
> `week31_nonparametric.ipynb`: from-scratch Wilcoxon with comparison to scipy, ANOVA F-table generated manually vs. `scipy.stats.f_oneway`, and written interpretation of all results.

---

### Week 32: Regression & Correlation + Phase 5 Capstone (Chapters 12–13)

**Study:**
- Freund Chapters 12–13 (Regression and Correlation, Multiple Regression)
- Focus: Gauss-Markov theorem, F-test for overall significance, adjusted R², multicollinearity

**Practice:**
- Prove the Gauss-Markov theorem (OLS is BLUE) in a LaTeX document
- Implement multiple regression with all diagnostics from scratch; apply to a real dataset

**Checkpoint:**
> `phase5_capstone.ipynb`: multiple regression from scratch (matrix form), Gauss-Markov proof (LaTeX embedded), VIF computation for multicollinearity, and a written 1-page analysis of your regression results.

---

---

</details>

---

<details>
<summary><h2>Phase 6: Applied Multivariate Statistics</h2></summary>

**Duration:** 8 Weeks
**Resource:** [PSU STAT 505 — Applied Multivariate Statistical Analysis](https://online.stat.psu.edu/stat505/)
**Depth Assessment:** ~15 course lessons, moderately advanced. Builds directly on Phase 5. Heavy on matrix algebra applications.
**Why this resource matters:** Matrix-based multivariate inference used in advanced modeling.
**Weekly Structure (Strict Completion):** Week 33–37 topics + 3 full lesson/implementation weeks.
**Phase Deliverables:** MVN/Hotelling/MANOVA/PCA/FA/DA notebooks + capstone analysis.
**Phase Mastery Criteria:** Every lesson and exercise completed; from-scratch implementations validated.

---

### Week 33: Multivariate Normal & Hotelling's T²

**Study:**
- STAT 505 Lessons 1–3 (Introduction, Multivariate Normal Distribution, Hotelling's T²)

**Practice:**
- Generate 3D multivariate normal data; visualize pairwise scatter plots and the ellipsoidal contours
- Implement Hotelling's T² test from scratch and compare to a reference implementation

**Checkpoint:**
> `week33_mvn.ipynb`: 3D MVN simulation, pairwise plot matrix with density curves, Hotelling's T² from scratch with p-value comparison.

---

### Week 34: MANOVA & Profile Analysis

**Study:**
- STAT 505 Lessons 4–6 (MANOVA, Profile Analysis)

**Practice:**
- Apply one-way MANOVA to the Iris dataset using both `statsmodels` and a manual computation
- Perform profile analysis comparing group mean profiles; test for parallelism, levels, and flatness

**Checkpoint:**
> `week34_manova.ipynb`: MANOVA with full output table, profile analysis plots with hypothesis test results, and written interpretation of each test.

---

### Week 35: Principal Component Analysis (PCA) — Theory & Application

**Study:**
- STAT 505 Lessons 7–8 (Principal Component Analysis)
- Review SVD from Week 18 — connect to PCA via eigendecomposition of the covariance matrix

**Practice:**
- Implement PCA from scratch using the covariance matrix eigendecomposition
- Apply to a high-dimensional dataset (e.g., gene expression data); plot scree plot and biplot

**Checkpoint:**
> `week35_pca.ipynb`: from-scratch PCA with scree plot, explained variance table, biplot, and a comparison vs. `sklearn.decomposition.PCA`. Must explain why standardization matters for PCA.

---

### Week 36: Factor Analysis & Discriminant Analysis

**Study:**
- STAT 505 Lessons 9–11 (Factor Analysis, Discriminant Analysis)

**Practice:**
- Perform exploratory factor analysis (EFA) using `factor_analyzer`; compare Varimax vs. Promax rotation
- Implement Linear Discriminant Analysis (LDA) from scratch; apply to a classification problem and compute classification error

**Checkpoint:**
> `week36_factor_lda.ipynb`: EFA with factor loading heatmap, LDA from scratch with decision boundary visualization, confusion matrix, and comparison to `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.

---

### Week 37: Cluster Analysis, MANCOVA & Phase 6 Capstone

**Study:**
- STAT 505 Lessons 12–15 (Clustering, Canonical Correlation, MANCOVA)

**Practice:**
- Implement k-means from scratch and apply hierarchical clustering with Ward linkage
- Perform canonical correlation analysis on two sets of variables; interpret the canonical variates

**Checkpoint:**
> `phase6_capstone.ipynb`: full multivariate analysis pipeline on a chosen dataset — PCA for dimensionality reduction → LDA for classification → Cluster analysis for unsupervised grouping. Include a 1-page written report summarizing findings.

---

---

</details>

---

<details>
<summary><h2>Phase 7: Bayesian Statistics & Missing Data</h2></summary>

**Duration:** 9 Weeks
**Resources (Interwoven):**
- [Think Bayes, 2nd Ed. — Allen B. Downey](https://allendowney.github.io/ThinkBayes2/)
- [Flexible Imputation of Missing Data (FIMD) — Stef van Buuren](https://stefvanbuuren.name/fimd/)

**Depth Assessment:** Think Bayes ~250 pages (moderate, Python-based); FIMD ~300 pages (moderate-hard, R-based). Interwoven: Bayesian methods first, then missing data as a Bayesian application.
**Why this resource matters:** Bayesian reasoning plus robust handling of incomplete data.
**Weekly Structure (Strict Completion):** Week 38–43 sequence + 3 consolidation weeks.
**Phase Deliverables:** Bayesian notebooks, PyMC models, missing-data case study capstone.
**Phase Mastery Criteria:** Both resources completed with exercises/labs; posterior and imputation decisions justified.

---

### Week 38: Bayes' Theorem to Bayesian Inference

**Study:**
- Think Bayes Chapters 1–4 (Bayes' Theorem, Distributions, Estimating Counts, Estimating Rates)

**Practice:**
- Implement the "Cookie Problem," "M&M Problem," and "Monty Hall" using the `Pmf` class
- Build a Bayesian A/B test from scratch for conversion rate comparison

**Checkpoint:**
> `week38_bayesian_basics.ipynb`: all three classic problems solved with grid approximation, A/B test with posterior distributions plotted, and a written explanation of why the Bayesian result is more informative than a p-value.

---

### Week 39: Bayesian Hypothesis Testing & Estimation

**Study:**
- Think Bayes Chapters 5–8 (Odds, Addends, Mixtures, Simulation)

**Practice:**
- Compute Bayes Factors for comparing two models
- Implement a Bayesian linear regression with conjugate priors (Normal-Inverse-Gamma); visualize the posterior predictive distribution

**Checkpoint:**
> `week39_bayes_estimation.ipynb`: Bayes Factor computation, Bayesian linear regression from scratch with posterior predictive interval plot, and comparison of credible intervals vs. frequentist confidence intervals.

---

### Week 40: PyMC & Probabilistic Programming

**Study:**
- Think Bayes Chapters 9–13 (Decision Analysis, Approximate Bayesian Computation, Hierarchical Models)
- Install PyMC; read the official "Getting Started" documentation

**Practice:**
- Implement a hierarchical model for school test scores using PyMC (8-schools problem)
- Use MCMC sampling; diagnose convergence with R-hat statistics and trace plots

**Checkpoint:**
> `week40_pymc_hierarchical.ipynb`: PyMC hierarchical model, trace plots, posterior predictive check plots, and written interpretation of partial pooling vs. no-pooling.

---

### Week 41: Missing Data — Mechanisms & Single Imputation

**Study:**
- FIMD Chapters 1–3 (Introduction, Multiple Imputation, Univariate Missing Data)
- Understand MCAR, MAR, MNAR

**Practice:**
- Simulate all three missing data mechanisms on a real dataset
- Apply and compare: complete case analysis, mean imputation, regression imputation — assess bias in each

**Checkpoint:**
> `week41_missing_data.ipynb`: simulation of MCAR/MAR/MNAR with empirical bias tables, comparison of 3 imputation strategies, and a written explanation of when each missing data mechanism produces biased estimates.

---

### Week 42: Multiple Imputation (MICE)

**Study:**
- FIMD Chapters 4–6 (Multivariate Missing Data, Analysis of Imputed Data, Imputation in Practice)

**Practice:**
- Implement the MICE algorithm conceptually step-by-step; use `miceforest` or `sklearn.impute.IterativeImputer` in Python
- Apply multiple imputation (m=20) to a dataset with 30%+ missingness; pool results using Rubin's rules

**Checkpoint:**
> `week42_mice.ipynb`: MICE imputation with m=20, Rubin's rules pooling for regression coefficients, trace plots of imputation convergence, and comparison of MICE estimates vs. listwise deletion.

---

### Week 43: Advanced Missing Data & Phase 7 Capstone

**Study:**
- FIMD Chapters 7–9 (Sensitive Variables, Longitudinal Data, Multilevel Data)
- Think Bayes Chapters 14–19 (review and advanced topics)

**Practice:**
- Apply multiple imputation to a longitudinal dataset (e.g., panel data with attrition)
- Build a Bayesian model that jointly handles missing data and inference (selection model)

**Checkpoint:**
> `phase7_capstone.ipynb`: end-to-end Bayesian + missing data pipeline — MICE imputation → PyMC model fitted on imputed data → pooled posterior inference. Written 1-page discussion of uncertainty sources.

---

---

</details>

---

<details>
<summary><h2>Phase 8: Statistical Learning with Python (ISLP)</h2></summary>

**Duration:** 12 Weeks
**Resource:** [An Introduction to Statistical Learning with Applications in Python (ISLP)](https://www.statlearning.com/)
**Depth Assessment:** ~600 pages, moderate-hard, with Python labs in every chapter. The gold-standard ML textbook for statisticians. Must complete every lab.
**Why this resource matters:** Canonical applied ML with statistical rigor.
**Weekly Structure (Strict Completion):** Week 44–51 chapter path + 4 lab/exercise completion weeks.
**Phase Deliverables:** Full ISLP lab portfolio + from-scratch algorithm notebooks + capstone.
**Phase Mastery Criteria:** Every lab completed; chapter exercises complete; diagnostics and interpretation complete.

---

### Week 44: Linear Regression — Theory & Lab

**Study:**
- ISLP Chapters 1–3 (Statistical Learning, Linear Regression) — full chapters and all Python labs

**Practice:**
- Complete the Chapter 3 Python lab in full (Boston dataset and beyond)
- Implement Ridge and Lasso regression from scratch using coordinate descent

**Checkpoint:**
> `week44_islp_linreg.ipynb`: full Chapter 3 lab + from-scratch Ridge/Lasso with coefficient paths plotted vs. λ.

---

### Week 45: Classification — Logistic Regression, LDA, QDA, KNN

**Study:**
- ISLP Chapter 4 (Classification) — full chapter and Python lab

**Practice:**
- Implement logistic regression from scratch using gradient descent (not Newton's method)
- Apply all 4 classifiers (LR, LDA, QDA, KNN) to a dataset; produce a comparative confusion matrix table

**Checkpoint:**
> `week45_classification.ipynb`: from-scratch logistic regression with learning curve, all 4 classifiers with a comparative table of Accuracy, Precision, Recall, F1, and AUC.

---

### Week 46: Resampling Methods — Cross-Validation & Bootstrap

**Study:**
- ISLP Chapter 5 (Resampling Methods) — full chapter and Python lab

**Practice:**
- Implement k-fold cross-validation from scratch
- Implement the bootstrap for estimating the standard error of any statistic (passed as a function)

**Checkpoint:**
> `week46_resampling.ipynb`: from-scratch k-fold CV with hyperparameter tuning demonstration, bootstrap SE for 3 statistics, and comparison of CV estimates vs. test set estimates.

---

### Week 47: Model Selection & Regularization

**Study:**
- ISLP Chapter 6 (Linear Model Selection and Regularization) — full chapter and lab
- Focus: Best subset selection, Ridge, Lasso, PCR, PLS

**Practice:**
- Implement best subset selection via exhaustive search for p ≤ 15
- Tune Ridge and Lasso via cross-validation; plot validation curve showing optimal λ

**Checkpoint:**
> `week47_regularization.ipynb`: subset selection with AIC/BIC/adjusted-R² comparison, regularization path for Ridge and Lasso, optimal λ selection via 10-fold CV.

---

### Week 48: Nonlinear Models & Splines

**Study:**
- ISLP Chapter 7 (Moving Beyond Linearity) — polynomial regression, step functions, splines, GAMs

**Practice:**
- Implement a cubic spline from scratch with knots at specified quantiles
- Fit a GAM to a dataset; plot partial dependence plots for each predictor

**Checkpoint:**
> `week48_nonlinear.ipynb`: from-scratch cubic spline vs. `scipy.interpolate.CubicSpline`, GAM with partial dependence plots, and a written explanation of the bias-variance trade-off for polynomial degree.

---

### Week 49: Decision Trees & Ensemble Methods

**Study:**
- ISLP Chapters 8 (Tree-Based Methods) — full chapter and lab
- Topics: CART, Random Forests, Boosting (AdaBoost, Gradient Boosting), Bagging

**Practice:**
- Implement CART (classification tree) from scratch using recursive binary splitting and Gini impurity
- Train Random Forest and Gradient Boosting on a real dataset; plot feature importances

**Checkpoint:**
> `week49_trees.ipynb`: from-scratch CART with tree visualization (using `graphviz`), RF and GBM comparison table, and feature importance bar chart for both ensemble methods.

---

### Week 50: Support Vector Machines

**Study:**
- ISLP Chapter 9 (Support Vector Machines) — full chapter and lab
- Understand: maximal margin classifier, soft-margin SVM, kernel trick (RBF, polynomial)

**Practice:**
- Implement a hard-margin SVM using `cvxpy` (quadratic programming formulation)
- Compare linear, polynomial, and RBF kernels on a non-linearly separable dataset; plot decision boundaries

**Checkpoint:**
> `week50_svm.ipynb`: hard-margin SVM from cvxpy, decision boundary plots for all 3 kernels, and a written explanation of what the kernel trick achieves mathematically.

---

### Week 51: Unsupervised Learning & Phase 8 Capstone

**Study:**
- ISLP Chapters 10–13 (Deep Learning overview, Survival Analysis, Multiple Testing, Unsupervised Learning)
- Focus primarily on Chapter 12 (PCA, K-Means, Hierarchical Clustering) and Chapter 13

**Practice:**
- Full pipeline: PCA → K-Means → Hierarchical clustering on a high-dimensional dataset
- Apply multiple testing correction (Bonferroni, BH-FDR) to a gene expression example

**Checkpoint:**
> `phase8_capstone.ipynb`: complete ML pipeline from data → feature engineering → model comparison → evaluation. Must test at least 5 model classes, use proper CV, and produce a final model card with performance metrics and limitations.

---

---

</details>

---

<details>
<summary><h2>Phase 9: Data Mining</h2></summary>

**Duration:** 10 Weeks
**Resource:** [Data Mining: Concepts and Techniques, 3rd Ed. — Jiawei Han](https://hanj.cs.illinois.edu/bk3/)
**Depth Assessment:** ~700 pages, moderate difficulty. Breadth-first treatment of mining techniques. Builds directly on Phase 8.
**Why this resource matters:** Pattern mining, clustering, outliers, stream/web mining breadth.
**Weekly Structure (Strict Completion):** Week 52–58 topics + 3 full exercise/project completion weeks.
**Phase Deliverables:** Mining implementations + full data mining project.
**Phase Mastery Criteria:** All chapter work completed; final project covers full mining lifecycle.

---

### Week 52: Data Preprocessing & Exploration

**Study:**
- Han Chapters 1–3 (Introduction, Getting to Know Your Data, Data Preprocessing)

**Practice:**
- Build a comprehensive data preprocessing pipeline class in Python: handles normalization, standardization, discretization, and outlier detection
- Apply to a new real-world dataset and produce a "data health report"

**Checkpoint:**
> `week52_data_preprocessing.ipynb`: reusable `DataPreprocessor` class with unit tests, applied to a new dataset with a written data health report.

---

### Week 53: Frequent Pattern Mining — Apriori & FP-Growth

**Study:**
- Han Chapters 5–7 (Mining Frequent Patterns, Association Rules, Advanced Pattern Mining)

**Practice:**
- Implement the Apriori algorithm from scratch
- Apply FP-Growth (via `mlxtend`) to a retail transaction dataset; discover and interpret top 10 association rules

**Checkpoint:**
> `week53_association_rules.ipynb`: from-scratch Apriori verified against `mlxtend.frequent_patterns.apriori`, FP-Growth rules with support/confidence/lift table, and practical interpretation of the top rules.

---

### Week 54: Classification — Advanced Techniques

**Study:**
- Han Chapters 8–9 (Classification, Advanced Classification)
- Focus: decision tree induction (ID3, C4.5, CART differences), Bayesian classification, rule-based, SVM, ensemble methods from a data mining perspective

**Practice:**
- Implement ID3 (using Information Gain) from scratch; compare split criteria with Gini (CART)
- Build a comparison table of 6 classifiers on 3 datasets using identical preprocessing and CV

**Checkpoint:**
> `week54_classification_dm.ipynb`: ID3 from scratch, 6-classifier comparison table across 3 datasets, and a written analysis of when each classifier family performs best.

---

### Week 55: Clustering — K-Means, DBSCAN & Hierarchical

**Study:**
- Han Chapter 10 (Cluster Analysis)
- Focus: partitioning, hierarchical, density-based, grid-based, model-based methods

**Practice:**
- Implement DBSCAN from scratch using only NumPy
- Compare K-Means, DBSCAN, and Agglomerative clustering on 3 synthetic datasets (blobs, moons, circles)

**Checkpoint:**
> `week55_clustering.ipynb`: from-scratch DBSCAN with comparison against `sklearn.cluster.DBSCAN`, side-by-side visualization of all 3 algorithms on all 3 datasets (9-panel figure), and Silhouette scores.

---

### Week 56: Outlier Detection

**Study:**
- Han Chapter 12 (Outlier Detection)
- Topics: statistical, distance-based, density-based (LOF), isolation-based (Isolation Forest)

**Practice:**
- Implement Local Outlier Factor (LOF) from scratch
- Apply LOF, Isolation Forest, and One-Class SVM to a real anomaly detection dataset (e.g., credit card fraud); compare AUC-ROC

**Checkpoint:**
> `week56_outlier.ipynb`: from-scratch LOF, AUC-ROC comparison of all 3 methods, precision-recall curve for imbalanced data, and written discussion of the computational complexity of each approach.

---

### Week 57: Mining Data Streams & Web Data

**Study:**
- Han Chapters 13–14 (Mining Data Streams, Mining Social Networks)
- Topics: sliding window, Flajolet-Martin, Bloom filter, PageRank

**Practice:**
- Implement a Bloom Filter from scratch for approximate set membership
- Implement the Flajolet-Martin algorithm for cardinality estimation; test on a stream of 1M items

**Checkpoint:**
> `week57_streaming.ipynb`: Bloom Filter class with false positive rate analysis, Flajolet-Martin cardinality estimation vs. exact count, and performance benchmark.

---

### Week 58: Phase 9 Capstone — Full Data Mining Project

**Study:**
- Han Chapters 4, 15–16 (Data Warehousing, Spatial Mining, Multimedia Mining — skim for awareness)

**Practice:**
- End-to-end data mining project on a dataset of your choice (recommend: [UCI ML Repository](https://archive.ics.uci.edu/))
- Must include: preprocessing → association rule mining → 3+ classification models → clustering → outlier removal → final model selection

**Checkpoint:**
> `phase9_capstone/` directory with a reproducible pipeline, a written report (~1000 words), and a results table. The project must tell a coherent data mining story from question to conclusion.

---

---

</details>

---

<details>
<summary><h2>Phase 10: Classical Machine Learning</h2></summary>

**Duration:** 16 Weeks
**Resources (Interwoven):**
- [Pattern Recognition and Machine Learning (PRML) — Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Interpretable Machine Learning — Christoph Molnar](https://christophm.github.io/interpretable-ml-book/)

**Depth Assessment:** PRML ~700 pages, high mathematical rigor — the hardest phase yet. IML ~300 pages, moderate. Strategy: PRML for theory, IML for interpretation of the same models.
**Why this resource matters:** Deep probabilistic ML theory plus interpretability discipline.
**Weekly Structure (Strict Completion):** Week 59–68 track + 6 derivation/code completion weeks.
**Phase Deliverables:** PRML derivation notes, from-scratch implementations, interpretation reports, capstone system.
**Phase Mastery Criteria:** Selected PRML math/implementations completed; interpretation methods correctly applied.

---

### Week 59: Polynomial Curve Fitting & Probability Theory (PRML Ch. 1)

**Study:**
- PRML Chapter 1 (Introduction) — all sections
- IML Chapter 1–2 (Introduction, Interpretability)

**Practice:**
- Implement polynomial regression with regularization (PRML Figure 1.4 reproduction)
- Plot overfitting progression for M=0,1,3,9 polynomials with 10 training points

**Checkpoint:**
> `week59_prml_intro.ipynb`: exact reproduction of Bishop Figure 1.4 and 1.5 from scratch (no sklearn), written explanation of the bias-variance trade-off in terms of the PRML framework.

---

### Week 60: Probability Distributions (PRML Ch. 2)

**Study:**
- PRML Chapter 2 (Probability Distributions) — Gaussian, Bernoulli, Multinomial, Dirichlet, Exponential family

**Practice:**
- Derive the MLE for the Dirichlet-Multinomial conjugate pair
- Implement the EM algorithm for a 1D Gaussian Mixture Model from scratch

**Checkpoint:**
> `week60_distributions.ipynb`: GMM-EM from scratch with visualization of convergence (log-likelihood vs. iteration), compared to `sklearn.mixture.GaussianMixture`.

---

### Week 61: Linear Models for Regression (PRML Ch. 3)

**Study:**
- PRML Chapter 3 (Linear Models for Regression)
- Focus: Bayesian linear regression, evidence approximation, Automatic Relevance Determination

**Practice:**
- Implement Bayesian linear regression from scratch with predictive distribution
- Plot the posterior over weights and the predictive uncertainty bands

**Checkpoint:**
> `week61_bayesian_linreg.ipynb`: Bayesian linear regression from scratch, posterior weight distribution visualization, predictive uncertainty bands at different data sizes.

---

### Week 62: Linear Models for Classification (PRML Ch. 4)

**Study:**
- PRML Chapter 4 (Linear Models for Classification)
- Focus: generative vs. discriminative, Laplace approximation for logistic regression, multi-class

**Practice:**
- Implement softmax regression (multinomial logistic regression) from scratch with Newton-Raphson optimizer
- Apply to MNIST (first 3 classes); visualize decision regions

**Checkpoint:**
> `week62_linear_classification.ipynb`: from-scratch softmax regression with Newton-Raphson, decision boundary plots, and comparison to `sklearn.linear_model.LogisticRegression`.

---

### Week 63: Neural Networks (PRML Ch. 5)

**Study:**
- PRML Chapter 5 (Neural Networks) — feed-forward nets, backpropagation, regularization

**Practice:**
- Implement a 2-layer neural network with backpropagation **entirely from scratch** using only NumPy
- Train on a classification dataset; implement gradient checking to verify your backprop

**Checkpoint:**
> `week63_neural_net_scratch.ipynb`: fully from-scratch neural network with gradient checking (must pass), training/validation loss curves, and accuracy comparison to a comparable `sklearn.neural_network.MLPClassifier`.

---

### Week 64: Kernel Methods & SVMs (PRML Ch. 6–7)

**Study:**
- PRML Chapters 6–7 (Kernel Methods, Sparse Kernel Machines)
- Focus: Gaussian Process regression, SVM dual formulation, relevance vector machine

**Practice:**
- Implement Gaussian Process Regression from scratch (RBF kernel)
- Plot GP posterior mean and 2σ uncertainty bands; vary the kernel hyperparameters and observe the effect

**Checkpoint:**
> `week64_gp.ipynb`: from-scratch GP regression with uncertainty bands, kernel hyperparameter sensitivity analysis (3-panel plot), and a written explanation of what each hyperparameter controls.

---

### Week 65: Graphical Models & EM Algorithm (PRML Ch. 8–9)

**Study:**
- PRML Chapters 8–9 (Graphical Models, Mixture Models and EM)

**Practice:**
- Implement the EM algorithm for a full Gaussian Mixture Model (K components, D dimensions)
- Visualize the EM convergence on 2D data with animated steps (each iteration = one frame)

**Checkpoint:**
> `week65_em_gmm.ipynb`: full multivariate GMM-EM from scratch (NumPy only), animated convergence saved as GIF, BIC/AIC model selection for K=1..8.

---

### Week 66: Sampling Methods & Approximate Inference (PRML Ch. 11)

**Study:**
- PRML Chapter 11 (Sampling Methods)
- Focus: Rejection sampling, Importance sampling, Metropolis-Hastings, Gibbs sampling

**Practice:**
- Implement Metropolis-Hastings from scratch for sampling from a 2D bimodal distribution
- Visualize the Markov chain trajectory and the final approximated distribution

**Checkpoint:**
> `week66_mcmc.ipynb`: MH sampler from scratch, chain trajectory plot, sample histogram vs. true distribution, and written discussion of acceptance rate tuning.

---

### Week 67: Model Interpretability — SHAP, LIME & PDP

**Study:**
- IML Chapters 4–8 (Regression Models, Linear Models, Decision Trees, RuleFit, Other Interpretable Models)
- IML Chapters 9–10 (Model-Agnostic Methods, Partial Dependence Plots)

**Practice:**
- Train a complex model (GBM) on a real dataset; apply SHAP values (using `shap` library)
- Produce: SHAP summary plot, SHAP waterfall plot for 3 individual predictions, PDP for top 3 features

**Checkpoint:**
> `week67_interpretability.ipynb`: full interpretability analysis with SHAP, LIME, and PDP. Must include both global and local explanations, and a written section on when each method should be preferred.

---

### Week 68: Phase 10 Capstone — From-Scratch Classical ML System

**Study:**
- PRML Chapter 12 (Continuous Latent Variables — PCA and probabilistic PCA)
- IML Chapter 11–12 (Neural Network Interpretation, Counter-factual Explanations)

**Practice:**
- Build a complete classical ML system from scratch: includes data pipeline, feature engineering, 3 model classes (one probabilistic), cross-validation, hyperparameter tuning, and interpretability analysis

**Checkpoint:**
> `phase10_capstone/`: modular Python package (not a notebook) with proper `__init__.py`, docstrings, and unit tests. Must include a `README.md` and reproduce your best result with a single `python run.py` command.

---

---

</details>

---

<details>
<summary><h2>Phase 11: Elements of Statistical Learning</h2></summary>

**Duration:** 14 Weeks
**Resource:** [The Elements of Statistical Learning, 2nd Ed. — Hastie, Tibshirani, Friedman](https://hastie.su.domains/ElemStatLearn/)
**Depth Assessment:** ~750 pages, extreme mathematical rigor. ESL is the graduate-level complement to ISLP. After Phase 10 (PRML), you have the prerequisite foundation. Emphasis on theory and derivations.
**Why this resource matters:** Graduate-level theory depth and statistical learning foundations.
**Weekly Structure (Strict Completion):** Week 69–76 sequence + 6 proof/replication weeks.
**Phase Deliverables:** Chapter derivation dossier + computational replications + capstone.
**Phase Mastery Criteria:** Full chapter progression completed; algorithm behavior understood at derivation level.

---

### Week 69: ESL Chapters 1–4 — Linear Methods

**Study:**
- ESL Chapters 1–4 (Introduction, Overview, Linear Regression, Linear Methods for Classification)
- Compare ESL's treatment of ridge regression and lasso to ISLP's — note the deeper mathematical treatment

**Practice:**
- Derive the ridge regression shrinkage in terms of SVD — show how ridge shrinks singular values
- Implement Linear Discriminant Analysis using Fisher's criterion (from-scratch, matrix form)

**Checkpoint:**
> LaTeX document with SVD-based ridge derivation + `week69_esl.ipynb` with from-scratch LDA and comparison to sklearn.

---

### Week 70: Basis Expansions, Splines & Smoothing (Ch. 5)

**Study:**
- ESL Chapter 5 (Basis Expansions and Regularization) — all sections including smoothing splines, RKHS

**Practice:**
- Implement natural cubic splines from scratch using the truncated power basis
- Implement a smoothing spline via the penalized least squares criterion; vary the smoothing parameter

**Checkpoint:**
> `week70_splines.ipynb`: natural cubic spline from scratch, smoothing spline with λ sensitivity plot (equivalent degrees of freedom vs. λ), and cross-validated optimal λ selection.

---

### Week 71: Kernel Smoothing & Model Assessment (Ch. 6–7)

**Study:**
- ESL Chapters 6–7 (Kernel Smoothing Methods, Model Assessment and Selection)
- Focus: Nadaraya-Watson estimator, local polynomial regression, AIC, BIC, MDL, VC dimension

**Practice:**
- Implement the Nadaraya-Watson kernel estimator from scratch
- Implement a leave-one-out CV estimator using the hat matrix shortcut (avoid refitting)

**Checkpoint:**
> `week71_kernel_cv.ipynb`: NW estimator with bandwidth sensitivity plot, LOO-CV via hat matrix (verify it matches brute-force LOO), and a written explanation of the optimism of in-sample error.

---

### Week 72: Additive Models & Boosting (Ch. 9–10)

**Study:**
- ESL Chapters 9–10 (Additive Models, Trees, Boosting)
- Focus: Gradient Boosting derivation (Friedman's view), AdaBoost as exponential loss

**Practice:**
- Implement Gradient Boosting (for regression) from scratch using decision stumps as base learners
- Show the equivalence between AdaBoost and forward stagewise additive modeling with exponential loss

**Checkpoint:**
> `week72_boosting.ipynb`: from-scratch gradient boosting with learning curve, proof of AdaBoost-exponential loss equivalence (LaTeX), and comparison vs. `sklearn.ensemble.GradientBoostingRegressor`.

---

### Week 73: Neural Networks & Radial Basis Functions (Ch. 11–12)

**Study:**
- ESL Chapters 11–12 (Neural Networks, Support Vector Machines and Flexible Discriminants)
- Note: compare ESL's NN treatment to PRML Ch. 5 — ESL is more optimization-focused

**Practice:**
- Implement a Radial Basis Function (RBF) network from scratch: learn centers via k-means, train output weights via least squares
- Compare the RBF network to a standard neural network on a regression task

**Checkpoint:**
> `week73_rbf.ipynb`: from-scratch RBF network, comparison with MLP, and a written analysis of inductive biases of each architecture.

---

### Week 74: Prototype Methods, Unsupervised Learning & EM (Ch. 13–14)

**Study:**
- ESL Chapters 13–14 (Prototype Methods, Unsupervised Learning)
- Focus: k-means, k-medoids, LVQ, hierarchical methods, ICA, PCA revisited

**Practice:**
- Implement K-Medoids (PAM algorithm) from scratch
- Implement FastICA from scratch; apply to cocktail party problem (blind source separation)

**Checkpoint:**
> `week74_kmedoids_ica.ipynb`: K-Medoids from scratch vs. K-Means on non-Euclidean data, FastICA demonstration on mixed audio signals with before/after plots.

---

### Week 75: Random Forests & Ensemble Methods (Ch. 15–16)

**Study:**
- ESL Chapters 15–16 (Random Forests, Ensemble Learning)
- Focus: variance reduction mechanism, out-of-bag error, stacking, super-learner

**Practice:**
- Implement Random Forest from scratch (bag of CART trees + random feature subsets)
- Implement a 2-level stacking ensemble with 3 base learners and a meta-learner

**Checkpoint:**
> `week75_ensemble.ipynb`: from-scratch Random Forest vs. `sklearn.ensemble.RandomForestClassifier`, stacking ensemble with out-of-fold predictions, and a written analysis of when stacking adds value.

---

### Week 76: High-Dimensional Problems & Phase 11 Capstone (Ch. 17–18)

**Study:**
- ESL Chapters 17–18 (Undirected Graphical Models, High-Dimensional Problems)
- Focus: graphical lasso, Dantzig selector, compressed sensing intuition

**Practice:**
- Implement the Graphical Lasso (glasso) algorithm for sparse precision matrix estimation
- Apply to financial return data; interpret the resulting conditional independence graph

**Checkpoint:**
> `phase11_capstone.ipynb`: graphical lasso on real financial data with network visualization, and a 500-word written synthesis comparing ESL's and ISLP's treatment of the same topics, noting key mathematical extensions in ESL.

---

---

</details>

---

<details>
<summary><h2>Phase 12: Deep Learning</h2></summary>

**Duration:** 20 Weeks
**Resources (Interwoven):**
- [Dive into Deep Learning (D2L.ai)](https://d2l.ai/)
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Deep Learning Book — Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/)

**Depth Assessment:** D2L.ai = massive (~1000+ pages with code); DL Book = ~800 pages; Illustrated Transformer = short but essential. Strategy: DL Book for theory, D2L for code-first implementation. This is the longest and most demanding phase.
**Why this resource matters:** Full modern deep learning stack from fundamentals to transformers and generative models.
**Weekly Structure (Strict Completion):** Week 77–87 path + 8 lab/reproduction/capstone-hardening weeks.
**Phase Deliverables:** From-scratch DL modules, transformer/generative notebooks, major capstone.
**Phase Mastery Criteria:** Assigned chapters/labs complete; models trained, evaluated, and explained rigorously.

---

### Week 77: Linear Neural Networks & MLP from Scratch

**Study:**
- DL Book Ch. 6 (Deep Feedforward Networks) — full chapter
- D2L Chapters 1–4 (Introduction, Preliminaries, Linear Networks, MLPs)

**Practice:**
- Implement a fully functional MLP from scratch in PyTorch **without using `nn.Module`** (only `torch.Tensor` and autograd)
- Train on MNIST; achieve >97% test accuracy

**Checkpoint:**
> `week77_mlp_scratch.ipynb`: raw-PyTorch MLP, training/validation loss curves, confusion matrix, and a written explanation of how `torch.autograd` computes gradients.

---

### Week 78: Optimization, Initialization & Regularization

**Study:**
- DL Book Ch. 7–8 (Regularization, Optimization for Deep Learning)
- D2L Ch. 5–6 (Builders' Guide, CNNs intro)

**Practice:**
- Implement and compare SGD, Momentum, RMSProp, and Adam optimizers from scratch
- Demonstrate the effect of initialization (Xavier vs. He vs. random) on training dynamics

**Checkpoint:**
> `week78_optimization.ipynb`: 4 optimizer implementations from scratch, side-by-side convergence plots on the same task, and initialization experiment with training loss curves.

---

### Week 79: Convolutional Neural Networks (CNNs)

**Study:**
- DL Book Ch. 9 (Convolutional Networks)
- D2L Ch. 7–8 (CNNs, Modern CNNs — AlexNet, VGG, NiN, GoogLeNet, ResNet)

**Practice:**
- Implement a 2D convolution operation from scratch using only NumPy
- Build and train a ResNet-20 from scratch in PyTorch on CIFAR-10; achieve >90% test accuracy

**Checkpoint:**
> `week79_cnn.ipynb`: from-scratch convolution, ResNet-20 training with final test accuracy and training curves, and feature map visualizations for 3 layers.

---

### Week 80: Recurrent Neural Networks & LSTMs

**Study:**
- DL Book Ch. 10 (Sequence Modeling: Recurrent and Recursive Nets)
- D2L Ch. 9 (RNNs) — all sections

**Practice:**
- Implement an LSTM cell from scratch using only `torch.Tensor` operations (no `nn.LSTM`)
- Build a character-level language model; generate text samples after training

**Checkpoint:**
> `week80_rnn_lstm.ipynb`: from-scratch LSTM cell verified against `nn.LSTM`, character LM with generated text samples at epoch 1, 10, 50, and final perplexity metric.

---

### Week 81: Attention Mechanism & The Transformer

**Study:**
- Read [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — multiple times, diagram by diagram
- D2L Ch. 11 (Attention Mechanisms and Transformers) — all sections
- DL Book Ch. 12 (Applications)

**Practice:**
- Implement scaled dot-product attention from scratch
- Implement multi-head attention from scratch
- Implement a full Transformer encoder block from scratch (no `nn.TransformerEncoder`)

**Checkpoint:**
> `week81_transformer.ipynb`: from-scratch attention mechanisms, a Transformer encoder block tested on a simple sequence task, and an annotated diagram (matplotlib) of your implementation matching the Illustrated Transformer's visualization.

---

### Week 82: Full Transformer & Pre-training Objectives

**Study:**
- D2L Ch. 15 (Natural Language Processing: Pretraining)
- Research papers: "Attention Is All You Need" (Vaswani et al., 2017), "BERT" (Devlin et al., 2019) — read abstracts and architecture sections

**Practice:**
- Implement a complete encoder-decoder Transformer for machine translation from scratch
- Train on a small bilingual dataset (e.g., English-French subset from Tatoeba)

**Checkpoint:**
> `week82_seq2seq_transformer.ipynb`: complete Transformer from scratch, training on translation task, BLEU score computation, and attention heatmap visualization.

---

### Week 83: Generative Models — VAEs & GANs

**Study:**
- DL Book Ch. 20 (Deep Generative Models)
- D2L Ch. 20 (Generative Adversarial Networks)

**Practice:**
- Implement a Variational Autoencoder (VAE) from scratch; train on MNIST
- Implement a DC-GAN from scratch; train on CelebA or CIFAR-10

**Checkpoint:**
> `week83_generative.ipynb`: VAE with latent space visualization (t-SNE of latent codes), image reconstructions, GAN with FID score estimation and generated samples grid.

---

### Week 84: Normalization, Modern Architectures & Tricks

**Study:**
- D2L Ch. 8 (Modern CNNs — DenseNet, BatchNorm)
- DL Book Ch. 8 (Optimization for Training Deep Models — second pass)
- Topics: Batch Norm, Layer Norm, Dropout, Label Smoothing, Mixed Precision Training

**Practice:**
- Implement Batch Normalization from scratch (both forward and backward pass)
- Ablation study: train the same model with and without BatchNorm, Dropout, and Label Smoothing — record performance

**Checkpoint:**
> `week84_normalization.ipynb`: from-scratch BatchNorm with gradient verification, ablation table (8 combinations of BatchNorm/Dropout/LabelSmoothing ON/OFF), and written conclusions.

---

### Week 85: Transfer Learning & Fine-Tuning

**Study:**
- D2L Ch. 14 (Computer Vision — Fine-Tuning, Object Detection intro)

**Practice:**
- Fine-tune a pre-trained ResNet-50 (from `torchvision`) on a custom 5-class image dataset
- Compare: full fine-tuning vs. head-only vs. last-2-layers; plot accuracy vs. epochs for all 3

**Checkpoint:**
> `week85_transfer.ipynb`: 3-strategy fine-tuning comparison table and learning curves, Grad-CAM visualizations for 5 sample predictions, and a written explanation of when to freeze vs. unfreeze layers.

---

### Week 86: Reinforcement Learning Fundamentals

**Study:**
- DL Book Ch. 17 (Monte Carlo Methods — skim as RL background)
- D2L Ch. 17 (Reinforcement Learning) — Markov Decision Processes, Q-Learning, Policy Gradient

**Practice:**
- Implement Deep Q-Network (DQN) from scratch; apply to CartPole-v1 (`gymnasium`)
- Implement REINFORCE (policy gradient) from scratch; compare convergence to DQN on CartPole

**Checkpoint:**
> `week86_rl.ipynb`: DQN and REINFORCE from scratch, training reward curves for both, and a written comparison of value-based vs. policy-based RL methods.

---

### Week 87: Scaling Laws, LLMs & Phase 12 Capstone

**Study:**
- D2L Ch. 11 (remaining sections on large-scale pre-training)
- Read: "Scaling Laws for Neural Language Models" (Kaplan et al., 2020) — key findings
- DL Book: Appendix review

**Practice:**
- Fine-tune a small pre-trained language model (e.g., GPT-2 small via HuggingFace) on a domain-specific dataset
- Implement a simple RAG (Retrieval-Augmented Generation) pipeline

**Checkpoint:**
> `phase12_capstone/`: a complete end-to-end deep learning project — choose one: (a) image classification system with custom dataset, (b) text generation with fine-tuned LM, or (c) multimodal system. Must include training code, evaluation, model card, and a 1-page technical report.

---

---

</details>

---

<details>
<summary><h2>Phase 13: R for Data Science</h2></summary>

**Duration:** 7 Weeks
**Resource:** [R for Data Science, 2nd Ed. — Hadley Wickham](https://r4ds.hadley.nz/)
**Depth Assessment:** ~500 pages, moderate difficulty. Deliberately placed here — you already know the statistical concepts. The goal is to become proficient in the R ecosystem for statistical computing.
**Why this resource matters:** Statistical computing fluency in the R/tidyverse ecosystem.
**Weekly Structure (Strict Completion):** Week 88–92 + 2 completion/polish weeks.
**Phase Deliverables:** R Markdown/Quarto analyses + reproducible R capstone.
**Phase Mastery Criteria:** All chapter tasks complete; reproducible workflow from import to reporting.

---

### Week 88: R Basics, Tidyverse & Data Import

**Study:**
- R4DS Chapters 1–9 (Introduction, Workflow Basics, Data Transformation, Workflow Scripts, Data Tidying, Data Import)
- Install R 4.3+, RStudio, and `tidyverse`

**Practice:**
- Reproduce your Phase 1 Capstone analysis in R using `dplyr` and `readr` — compare the pandas vs. tidyverse syntax
- Create a `tibble`, perform GroupBy aggregation with `group_by()` + `summarise()`

**Checkpoint:**
> `week88_r_basics.Rmd`: R Markdown document with the Phase 1 analysis reproduced in R, a comparison table of pandas vs. dplyr syntax for 10 common operations.

---

### Week 89: Data Visualization with ggplot2

**Study:**
- R4DS Chapters 10–12 (Layers, Exploratory Data Analysis, Communication with ggplot2)

**Practice:**
- Reproduce 6 of your Phase 1 matplotlib figures using `ggplot2` — comment on aesthetic differences
- Create a publication-quality `ggplot2` figure with custom theme, annotations, and a multi-panel layout using `patchwork`

**Checkpoint:**
> `week89_ggplot2.Rmd`: 6 reproduced figures + 1 publication-quality multi-panel figure. Each figure must have a caption. Export as PDF.

---

### Week 90: Functions, Iteration & Functional Programming

**Study:**
- R4DS Chapters 13–27 (Logical Vectors, Numbers, Strings, Regular Expressions, Factors, Dates, Missing Values, Joins, Strings advanced)

**Practice:**
- Write a suite of custom R functions that replicate your Python preprocessing pipeline from Phase 9
- Use `purrr::map()` family for iteration — compare to `for` loops and `lapply`

**Checkpoint:**
> `week90_r_programming.Rmd`: custom preprocessing pipeline in R, a benchmark comparing `for` loop vs. `lapply` vs. `purrr::map` for 5 operations.

---

### Week 91: Statistical Modeling in R — `lm`, `glm`, `tidymodels`

**Study:**
- R4DS Chapters 28–29 (Modeling intro — note: R4DS 2e defers to tidymodels)
- Supplement: [tidymodels.org Getting Started tutorial](https://www.tidymodels.org/start/)

**Practice:**
- Fit a linear model and GLM using `lm()` and `glm()` in base R
- Reproduce your Phase 8 ISLP analysis using `tidymodels`: 5-fold CV, Lasso tuning, final fit

**Checkpoint:**
> `week91_tidymodels.Rmd`: complete tidymodels workflow with `recipe`, `workflow`, `tune_grid`, and `last_fit`, compared to your Python sklearn pipeline.

---

### Week 92: R Capstone — Reproducible Research with Quarto

**Study:**
- R4DS Chapters 28–30 (Quarto, Workflow: Getting Help) — create a Quarto document

**Practice:**
- Convert your most important Python analysis (Phase 8 or Phase 10 Capstone) into a fully reproducible Quarto report
- The report must render to both HTML and PDF without modification

**Checkpoint:**
> `phase13_capstone.qmd`: a complete Quarto document with code, figures, tables, LaTeX equations, and cross-references. Must compile to both HTML and PDF via a single `quarto render` command.

---

---

</details>

---

<details>
<summary><h2>Phase 14: Econometrics, Time Series & Financial Econometrics</h2></summary>

**Duration:** 16 Weeks
**Resources (Interwoven):**
- [Basic Econometrics, 5th Ed. — Gujarati & Porter](https://www.mheducation.com/highered/product/basic-econometrics-gujarati-porter/M912000001.html)
- [Financial Econometrics Notes — Kevin Sheppard](https://www.kevinsheppard.com/teaching/mfe/)

**Depth Assessment:** Gujarati ~900 pages, moderate-hard (economic applications context); Sheppard MFE notes ~400 pages + Python code, hard. Strategy: Gujarati for foundational theory, Sheppard for financial applications and Python implementation.
**Why this resource matters:** Causal/statistical modeling for economics and financial time series.
**Weekly Structure (Strict Completion):** Week 93–101 + 6 diagnostics/robustness/replication weeks.
**Phase Deliverables:** Econometrics notebooks, time-series models, financial econometrics capstone.
**Phase Mastery Criteria:** Required chapters/notes completed; diagnostics and assumptions fully verified.

---

### Week 93: OLS — Theory & Gauss-Markov Assumptions

**Study:**
- Gujarati Chapters 1–5 (Nature of Regression, Two-Variable Regression, Interval Estimation, Hypothesis Testing)

**Practice:**
- Implement OLS in matrix form with full diagnostic output (coefficients, SEs, t-stats, p-values, F-stat, R²)
- Apply to a real economic dataset from FRED; interpret every coefficient

**Checkpoint:**
> `week93_ols_econometrics.ipynb`: from-scratch OLS with formatted output table matching a statsmodels summary, interpretation of all statistics for a real economic regression.

---

### Week 94: Multiple Regression & Specification Issues

**Study:**
- Gujarati Chapters 6–10 (Multiple Regression, Dummy Variables, Multicollinearity)

**Practice:**
- Demonstrate the Frisch-Waugh-Lovell theorem numerically
- Run a regression with dummies for quarters; interpret the seasonal effects

**Checkpoint:**
> `week94_multiple_regression.ipynb`: FWL theorem numerical verification, seasonal dummy regression with plotted seasonal components, VIF computation and remediation.

---

### Week 95: Heteroskedasticity & Autocorrelation

**Study:**
- Gujarati Chapters 11–12 (Heteroskedasticity, Autocorrelation)
- Sheppard MFE: HAC standard errors section

**Practice:**
- Implement the Breusch-Pagan and White tests for heteroskedasticity from scratch
- Implement Newey-West HAC standard errors; compare to OLS SEs on financial time series data

**Checkpoint:**
> `week95_diagnostics.ipynb`: from-scratch BP and White tests, Newey-West implementation with comparison table, and a simulated Monte Carlo showing OLS inference failure under heteroskedasticity.

---

### Week 96: Time Series I — Stationarity, ARIMA

**Study:**
- Gujarati Chapters 21–22 (Time Series Econometrics)
- Sheppard MFE: Time Series chapter

**Practice:**
- Implement the Augmented Dickey-Fuller (ADF) test from scratch
- Fit ARIMA(p,d,q) models to an economic time series; select orders using AIC/BIC; generate forecasts

**Checkpoint:**
> `week96_arima.ipynb`: from-scratch ADF test vs. `statsmodels.tsa.stattools.adfuller`, ARIMA model selection ACF/PACF plots, 12-step-ahead forecast with confidence intervals.

---

### Week 97: Time Series II — VAR, Cointegration & ECM

**Study:**
- Gujarati Chapter 22 (continued: VAR, VECM, Cointegration)
- Sheppard MFE: Multivariate Time Series

**Practice:**
- Test for cointegration between two price series using the Engle-Granger two-step method
- Estimate a Vector Error Correction Model (VECM); interpret the adjustment coefficients

**Checkpoint:**
> `week97_var_vecm.ipynb`: cointegration test with visual of residuals, VECM estimation and interpretation, impulse response functions plotted for a 10-period horizon.

---

### Week 98: Volatility Modeling — ARCH, GARCH, Variations

**Study:**
- Sheppard MFE: ARCH, GARCH, GJR-GARCH, EGARCH
- Read the original Engle (1982) ARCH paper and Bollerslev (1986) GARCH paper — abstract + model sections

**Practice:**
- Implement GARCH(1,1) estimation via maximum likelihood from scratch (optimize with `scipy.optimize`)
- Apply to S&P 500 daily returns; compare GARCH, GJR-GARCH, and EGARCH using AIC

**Checkpoint:**
> `week98_garch.ipynb`: from-scratch GARCH(1,1) MLE, volatility forecast plot with `arch` library comparison, model comparison table, and written interpretation of asymmetric volatility (leverage effect).

---

### Week 99: Panel Data & Instrumental Variables

**Study:**
- Gujarati Chapters 14–16 (Panel Data Regression, Dummy Variables advanced, IV Estimation)

**Practice:**
- Estimate Fixed Effects and Random Effects models on a panel dataset; run the Hausman test
- Implement 2SLS (Two-Stage Least Squares) from scratch for an IV estimation problem

**Checkpoint:**
> `week99_panel_iv.ipynb`: FE vs. RE comparison with Hausman test, 2SLS from scratch vs. `linearmodels.iv.IV2SLS`, and written discussion of endogeneity and instrument validity.

---

### Week 100: Financial Econometrics — Asset Pricing & Factor Models

**Study:**
- Sheppard MFE: Asset Pricing, Factor Models, Portfolio Evaluation

**Practice:**
- Estimate the Fama-French 3-Factor model for 10 portfolios using OLS; test the zero-alpha hypothesis
- Implement Mean-Variance portfolio optimization (Markowitz) from scratch using `cvxpy`

**Checkpoint:**
> `week100_asset_pricing.ipynb`: FF3 factor model with GRS test, efficient frontier plot from scratch vs. `cvxpy`, and written interpretation of alpha, beta, and factor loadings.

---

### Week 101: Advanced Topics & Econometrics Capstone

**Study:**
- Gujarati Chapters 17–20 (Qualitative Response Models, Limited Dependent Variables, Probit/Logit/Tobit)
- Sheppard MFE: remaining advanced topics

**Practice:**
- Estimate a Probit and Logit model; compute marginal effects at the mean and average marginal effects
- Full econometric study: pose an economic question, collect data, estimate a model, check robustness, write up results

**Checkpoint:**
> `phase14_capstone.ipynb` + a 1500-word econometric report in Quarto/LaTeX: research question, data description, estimation results table (formatted like an academic paper), robustness checks, and conclusion.

---

---

</details>

---

<details>
<summary><h2>Phase 15: Causal Inference</h2></summary>

**Duration:** 10 Weeks
**Resource:** [Causal Inference: The Mixtape — Scott Cunningham](https://mixtape.scunning.com/)
**Depth Assessment:** ~600 pages, moderate-hard. Essential for anyone working with observational data. Rich in intuition and applied examples. Python and R code available.
**Why this resource matters:** Correct decision-making under observational data and policy settings.
**Weekly Structure (Strict Completion):** Week 102–107 + 4 identification-strategy implementation weeks.
**Phase Deliverables:** DAG analyses, DiD/RDD/IV/SC implementations, causal capstone.
**Phase Mastery Criteria:** Identification assumptions defended; all method exercises/implementations complete.

---

### Week 102: DAGs, Potential Outcomes & Selection Bias

**Study:**
- Mixtape Chapters 1–3 (Introduction, Probability and Regression Review, Directed Acyclical Graphs)

**Practice:**
- Draw DAGs for 3 different economic questions using `graphviz` or `dagitty`
- Identify confounders, mediators, and colliders in each DAG; determine the correct adjustment set

**Checkpoint:**
> `week102_dags.ipynb`: 3 DAGs with annotated confounders/mediators/colliders, written d-separation analysis for each, and simulation demonstrating collider bias.

---

### Week 103: Randomization & Matching Methods

**Study:**
- Mixtape Chapters 4–5 (Potential Outcomes Causal Model, Matching and Subclassification)

**Practice:**
- Implement Propensity Score Matching from scratch using logistic regression for the propensity score
- Apply to the LaLonde (1986) job training dataset; estimate the Average Treatment Effect (ATE)

**Checkpoint:**
> `week103_matching.ipynb`: PSM from scratch with love plot (covariate balance before/after matching), ATE estimate with confidence interval, and written comparison of matched vs. unmatched estimates.

---

### Week 104: Regression Discontinuity Design

**Study:**
- Mixtape Chapter 6 (Regression Discontinuity)
- Read the Lee & Lemieux (2010) RD survey — key sections

**Practice:**
- Implement a sharp RD estimator from scratch (local linear regression on each side of cutoff)
- Perform bandwidth selection (Imbens-Kalyanaraman), placebo tests at false cutoffs, and McCrary density test

**Checkpoint:**
> `week104_rdd.ipynb`: sharp RD implementation, optimal bandwidth selection, 3-panel diagnostic figure (RD plot, density test, placebo at false cutoffs), and written causal interpretation.

---

### Week 105: Difference-in-Differences & Event Studies

**Study:**
- Mixtape Chapter 9 (Difference-in-Differences)
- Read Callaway & Sant'Anna (2021) for staggered DiD

**Practice:**
- Implement canonical 2×2 DiD estimator; verify the parallel trends assumption visually
- Apply to a real policy change dataset; produce an event study plot

**Checkpoint:**
> `week105_did.ipynb`: 2×2 DiD with parallel trends test, event study plot (leads and lags), and written discussion of the parallel trends assumption and its plausibility.

---

### Week 106: Instrumental Variables & Local Average Treatment Effect

**Study:**
- Mixtape Chapter 7 (Instrumental Variables)
- Focus: LATE interpretation, weak instruments, Angrist-Pischke F-statistic

**Practice:**
- Apply IV estimation to a classic example (e.g., compulsory schooling laws and earnings — Angrist & Krueger 1991 setup)
- Test for instrument relevance (first-stage F-stat) and construct a Wald estimator

**Checkpoint:**
> `week106_iv_late.ipynb`: IV estimation with first-stage diagnostics, LATE interpretation, and written discussion of the monotonicity assumption.

---

### Week 107: Synthetic Control & Phase 15 Capstone

**Study:**
- Mixtape Chapter 10 (Synthetic Control)

**Practice:**
- Implement Synthetic Control from scratch (solve the constrained optimization problem using `scipy.optimize`)
- Apply to a real policy evaluation (e.g., California tobacco control program)
- Write a 1500-word causal analysis report using the most appropriate method for your chosen question

**Checkpoint:**
> `phase15_capstone/`: Synthetic Control implementation, donor pool visualization, pre/post comparison, and a full causal inference report. Must clearly state the identification assumption, why it holds (or is plausible), and what the estimate means causally.

---

---

</details>

---

<details>
<summary><h2>Phase 16: MLOps & Data Engineering</h2></summary>

**Duration:** 16 Weeks
**Resources (Interwoven):**
- [MLOps Zoomcamp — DataTalksClub](https://github.com/DataTalksClub/mlops-zoomcamp)
- [Machine Learning Systems — mlsysbook.ai](https://mlsysbook.ai/)
- [Data Engineering Zoomcamp — DataTalksClub](https://github.com/DataTalksClub/data-engineering-zoomcamp)

**Depth Assessment:** MLOps Zoomcamp = 9 structured modules; mlsysbook = ~400 pages; DE Zoomcamp = 9 structured modules. All are very practical and hands-on. This phase demands significant infrastructure work.
**Why this resource matters:** Production-grade systems, deployment, monitoring, and infrastructure reliability.
**Weekly Structure (Strict Completion):** Week 108–117 + 6 infra hardening and observability validation weeks.
**Phase Deliverables:** Containerized pipelines, orchestration, deployment, monitoring stack, production capstone.
**Phase Mastery Criteria:** End-to-end system reproducible/monitored; all modules/labs complete with passing checks.

---

### Week 108: Docker, Containers & ML Project Structure

**Study:**
- MLOps Zoomcamp Module 1 (Introduction & MLOps Maturity)
- mlsysbook Chapters 1–3 (Introduction, ML Lifecycle, Data Engineering)
- Install: Docker, Docker Compose

**Practice:**
- Containerize your Phase 10 Classical ML project in Docker
- Write a `Dockerfile`, `docker-compose.yml`, and a `Makefile` with targets for `train`, `predict`, `test`

**Checkpoint:**
> A fully containerized ML project: `docker-compose up` must reproduce training, evaluation, and inference. Include a `README.md` with exact reproduction instructions.

---

### Week 109: Experiment Tracking with MLflow

**Study:**
- MLOps Zoomcamp Module 2 (Experiment Tracking — MLflow)
- mlsysbook Chapter 4 (Model Training)

**Practice:**
- Integrate MLflow into your Phase 12 Deep Learning project: log all hyperparameters, metrics, and artifacts
- Set up an MLflow tracking server; compare 20 runs and select the best model using the MLflow UI

**Checkpoint:**
> An MLflow-instrumented training script with: experiment tracking for 20 hyperparameter configurations, a registered model in the MLflow Model Registry, and a screenshot of the MLflow UI showing the run comparison.

---

### Week 110: ML Pipelines & Workflow Orchestration

**Study:**
- MLOps Zoomcamp Module 3 (Orchestration — Prefect or Mage)
- mlsysbook Chapter 5 (Model Deployment)

**Practice:**
- Build an end-to-end ML pipeline using Prefect (or Mage): data ingestion → feature engineering → training → evaluation → model registration
- The pipeline must be schedulable and handle failures gracefully with retries

**Checkpoint:**
> A working Prefect/Mage pipeline with at least 5 tasks, retry logic, and logging. Must complete a full run end-to-end without manual intervention.

---

### Week 111: Model Deployment — REST APIs & Batch Scoring

**Study:**
- MLOps Zoomcamp Module 4 (Deployment)
- mlsysbook Chapter 6 (Model Serving)

**Practice:**
- Deploy your best model as a REST API using FastAPI
- Build a batch scoring script that processes a CSV of 10,000 records; containerize both services

**Checkpoint:**
> A `FastAPI` model service with: a `/predict` endpoint, request validation (Pydantic), health check endpoint, and a batch scoring script. Both containerized and tested with `pytest`.

---

### Week 112: Model Monitoring & Drift Detection

**Study:**
- MLOps Zoomcamp Module 5 (Model Monitoring)
- mlsysbook Chapter 7 (Model Monitoring)

**Practice:**
- Set up Evidently to monitor prediction drift and data drift for your deployed model
- Simulate distribution shift; trigger an automated retraining job when drift exceeds a threshold

**Checkpoint:**
> A monitoring pipeline: Evidently reports generated for 5 simulated time periods showing drift progression, automated alert (email or log) when threshold is breached, and a dashboard (Grafana or Streamlit).

---

### Week 113: Data Engineering I — Batch Pipelines & Warehousing

**Study:**
- DE Zoomcamp Modules 1–3 (Introduction, Workflow Orchestration, Data Warehouse)
- Install: Google Cloud SDK or use local alternatives; DuckDB for local warehousing

**Practice:**
- Build a batch ELT pipeline: ingest NYC TLC taxi data → load to DuckDB → transform with dbt → serve analytics queries
- Write 5 dbt models with tests and documentation

**Checkpoint:**
> A dbt project with: source definitions, 5 models (staging → intermediate → mart), `dbt test` passing, and `dbt docs generate` producing browsable documentation.

---

### Week 114: Data Engineering II — Streaming with Kafka

**Study:**
- DE Zoomcamp Modules 4–5 (Analytics Engineering, Batch Processing with Spark)

**Practice:**
- Set up a local Kafka cluster using Docker Compose
- Build a producer that streams taxi ride events; a consumer that aggregates running totals by zone in 1-minute windows

**Checkpoint:**
> A Kafka streaming pipeline: producer script, consumer with windowed aggregation, visualized real-time metrics in a simple Streamlit dashboard.

---

### Week 115: Apache Spark & Distributed Computing

**Study:**
- DE Zoomcamp Module 5 (Spark) — all sections
- mlsysbook Chapter 8 (Efficient AI Training and Inference)

**Practice:**
- Process the NYC TLC dataset (full year, ~100M rows) using PySpark on your local machine or a cloud cluster
- Implement a Spark ML pipeline: feature engineering → training → evaluation

**Checkpoint:**
> A PySpark job that: reads Parquet, applies transformations, trains a model via `pyspark.ml`, and writes results back to Parquet. Must run to completion and log runtime metrics.

---

### Week 116: CI/CD for ML & Infrastructure as Code

**Study:**
- mlsysbook Chapters 9–10 (MLOps Tooling, Production Infrastructure)
- DE Zoomcamp Module 6 (Streaming continued)

**Practice:**
- Set up a GitHub Actions CI/CD pipeline that: runs `pytest`, builds the Docker image, runs MLflow training, and deploys to a staging endpoint on every PR merge
- Write Terraform (or Pulumi) code to provision your infrastructure

**Checkpoint:**
> A GitHub repository with: GitHub Actions workflow file, Dockerfile, Terraform config, and a passing CI/CD run log. The pipeline must complete end-to-end in under 15 minutes.

---

### Week 117: Phase 16 Capstone — Production ML System

**Study:**
- mlsysbook final chapters (review)
- Review all Zoomcamp best practices

**Practice:**
- Build a production-ready ML system end-to-end:
  - Data ingestion from an external API (scheduled via Prefect)
  - Feature store (use Feast or a simple DuckDB-based store)
  - Automated training pipeline with MLflow tracking
  - FastAPI serving with monitoring
  - CI/CD via GitHub Actions

**Checkpoint:**
> A public GitHub repository with complete documentation, a system architecture diagram, and a 2-minute recorded demo video. The system must be deployable from scratch with a single `make deploy` command.

---

---

</details>

---

<details>
<summary><h2>Phase 17: Flow Matching & Diffusion Models</h2></summary>

**Duration:** 8 Weeks
**Resource:** [MIT Diffusion Course 2026](https://diffusion.csail.mit.edu/2026/index.html)
**Supplementary:** "Score-Based Generative Modeling through Stochastic Differential Equations" (Song et al., 2021); "Flow Matching for Generative Modeling" (Lipman et al., 2022)
**Depth Assessment:** A cutting-edge graduate-level course (MIT CSAIL, 2026). Lectures + problem sets. Requires Phase 12 (DL) as a firm prerequisite. This is the frontier of generative AI theory.
**Why this resource matters:** Frontier generative modeling theory and implementation.
**Weekly Structure (Strict Completion):** Week 118–122 + 3 problem-set/paper re-derivation weeks.
**Phase Deliverables:** DDPM/score-based/flow-matching implementations + research-style capstone report.
**Phase Mastery Criteria:** Lectures/problem sets/papers completed; implementations reproduce expected behavior.

---

### Week 118: Diffusion Fundamentals — Forward & Reverse Processes

**Study:**
- MIT Diffusion: Lectures 1–3 (Introduction, Denoising Score Matching, DDPM)
- Read Ho et al. (2020) DDPM paper in full

**Practice:**
- Implement DDPM (Denoising Diffusion Probabilistic Models) from scratch in PyTorch
- Train on MNIST; visualize the forward noising process and the reverse denoising chain at each step

**Checkpoint:**
> `week118_ddpm.ipynb`: from-scratch DDPM, visualization of forward process (T=0 to T=1000), and a grid of reverse-process samples showing progressive denoising.

---

### Week 119: Score Matching & SDE Formulation

**Study:**
- MIT Diffusion: Lectures 4–6 (Score Matching, SDEs, Continuous-Time Diffusion)
- Read Song et al. (2021) SDE paper — Sections 1–4

**Practice:**
- Implement denoising score matching loss from scratch
- Train a score network on 2D toy data (e.g., a mixture of Gaussians); visualize the learned score field as a vector field

**Checkpoint:**
> `week119_score_matching.ipynb`: score matching implementation, 2D score field visualization (quiver plot), and a comparison of the learned vs. true score fields.

---

### Week 120: Flow Matching — Theory & Implementation

**Study:**
- MIT Diffusion: Lectures 7–9 (Continuous Normalizing Flows, Flow Matching, Optimal Transport)
- Read Lipman et al. (2022) Flow Matching paper in full

**Practice:**
- Implement Conditional Flow Matching (CFM) from scratch using the `torchdiffeq` ODE solver
- Train on 2D toy data; visualize the learned flow trajectories

**Checkpoint:**
> `week120_flow_matching.ipynb`: CFM implementation, flow trajectory visualization (animated), and a comparison of sample quality (2D density estimates) between DDPM and CFM on the same toy dataset.

---

### Week 121: Latent Diffusion & Guidance Techniques

**Study:**
- MIT Diffusion: Lectures 10–12 (Latent Diffusion, Classifier-Free Guidance, DALL-E / Stable Diffusion architecture)
- Read Rombach et al. (2022) LDM paper — key sections

**Practice:**
- Implement classifier-free guidance from scratch for a conditional DDPM on MNIST (condition on digit class)
- Vary guidance scale (w = 0, 1, 3, 7, 10); observe the quality/diversity trade-off

**Checkpoint:**
> `week121_cfg.ipynb`: conditional DDPM with CFG, a grid showing generated samples at 5 guidance scales for each class, and a written analysis of the diversity-fidelity trade-off.

---

### Week 122: Advanced Topics & Phase 17 Capstone

**Study:**
- MIT Diffusion: Lectures 13+ (Consistency Models, Rectified Flow, Applications to proteins, audio, video)
- Read any one of: Consistency Models (Song et al., 2023) or Rectified Flow (Liu et al., 2022)

**Practice:**
- Implement Rectified Flow from scratch as a simplified flow matching variant
- Phase 17 Capstone: train a diffusion or flow-matching model on a domain of your choice (audio mel-spectrograms, molecular structures, time series, or image patches)
- Compare DDPM vs. Flow Matching on your domain: training efficiency, sample quality, inference speed

**Checkpoint:**
> `phase17_capstone/`: complete generative modeling project with a written technical report (~2000 words) including: mathematical derivation of your chosen method, architecture decisions, training details, quantitative evaluation (FID or domain-specific metric), and a limitations section. This is your final deliverable — treat it as a paper draft.

---

---

</details>

---

---

## 🏆 Final Synthesis & Portfolio

Upon completing all 17 phases, you will have accumulated:

- **10+ end-to-end projects** across the full data science and AI stack
- **From-scratch implementations** of every major algorithm: OLS, MLE, MCMC, SVM, neural networks, transformers, diffusion models
- **Rigorous mathematical foundations**: probability theory, mathematical statistics, linear algebra, optimization
- **Production engineering skills**: Docker, CI/CD, Kafka, Spark, model monitoring
- **Causal reasoning toolkit**: DAGs, DiD, RDD, IV, Synthetic Control
- **Generative AI expertise**: VAEs, GANs, diffusion models, flow matching

### Recommended Portfolio Structure

```
portfolio/
├── README.md              # Overview with links to all projects
├── phase01_data_analysis/ # pandas EDA capstone
├── phase08_islp/          # ML pipeline capstone
├── phase10_classical_ml/  # Python package (from-scratch ML)
├── phase12_deep_learning/ # DL project (CNN/Transformer/Generative)
├── phase14_econometrics/  # Econometric paper
├── phase15_causal/        # Causal inference study
├── phase16_mlops/         # Production ML system
└── phase17_diffusion/     # Generative model paper draft
```

### Continuing Education

After completing this roadmap, consider:

- Read recent NeurIPS/ICML/ICLR proceedings in your specialization
- Contribute to an open-source project (D2L.ai, scikit-learn, HuggingFace)
- Write and publish one of your capstone projects as a blog post or preprint
- Consider graduate coursework or research if you want to push further into theory

---

> *"The impediment to action advances action. What stands in the way becomes the way."*
> — Marcus Aurelius
>
> Start with Week 1. The rest will follow.

---

*Roadmap Version 2.0 | Designed for 12–15 hrs/week | Total: ~192 weeks (~3.7 years)*
