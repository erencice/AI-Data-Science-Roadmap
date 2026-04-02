# AI & Data Science Weekly Plan — Activities, Practice, and Pass Criteria

![Duration](https://img.shields.io/badge/duration-~162_weeks-6f42c1)
![Pace](https://img.shields.io/badge/pace-8–10_h%2Fweek-0e8a16)
![Path](https://img.shields.io/badge/path-beginner%E2%86%92practitioner-0366d6)
![Style](https://img.shields.io/badge/style-cumulative%2C_concept%E2%86%92practice-555)

Zero prior knowledge is assumed. Learning order is strictly top-to-bottom. Each week includes a clear “Pass” requirement aligned to the primary resource.

— Quick jump —
- Phase 1 · Data Analysis Foundations
- Phase 2 · SQL
- Phase 3 · Mathematics for Machine Learning and Data Science
- Phase 4 · Introduction to Probability
- Phase 5 · Statistics Fundamentals
- Phase 6 · Mathematical Statistics
- Phase 7 · Applied Multivariate Statistics
- Phase 8 · Bayesian Statistics & Missing Data
- Phase 9 · Statistical Learning with Python (ISLP)
- Phase 10 · Data Mining
- Phase 11 · Classical Machine Learning
- Phase 12 · Elements of Statistical Learning
- Phase 13 · Deep Learning
- Phase 14 · R for Data Science
- Phase 15 · Econometrics, Time Series & Financial Econometrics
- Phase 16 · Causal Inference
- Phase 17 · MLOps & Data Engineering
- Phase 18 · Flow Matching and Diffusion Models

Legend
- 📖 Activities (primary source)
- 🧪 Practice (small tasks)
- ✅ Pass (weekly pass criterion)
- 🛠️ How (implementation hint)

Duration and pacing
- Duration: ~162 weeks (≈4.2 years), 8–10 h/week
- Weekly output: small practical tasks only

---------------------------------------------------------------------

<details>
<summary><b>Phase 1 · Data Analysis Foundations — Weeks 1–9</b></summary>

Week 1 — P4DA Ch. 1–2
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Review basic Python usage (variables, simple control flow, functions) and revisit the relevant chapters.
- ✅ Pass: Create a simple notebook that shows basic Python syntax (a few variables, a simple function, and basic output).
- 🛠️ How: Open Jupyter and review examples from the relevant sections in Ch.1 and Ch.2; no advanced setup or libraries required.

Week 2 — P4DA Ch. 3–4
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Work with tuples, lists, dicts, sets (Ch.3); create and manipulate NumPy ndarrays; practice array indexing, slicing, and vectorized operations (Ch.4).
- ✅ Pass: Build a notebook that: (1) demonstrates list/dict/set operations; (2) creates 2D NumPy arrays, performs element-wise and matrix operations; (3) uses boolean indexing to filter data; (4) times vectorized vs loop-based computation.
- 🛠️ How: `np.array`, `np.arange`, `np.reshape`, boolean masks, `np.where`, `%timeit` to compare performance.

Week 3 — P4DA Ch. 5–6
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Create Series and DataFrames; use `.loc/.iloc` indexing; load data from CSV/JSON/Excel files (Ch.5–6).
- ✅ Pass: Load a dataset from CSV, inspect with `.head()/.info()/.describe()`, select columns via `.loc/.iloc`, filter rows with boolean masks, and export cleaned data to a new CSV.
- 🛠️ How: `pd.read_csv`, `pd.read_json`, `df.loc[rows, cols]`, `df.iloc[row_idx, col_idx]`, `df.to_csv`.

Week 4 — P4DA Ch. 7–8
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Handle missing data; clean strings with `.str` methods; merge/join DataFrames; reshape with `stack/unstack/pivot/melt` (Ch.7–8).
- ✅ Pass: Take a messy dataset and: (1) handle missing values (drop or fill); (2) standardize string columns (trim/lower); (3) merge with a second table; (4) pivot or melt the result; document row counts at each step.
- 🛠️ How: `df.dropna`, `df.fillna`, `df["col"].str.strip().str.lower()`, `pd.merge`, `pd.pivot_table`, `pd.melt`.

Week 5 — P4DA Ch. 9–10
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Create plots with matplotlib/seaborn (Ch.9); perform aggregation with `groupby` (Ch.10).
- ✅ Pass: Produce 4 visualizations (histogram, scatter, line, bar) with proper labels/titles; use `groupby().agg()` to compute multi-column summaries; combine groupby results with plots.
- 🛠️ How: `plt.plot`, `plt.hist`, `sns.scatterplot`, `df.groupby("col").agg({"num":"mean"})`, `plt.savefig`.

Week 6 — P4DA Ch. 11–12 (+appendices)
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Work with time series: DateTimeIndex, resampling, rolling windows (Ch.11); explore advanced pandas: Categoricals, method chaining, performance (Ch.12).
- ✅ Pass: Load time series data, set DateTimeIndex, resample to weekly/monthly, compute rolling statistics; convert a column to Categorical; refactor pipeline using method chaining; time vectorized vs apply.
- 🛠️ How: `pd.to_datetime`, `df.set_index`, `df.resample("W").mean()`, `.rolling(7).mean()`, `pd.Categorical`, `.pipe()`.

Week 7 — P4DA Project A
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: End-to-end EDA pipeline using all chapters 1–12: load, clean, transform, aggregate, visualize.
- ✅ Pass: Apply a complete EDA workflow to a new dataset; produce ≥5 visualizations; write a 1-page summary with ≥3 insights, ≥2 hypotheses, and ≥1 data quality issue identified.
- 🛠️ How: Combine prior weeks' functions into a reusable pipeline; keep code modular and well-documented.

Week 8 — P4DA Project B
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Feature engineering using transforms from the book: date/time features, categorical encoding, ratios, binning.
- ✅ Pass: Create ≥5 derived features (date parts, ratios, binned numerics, category combinations); document each feature's rationale, potential predictive value, and leakage risk.
- 🛠️ How: `df["date"].dt.month`, `df.assign(ratio=lambda x: x["a"]/x["b"])`, `pd.cut`, `pd.get_dummies`.
Week 9 — Phase 1 — End-to-End Mini Project
- 📖 Activities: Phase 1 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 1.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 2 · SQL — Weeks 10–13</b></summary>
  
Week 10 — SQL Basics, Syntax & Data Manipulation
- 📖 Activities: [SQL Roadmap (GeeksforGeeks)](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- 🧪 **Practice**: Understand the foundation of databases (**SQL Basics**: RDBMS, SQL vs NoSQL). Master **Basic Syntax** including SQL data types and operators. Learn **Data Definition Language (DDL)** commands and syntax to define database structures. Execute **Data Manipulation Language (DML)** commands and SQL Clauses for day-to-day operations (inserting, updating, deleting, and retrieving data), and understand the difference between DML vs DDL and DML vs TCL.
- ✅ **Pass**: Install and configure an RDBMS; create a sample database demonstrating DDL commands; write queries covering basic syntax and operators; perform DML operations (INSERT, UPDATE, DELETE) and basic retrieval (SELECT) using appropriate clauses; document the differences between DDL and DML operations.
- 🛠️ **How**: Start with database theory (Relational vs NoSQL); use `CREATE TABLE`, `ALTER TABLE`, and `DROP` for DDL; use `INSERT INTO`, `UPDATE`, and `DELETE` for DML; practice basic `SELECT` queries with operators (WHERE, AND/OR, LIKE) to filter data.

Week 11 — Aggregate Queries, Constraints, JOINs & Subqueries
- 📖 Activities: [SQL Roadmap (GeeksforGeeks)](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- 🧪 **Practice**: Master **Aggregate Queries** and aggregate functions to perform calculations on grouped data. Apply **Data Constraints** to enforce data rules and maintain integrity. Work with **JOIN Queries** (Inner Join, Outer Join, Cartesian Join, and Self Join) to combine data from multiple tables. Utilize **Subqueries** (Correlated, Nested) for complex filtering, and analyze the use cases of SQL Join vs Subquery.
- ✅ **Pass**: Create a relational schema with constraints enforced (Primary Key, Foreign Key, Not Null); write queries using aggregate functions with GROUP BY and HAVING; write queries demonstrating all major JOIN types (Inner, Outer, Self, Cartesian); create complex queries using nested and correlated subqueries; document how constraints maintain data accuracy.
- 🛠️ **How**: Use `COUNT`, `SUM`, `AVG`, `MIN`, `MAX` for aggregations; enforce rules with constraints; combine tables using `INNER JOIN`, `LEFT/RIGHT OUTER JOIN`, and `CROSS JOIN`; write subqueries in `WHERE`, `FROM`, and `SELECT` clauses; compare the execution and readability of JOINs vs Subqueries.

Week 12 — Advanced Functions, Performance & Advanced SQL Features
- 📖 Activities: [SQL Roadmap (GeeksforGeeks)](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- 🧪 **Practice**: Explore **Advanced Functions** (String, Date and Time, Numeric). Create **Views** to simplify complex queries. Utilize **Indexes** for query optimization and best practices. Understand **Transactions** for data reliability and **Integrity Constraints**. Work with **Stored Procedures** and functions. Master **Performance Optimization** techniques and **Advanced SQL** features like Window Functions, Common Table Expressions (CTEs), Pivot and Unpivot, Dynamic SQL, and SQL Triggers.
- ✅ **Pass**: Build an optimized database schema; write queries using advanced string, numeric, and date functions; create and manage Views; demonstrate performance optimization by creating Indexes and analyzing queries; write a Transaction block ensuring consistency; create a Stored Procedure; implement Window Functions, CTEs, and Triggers for advanced data manipulation.
- 🛠️ **How**: Use specific functions like `CONCAT`, `EXTRACT`, `ROUND`; `CREATE VIEW view_name`; `CREATE INDEX` and apply query optimization best practices; manage transactions with `BEGIN`, `COMMIT`, `ROLLBACK`; write `CREATE PROCEDURE`; apply `OVER (PARTITION BY... ORDER BY...)` for Window Functions; structure queries with `WITH` (CTEs); implement `CREATE TRIGGER` and explore Pivot/Unpivot operations.

Week 13 — Phase 2 — End-to-End Mini Project
- 📖 Activities: Phase 2 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 2.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 3 · Mathematics for Machine Learning and Data Science — Weeks 14–23</b></summary>

Week 14 — Linear Algebra (Course 1, Part 1)
- 📖 Activities: [Course 1: Linear Algebra](https://learn.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science/information?tab=0)
- 🧪 Practice: Systems of linear equations, vectors, and matrices.
- ✅ Pass: Complete the week's quizzes and Python (NumPy) labs without using solver libraries.
- 🛠️ How: Implement basic matrix operations from scratch in Python.

Week 15 — Linear Algebra (Course 1, Part 2)
- 📖 Activities: [Course 1: Linear Algebra](https://learn.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science/information?tab=0)
- 🧪 Practice: Linear independence, basis, rank, and linear transformations.
- ✅ Pass: Pass the week's assignments on transforming datasets and identifying dependent features.
- 🛠️ How: Use NumPy to compute matrix rank and apply geometric transformations.

Week 16 — Linear Algebra (Course 1, Part 3)
- 📖 Activities: [Course 1: Linear Algebra](https://learn.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science/information?tab=0)
- 🧪 Practice: Eigenvalues, eigenvectors, and PCA intuitions.
- ✅ Pass: Calculate eigenvalues of a matrix manually and verify with NumPy.
- 🛠️ How: Apply `np.linalg.eig` inside the weekly lab.

Week 17 — Calculus (Course 2, Part 1)
- 📖 Activities: [Course 2: Calculus](https://learn.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science/information?tab=1)
- 🧪 Practice: Derivatives, gradients, and optimization foundations.
- ✅ Pass: Compute gradients of standard loss functions and pass lab assignments.
- 🛠️ How: Python labs using numerical differentiation.

Week 18 — Calculus (Course 2, Part 2)
- 📖 Activities: [Course 2: Calculus](https://learn.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science/information?tab=1)
- 🧪 Practice: Chain rule, backpropagation, and partial derivatives.
- ✅ Pass: Implement a basic step of gradient descent.
- 🛠️ How: Update weights using `w = w - alpha * dw`.

Week 19 — Calculus (Course 2, Part 3)
- 📖 Activities: [Course 2: Calculus](https://learn.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science/information?tab=1)
- 🧪 Practice: Newton's method and advanced optimization.
- ✅ Pass: Compare gradient descent vs Newton's method in the lab.
- 🛠️ How: Write a script recording the number of iterations to converge.

Week 20 — Probability & Statistics (Course 3, Part 1)
- 📖 Activities: [Course 3: Probability & Statistics](https://learn.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science/information?tab=2)
- 🧪 Practice: Descriptive stat distributions and probability axioms.
- ✅ Pass: Visualize different distributions in Python.
- 🛠️ How: Use `scipy.stats` for distributions and basic plotting.

Week 21 — Probability & Statistics (Course 3, Part 2)
- 📖 Activities: [Course 3: Probability & Statistics](https://learn.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science/information?tab=2)
- 🧪 Practice: Maximum likelihood estimation and confidence intervals.
- ✅ Pass: Perform an MLE on a given dataset in the assignments.
- 🛠️ How: Compute log-likelihood manually.

Week 22 — Probability & Statistics (Course 3, Part 3)
- 📖 Activities: [Course 3: Probability & Statistics](https://learn.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science/information?tab=2)
- 🧪 Practice: Hypothesis testing, p-values, and A/B testing basics.
- ✅ Pass: Execute a two-sample t-test lab.
- 🛠️ How: Use statistical hypothesis testing formulas.
Week 23 — Phase 3 — End-to-End Mini Project
- 📖 Activities: Phase 3 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 3.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 4 · Introduction to Probability — Weeks 24–28</b></summary>

Week 24 — Probability Ch. 1–2 (Discrete Probability, Continuous Probability)
- 📖 [Probability (PDF)](https://math.dartmouth.edu/~prob/prob/prob.pdf)
- 🧪 Practice: Understand sample spaces, events, and probability axioms; work with discrete and continuous random variables; compute probabilities using counting methods; master conditional probability and independence; apply Bayes' theorem.
- ✅ Pass: Solve ≥10 probability problems covering: sample space construction, probability calculations using combinations/permutations, conditional probability, independence tests, and Bayes' theorem applications; verify solutions analytically and via simulation; demonstrate Law of Total Probability.
- 🛠️ How: Use combinatorics: `math.comb(n,k)`, `math.perm(n,k)`; simulate outcomes with `np.random.choice`; verify `P(A|B) = P(A∩B)/P(B)`; Bayes: `P(A|B) = P(B|A)P(A)/P(B)`; compare analytical vs empirical probabilities.

Week 25 — Probability Ch. 3–4 (Expectation, Distributions)
- 📖 [Probability (PDF)](https://math.dartmouth.edu/~prob/prob/prob.pdf)
- 🧪 Practice: Compute expected values and variance; work with common discrete distributions (Bernoulli, Binomial, Geometric, Poisson); understand continuous distributions (Uniform, Exponential, Normal); apply moment generating functions; explore distribution relationships.
- ✅ Pass: Compute expectations analytically for ≥5 distributions; derive variance from definition; generate samples and verify empirical moments match theoretical values (within 5% for n≥1000); use MGFs to derive moments; demonstrate Central Limit Theorem convergence with visualizations.
- 🛠️ How: `scipy.stats` for distributions; `np.random.binomial`, `np.random.poisson`, `np.random.normal`; empirical mean: `np.mean(samples)`; plot sampling distributions; CLT: plot standardized sample means for increasing n.

Week 26 — Probability Ch. 5–7 (Markov Chains, Random Walks)
- 📖 [Probability (PDF)](https://math.dartmouth.edu/~prob/prob/prob.pdf)
- 🧪 Practice: Understand Markov chain fundamentals: states, transitions, transition matrices; compute stationary distributions; classify states (transient, recurrent, absorbing); work with random walks; understand gambler's ruin problem; explore applications.
- ✅ Pass: Implement discrete-time Markov chain simulator; compute n-step transition probabilities via matrix powers; find stationary distribution by solving πP = π; classify states and compute expected hitting times; simulate random walks and verify theoretical properties (e.g., return probabilities); solve gambler's ruin analytically and verify via simulation.
- 🛠️ How: Transition matrix: `P = np.array([[p11, p12,...], [...]])`; n-step: `np.linalg.matrix_power(P, n)`; stationary: eigenvalue problem with `np.linalg.eig`, find eigenvector for λ=1; simulation: iterate `state = np.random.choice(states, p=P[state])`; random walk: `positions = np.cumsum(steps)`.

Week 27 — Probability Ch. 8–10 (Law of Large Numbers, Limit Theorems)
- 📖 [Probability (PDF)](https://math.dartmouth.edu/~prob/prob/prob.pdf)
- 🧪 Practice: Understand weak and strong law of large numbers; master Central Limit Theorem and its applications; work with generating functions for sums; understand convergence concepts; apply limit theorems to approximation problems.
- ✅ Pass: Demonstrate Law of Large Numbers: plot sample mean convergence to theoretical mean for increasing sample sizes; verify CLT: show standardized sum converges to Normal via QQ plots and hypothesis tests for n=[10,30,100,1000]; use generating functions to compute distribution of sums; apply continuity correction for Normal approximation to Binomial; compute confidence intervals using CLT.
- 🛠️ How: LLN: `running_mean = np.cumsum(samples)/np.arange(1, n+1)`; plot vs theoretical mean; CLT: `(sum(samples) - n*mu)/(sigma*sqrt(n))` should be N(0,1); `scipy.stats.normaltest` for normality; `scipy.stats.probplot` for QQ plots; confidence intervals: `mean ± z*SE`.

Week 28 — Phase 4 — End-to-End Mini Project
- 📖 Activities: Phase 4 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 4.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 5 · Statistics Fundamentals — Weeks 29–35</b></summary>

Week 29 — Think Stats Ch. 1
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Explore a dataset; compute summary statistics; build histograms and PMFs; construct ECDFs.
- ✅ Pass: Implement ECDF from scratch on real data; verify it is non-decreasing and ends at 1.0; overlay histogram and ECDF to compare distributional insights; interpret outliers.
- 🛠️ How: `np.sort`; `np.arange(1,n+1)/n`; `plt.step` for ECDF; `plt.hist` for histogram.

Week 30 — Think Stats Ch. 2
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Compute central tendency (mean, median, mode) and spread (variance, std, range, IQR); explore effect of outliers on these measures.
- ✅ Pass: Compare mean/SD vs median/MAD/IQR on 2 datasets (one symmetric, one skewed); explain when each measure is appropriate; show outlier impact graphically.
- 🛠️ How: `np.mean`, `np.median`, `np.std`; `scipy.stats.median_abs_deviation`; `np.percentile` for IQR.

Week 31 — Think Stats Ch. 3–4
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Work with CDFs and PMFs; model data with probability distributions; compare empirical vs theoretical distributions.
- ✅ Pass: Fit data to common distributions (Normal, Exponential); use CDF plots to assess fit; compute percentiles and quantiles; explain when to use PMF vs CDF.
- 🛠️ How: `scipy.stats.norm.fit`, `scipy.stats.expon`; `probplot` for QQ plots; CDF comparison plots.

Week 32 — Think Stats Ch. 5–6
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Model data with analytical distributions; explore relationships between variables; compute conditional probabilities.
- ✅ Pass: Fit a parametric model to real data; compute and interpret correlation and covariance; demonstrate conditional probability with a contingency table.
- 🛠️ How: `scipy.stats` distribution fitting; `np.corrcoef`; `pd.crosstab` for contingency tables.

Week 33 — Think Stats Ch. 7–8
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Estimate parameters with confidence intervals; perform hypothesis tests; understand p-values and statistical significance.
- ✅ Pass: Compute confidence intervals via bootstrap and analytical methods; run a hypothesis test; simulate to show Type I error ≈ α; produce a power curve for detecting effect sizes.
- 🛠️ How: Bootstrap resampling; `scipy.stats.ttest_ind`; simulation to count rejections under H₀ and H₁.

Week 34 — Think Stats Ch. 9–10 (+wrap)
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Explore linear relationships; fit simple and multiple regression; interpret coefficients; check regression assumptions.
- ✅ Pass: Fit OLS regression; interpret R², coefficients, and p-values; produce diagnostic plots (residuals vs fitted, QQ plot); compute VIFs and flag multicollinearity.
- 🛠️ How: Use `scipy.stats.linregress` for simple regression; compute multiple regression coefficients via normal equations `np.linalg.inv(X.T @ X) @ X.T @ y`.
Week 35 — Phase 5 — End-to-End Mini Project
- 📖 Activities: Phase 5 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 5.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 6 · Mathematical Statistics — Weeks 36–46</b></summary>

Week 36 — Freund's Ch. 1–2: Introduction & Probability Distributions
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Review probability foundations for statistical inference; master probability distributions (discrete and continuous); understand distribution functions, expectation, variance, and moment-generating functions; work with common distributions (binomial, Poisson, normal, exponential, gamma).
- ✅ Pass: Solve ≥15 problems from Chapters 1-2 covering: probability axioms, conditional probability, distribution functions (PMF, PDF, CDF), expectation and variance calculations, moment-generating functions, and distribution properties; verify ≥5 solutions analytically and compare with simulation; compute MGFs and use them to derive moments for ≥3 distributions.
- 🛠️ How: Use `scipy.stats` for distribution verification; simulate samples to verify theoretical moments; plot empirical vs theoretical distributions; derive MGFs by hand and verify moment formulas; `np.mean`, `np.var` for empirical moments.

Week 37 — Freund's Ch. 3–4: Multivariate Distributions & Sampling Distributions
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Master joint, marginal, and conditional distributions; compute covariance and correlation; understand independence; derive sampling distributions of sample mean and sample variance; master chi-square, t, and F distributions; understand Central Limit Theorem applications.
- ✅ Pass: Solve ≥20 problems from Chapters 3-4 covering: joint and marginal distributions, conditional distributions, covariance and correlation computations, independence tests, sampling distribution derivations; derive sampling distribution of sample mean for normal populations; prove and verify that (n-1)S²/σ² follows chi-square distribution; demonstrate CLT convergence with simulations showing standardized sample means converge to N(0,1) for n ≥ 30.
- 🛠️ How: Compute joint probabilities via integration/summation; `scipy.stats.chi2`, `scipy.stats.t`, `scipy.stats.f` for theoretical distributions; simulate samples and compute sample statistics; plot sampling distributions; verify theoretical results via simulation with n=1000 iterations.

Week 38 — Freund's Ch. 5: Point Estimation
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Understand properties of point estimators (unbiasedness, consistency, efficiency, sufficiency); master method of moments and maximum likelihood estimation; compute Fisher information and Cramér-Rao lower bound; understand sufficient and complete statistics; apply Rao-Blackwell theorem.
- ✅ Pass: Solve ≥15 problems from Chapter 5 covering: method of moments estimation for ≥3 distributions, maximum likelihood estimation with analytic solutions, Fisher information computation, verification of unbiasedness via expectation, efficiency comparisons using CRLB, sufficiency verification using factorization theorem; derive MLE for ≥5 parametric families; compute Fisher information and verify CRLB for variance of unbiased estimators; implement Rao-Blackwell improvement for an estimator.
- 🛠️ How: Solve likelihood equations analytically; compute `E[θ̂]` to check unbiasedness; Fisher information: `I(θ) = -E[∂²log L/∂θ²]`; CRLB: `Var(θ̂) ≥ 1/I(θ)`; numerical verification via simulation; implement factorization to find sufficient statistics.

Week 39 — Freund's Ch. 6: Interval Estimation
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Construct confidence intervals for means (known and unknown variance), proportions, variances, and differences; understand confidence coefficient interpretation; apply large-sample approximations; construct confidence intervals for parameters of common distributions; understand relationship between confidence intervals and hypothesis tests.
- ✅ Pass: Solve ≥15 problems from Chapter 6 constructing: confidence intervals for population mean (σ known and unknown), confidence intervals for proportions using normal approximation, confidence intervals for variance using chi-square distribution, confidence intervals for difference of means (independent and paired samples), two-sided and one-sided intervals; verify coverage probability via simulation (generate 1000 samples, construct CIs, verify ~95% contain true parameter); demonstrate duality between CIs and hypothesis tests.
- 🛠️ How: Normal CI: `x̄ ± z*σ/√n` (σ known); t-based: `x̄ ± t*s/√n` (σ unknown); proportion: `p̂ ± z*√(p̂(1-p̂)/n)`; variance: `[(n-1)s²/χ²_upper, (n-1)s²/χ²_lower]`; simulate samples with `np.random.normal`, compute CIs, check coverage; `scipy.stats.t.ppf`, `scipy.stats.chi2.ppf` for critical values.

Week 40 — Freund's Ch. 7: Hypothesis Testing
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Master hypothesis testing framework (null and alternative hypotheses, Type I and II errors, power); perform z-tests and t-tests for means; chi-square tests for variance; understand p-values and significance levels; compute power functions; understand Neyman-Pearson lemma and likelihood ratio tests.
- ✅ Pass: Solve ≥20 problems from Chapter 7 covering: one-sample and two-sample tests for means, one-sided and two-sided tests, tests for proportions, tests for variance; compute Type I error probability (α) and Type II error probability (β); derive and plot power functions showing power vs true parameter value; apply Neyman-Pearson lemma to find most powerful tests; conduct likelihood ratio tests; verify Type I error rate via simulation (generate data under H₀, compute rejection rate ≈ α).
- 🛠️ How: Test statistic: `z = (x̄-μ₀)/(σ/√n)` or `t = (x̄-μ₀)/(s/√n)`; p-value: `P(|Z|>|z_obs|)` for two-sided; power: `P(reject H₀ | H₁ true)`; simulate under H₀ and H₁ to compute α and power; plot power curve; `scipy.stats` for test statistics and p-values; implement LRT: `λ = L(θ₀)/L(θ̂_MLE)`.

Week 41 — Freund's Ch. 8: Inferences About Means
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Conduct inferences about single means (t-test), difference of means (independent samples), paired samples; understand pooled vs unpooled variance; check assumptions (normality, equal variance); apply Welch's t-test for unequal variances; understand one-way ANOVA for comparing multiple means.
- ✅ Pass: Solve ≥15 problems from Chapter 8 including: one-sample t-tests with confidence intervals, two-sample t-tests (pooled and unpooled), paired t-tests, one-way ANOVA with ≥3 groups; verify assumptions using normality tests (Shapiro-Wilk) and equal variance tests (Levene's, Bartlett's); conduct post-hoc comparisons after ANOVA; compute effect sizes (Cohen's d); reproduce ANOVA F-test result manually from sum of squares decomposition.
- 🛠️ How: `scipy.stats.ttest_1samp`, `ttest_ind`, `ttest_rel`; pooled variance: `s²_p = ((n₁-1)s₁² + (n₂-1)s₂²)/(n₁+n₂-2)`; Welch's t-test for unequal variances; `scipy.stats.f_oneway` for ANOVA; manual ANOVA: compute SSB (between), SSW (within), MSB, MSW, F = MSB/MSW; `scipy.stats.shapiro`, `scipy.stats.levene`; post-hoc: Tukey HSD or Bonferroni correction.

Week 42 — Freund's Ch. 9–10: Inferences About Proportions & Variances
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Conduct inferences about single proportions and differences; perform chi-square tests for variance; test equality of two variances (F-test); apply goodness-of-fit tests; understand contingency table analysis and tests of independence; compute relative risk and odds ratios.
- ✅ Pass: Solve ≥15 problems from Chapters 9-10 covering: confidence intervals and hypothesis tests for single proportions, two-proportion z-tests, chi-square goodness-of-fit tests, chi-square tests of independence with contingency tables, F-tests for equality of variances; verify normal approximation validity (np ≥ 5, n(1-p) ≥ 5); conduct Fisher's exact test when cell counts are small; compute and interpret odds ratios and relative risk; verify chi-square test Type I error via simulation.
- 🛠️ How: Proportion test: `z = (p̂-p₀)/√(p₀(1-p₀)/n)`; two-proportion: pooled estimate; chi-square GOF: `χ² = Σ(O-E)²/E`; independence: `χ² = Σ(O_ij - E_ij)²/E_ij` where `E_ij = (row_i × col_j)/n`; F-test: `F = s₁²/s₂²`; `scipy.stats.chi2_contingency`, `fisher_exact`; odds ratio: `OR = (a×d)/(b×c)` for 2×2 table.

Week 43 — Freund's Ch. 11: Regression & Correlation
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Understand simple linear regression model assumptions; derive least squares estimators; conduct inference on slope and intercept; test for significance of regression; understand correlation coefficient and its inference; distinguish correlation from causation; compute residuals and check assumptions.
- ✅ Pass: Solve ≥15 problems from Chapter 11 including: fitting simple linear regression using least squares, deriving normal equations, computing confidence intervals for slope and intercept, testing H₀: β₁=0, constructing confidence and prediction intervals for Y, computing correlation coefficient and testing its significance; verify regression assumptions via residual plots (residuals vs fitted, QQ plot, scale-location); derive least squares estimators from first principles; demonstrate Gauss-Markov theorem via simulation; compute R² and interpret as proportion of variance explained.
- 🛠️ How: Least squares: `β̂₁ = Σ(xᵢ-x̄)(yᵢ-ȳ)/Σ(xᵢ-x̄)²`, `β̂₀ = ȳ - β̂₁x̄`; `statsmodels.api.OLS` for inference; confidence interval for β₁: `β̂₁ ± t*SE(β̂₁)`; prediction interval wider than confidence interval; correlation test: `t = r√(n-2)/√(1-r²)`; residual diagnostics with `plt.scatter`; verify homoscedasticity and normality.

Week 44 — Freund's Ch. 12: Analysis of Variance (ANOVA)
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Master one-way and two-way ANOVA; understand completely randomized designs and randomized block designs; perform multiple comparisons; understand ANOVA assumptions and diagnostics; compute effect sizes; understand relationship between ANOVA and regression.
- ✅ Pass: Solve ≥15 problems from Chapter 12 including: one-way ANOVA with ≥3 groups, two-way ANOVA with interaction, randomized complete block design, multiple comparison procedures (Tukey HSD, Bonferroni), effect size calculations (η², ω²); manually compute ANOVA table (SS, df, MS, F) and verify with software; check assumptions (normality, homogeneity of variance, independence); demonstrate equivalence of one-way ANOVA F-test to two-sample t-test when k=2 groups; interpret interaction plots for two-way ANOVA.
- 🛠️ How: One-way ANOVA: `F = MSB/MSW` where `MSB = SSB/(k-1)`, `MSW = SSW/(n-k)`; two-way: include main effects and interaction; Compute sums of squares (SST, SSB, SSW) manually using `np.sum` and `np.mean`; calculate F-statistic and compare to `scipy.stats.f`; calculate effects strictly via formulas.

Week 45 — Freund's Ch. 13–14: Review & Nonparametric Methods
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Review all statistical inference concepts; master distribution-free nonparametric methods: sign test, Wilcoxon signed-rank test, Mann-Whitney U test, Kruskal-Wallis test, runs test; understand when to use parametric vs nonparametric tests; compare power of parametric vs nonparametric tests.
- ✅ Pass: Solve ≥15 problems covering: sign test for median, Wilcoxon signed-rank test for paired data, Mann-Whitney U test for two independent samples, Kruskal-Wallis test for multiple groups, Spearman rank correlation, runs test for randomness; compare results of parametric vs nonparametric tests on same data; conduct power analysis via simulation showing power loss of nonparametric tests under normality and power gain under non-normality; verify test assumptions and justify method selection; integrate all inference concepts from Chapters 1-14 in a comprehensive analysis.
- 🛠️ How: `scipy.stats.wilcoxon`, `mannwhitneyu`, `kruskal`, `spearmanr`; sign test: compare median to hypothesized value using binomial; runs test for independence; simulate data from normal and heavy-tailed distributions, apply both parametric and nonparametric tests, compare power; decision tree for test selection based on assumptions.
Week 46 — Phase 6 — End-to-End Mini Project
- 📖 Activities: Phase 6 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 6.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 7 · Applied Multivariate Statistics — Weeks 47–50</b></summary>

Week 47 — Principal Components Analysis
- 📖 Activities: [PSU STAT 505 Lesson 11](https://online.stat.psu.edu/stat505/lesson/11)
- 🧪 Practice: Perform PCA on correlation and covariance matrices.
- ✅ Pass: Apply PCA to dataset with ≥6 variables; create scree plot.
- 🛠️ How: Compute covariance using `np.cov`, extract eigenvectors with `np.linalg.eig`; standardize manually `(X - X.mean(axis=0)) / X.std(axis=0)`.

Week 48 — Cluster Analysis (Hierarchical & K-Means)
- 📖 Activities: [PSU STAT 505 Lesson 14](https://online.stat.psu.edu/stat505/lesson/14)
- 🧪 Practice: Apply hierarchical clustering with different linkage methods.
- ✅ Pass: Perform hierarchical clustering; apply k-means with multiple k values.
- 🛠️ How: `scipy.cluster.hierarchy` for hierarchical clustering.

Week 49 — Discriminant Analysis (LDA & QDA)
- 📖 Activities: [PSU STAT 505 Lesson 10](https://online.stat.psu.edu/stat505/lesson/10)
- 🧪 Practice: Perform linear and quadratic discriminant analysis.
- ✅ Pass: Apply LDA and QDA to classification problem.
- 🛠️ How: Compute class means and pooled covariance matrix manually; evaluate classification rules via matrix operations.
Week 50 — Phase 7 — End-to-End Mini Project
- 📖 Activities: Phase 7 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 7.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 8 · Bayesian Statistics & Missing Data — Weeks 51–53</b></summary>

Week 51 — Think Bayes (Ch. 1–14, paced)
- 📖 [Think Bayes](https://allendowney.github.io/ThinkBayes2/)
- 🧪 Practice: Apply Bayes' theorem to update beliefs; implement conjugate prior models (Beta-Binomial, Gamma-Poisson, Normal-Normal); perform posterior predictive checks; compare models.
- ✅ Pass (weekly): Implement a Bayesian model aligned with the chapter's topic; show prior sensitivity analysis (vary prior parameters and observe posterior changes); generate posterior predictive samples and compare to observed data using a suitable test statistic.
- 🛠️ How: Use analytical posteriors when available; for PPC, draw samples from posterior, then from likelihood, and compare summary stats to data.

Week 52 — Flexible Imputation of Missing Data (complete)
- 📖 [FIMD](https://stefvanbuuren.name/fimd/)
- 🧪 Practice: Missingness mechanisms; MICE; sensitivity (as in book)
- ✅ Pass (weekly): Run MICE (m≥5) on a dataset; report pooled estimates per Rubin’s rules; compare to complete-case; perform delta-adjustment sensitivity where relevant.
- 🛠️ How: Write a basic iterative imputation loop in Python: estimate missing values using chained equations and simple models.
Week 53 — Phase 8 — End-to-End Mini Project
- 📖 Activities: Phase 8 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 8.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 9 · Statistical Learning with Python (ISLP) — Weeks 54–64</b></summary>

Week 54 — ISLP Ch. 1–2 (Intro + Statistical Learning)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand the statistical learning framework; implement train/test splits; explore the bias-variance trade-off with KNN at different k values.
- ✅ Pass: On a dataset, demonstrate how training error decreases with model complexity while test error shows U-shape; implement 5-fold CV and compare to hold-out estimate; discuss flexibility vs interpretability.
- 🛠️ How: `train_test_split`; `KFold`/`cross_val_score`; vary KNN's k parameter; plot training vs test error curves.

Week 55 — ISLP Ch. 3 (Linear Regression)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Fit simple and multiple linear regression; interpret coefficients; add interaction and polynomial terms; assess model fit with residual diagnostics.
- ✅ Pass: Fit OLS with and without interaction/polynomial terms; compare R² vs adjusted R²; produce residual plots; select optimal polynomial degree via CV; interpret coefficient confidence intervals.
- 🛠️ How: `LinearRegression`; `PolynomialFeatures`; `cross_val_score`; `statsmodels` for CIs; residual diagnostics.

Week 56 — ISLP Ch. 4 (Classification)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Implement logistic regression; understand LDA/QDA assumptions; apply KNN for classification; explore classification metrics beyond accuracy.
- ✅ Pass: Compare logistic regression, LDA, QDA, and KNN using stratified 5-fold CV; report confusion matrix, precision, recall, and ROC-AUC; select optimal classification threshold based on problem context.
- 🛠️ How: `LogisticRegression`; `LinearDiscriminantAnalysis`; `QuadraticDiscriminantAnalysis`; `KNeighborsClassifier`; `roc_curve` for threshold selection.

Week 57 — ISLP Ch. 5 (Resampling Methods)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Compare validation approaches: hold-out, LOOCV, k-fold CV; use bootstrap for uncertainty estimation; understand variance-bias trade-off in resampling.
- ✅ Pass: Compare test error estimates from LOOCV vs 5-fold vs 10-fold CV; implement bootstrap to estimate coefficient standard errors; compare bootstrap SEs to analytic SEs.
- 🛠️ How: `LeaveOneOut`; `KFold`; implement bootstrap loop with `np.random.choice`; fix seeds for reproducibility.

Week 58 — ISLP Ch. 6 (Model Selection & Regularization)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand the motivation for regularization; implement ridge and lasso regression; interpret coefficient shrinkage and sparsity; tune regularization parameter via CV.
- ✅ Pass: Plot ridge and lasso coefficient paths as λ varies; select optimal λ via CV; compare test error of OLS vs ridge vs lasso; explain when lasso produces sparse solutions.
- 🛠️ How: `Ridge`; `Lasso`; `RidgeCV`; `LassoCV`; `StandardScaler` (scale features first); `lasso_path` for path plots.

Week 59 — ISLP Ch. 7 (Beyond Linearity)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Move beyond linearity with polynomial regression, step functions, and splines; understand degrees of freedom; fit GAM-style models.
- ✅ Pass: Fit polynomial, step function, and spline models; compare their flexibility and test errors; produce partial dependence plots; select appropriate number of knots/degrees via CV.
- 🛠️ How: `PolynomialFeatures`; `SplineTransformer`; `pd.cut` for step functions; compare MSE on held-out data.

Week 60 — ISLP Ch. 8 (Tree-Based Methods)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Fit decision trees; understand bagging and the random forest algorithm; implement gradient boosting; interpret tree-based models.
- ✅ Pass: Fit and prune a decision tree; compare single tree vs random forest vs gradient boosting on test error; show OOB error for RF; plot feature importances and partial dependence plots.
- 🛠️ How: `DecisionTreeClassifier/Regressor`; `RandomForestClassifier/Regressor`; `GradientBoostingClassifier/Regressor`; `permutation_importance`; `plot_partial_dependence`.

Week 61 — ISLP Ch. 9 (Support Vector Machines)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand maximal margin classifiers and support vectors; fit SVMs with linear and non-linear kernels; tune hyperparameters (C, gamma).
- ✅ Pass: Fit SVM with linear and RBF kernels; tune C and gamma via grid search with CV; visualize decision boundaries on 2D data; identify and highlight support vectors; compare to logistic regression.
- 🛠️ How: `SVC`; `GridSearchCV`; `plt.contourf` for decision boundaries; access `support_vectors_` attribute.

Week 62 — ISLP Ch. 10 (Unsupervised Learning)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Perform dimensionality reduction with PCA; apply k-means and hierarchical clustering; understand the importance of scaling; evaluate clustering quality.
- ✅ Pass: Apply PCA and plot cumulative explained variance; choose number of components; cluster with k-means (elbow method for k) and hierarchical clustering (dendrogram); evaluate with silhouette score and compare cluster stability across random seeds.
- 🛠️ How: `StandardScaler` (always scale first); `PCA`; `KMeans` with inertia plots; `AgglomerativeClustering`; `dendrogram`; `silhouette_score`.

Week 63 — ISLP Labs/Wrap-up
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Complete an end-to-end ML project using techniques from all ISLP chapters: EDA, preprocessing, model selection, hyperparameter tuning, evaluation, and interpretation.
- ✅ Pass: Deliver a reproducible notebook with proper train/test split, cross-validation, model comparison, hyperparameter tuning, error analysis, and a 1-page summary documenting decisions, limitations, and risks.
- 🛠️ How: `Pipeline`; `ColumnTransformer` for mixed feature types; `GridSearchCV`/`RandomizedSearchCV`; fixed `random_state` throughout; clean documentation.
Week 64 — Phase 9 — End-to-End Mini Project
- 📖 Activities: Phase 9 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 9.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 10 · Data Mining — Weeks 65–67</b></summary>

Week 65 — Data Preprocessing & Frequent Patterns
- 📖 Activities: [Data Mining 3e (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- 🧪 Practice: Per-chapter algorithmic work matching chapters on preprocessing and Apriori.
- ✅ Pass: Implement Apriori or integrate a library; verify exact itemsets.
- 🛠️ How: Construct synthetic datasets; assert rules match expectations.

Week 66 — Classification & Cluster Analysis
- 📖 Activities: [Data Mining 3e (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- 🧪 Practice: Evaluate decision trees and k-means/DBSCAN algorithms.
- ✅ Pass: Implement DBSCAN or tree logic and compare against a library.
- 🛠️ How: Compare custom tree splits vs library nodes.
Week 67 — Phase 10 — End-to-End Mini Project
- 📖 Activities: Phase 10 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 10.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 11 · Classical Machine Learning — Weeks 68–70</b></summary>

Week 68 — PRML (Ch. 1–13 + review)
- 📖 [PRML (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- 🧪 Practice: Implement core algorithms from each chapter from scratch: probability distributions, linear models, neural networks, kernel methods, graphical models, mixture models, EM algorithm, approximate inference, and sampling methods.
- ✅ Pass (weekly): Implement the chapter's focal algorithm from scratch; verify correctness by comparing to sklearn/scipy baseline (within 2-5% accuracy); document mathematical derivations; use fixed seeds for reproducibility.
- 🛠️ How: Use NumPy for implementations; sklearn only as verification oracle; work on toy datasets; keep detailed notes linking code to book equations.

Week 69 — Interpretable ML (complete)
- 📖 [Interpretable ML](https://christophm.github.io/interpretable-ml-book/)
- 🧪 Practice: Apply model-agnostic interpretation methods: PDP, ICE, permutation importance, LIME, SHAP; understand intrinsically interpretable models; explore feature interaction methods.
- ✅ Pass (weekly): For a trained model, produce PDP/ICE plots for top features; compute permutation importance; generate SHAP values for individual predictions; write a 1-page analysis comparing methods' stability across 3 bootstrap resamples.
- 🛠️ How: `sklearn.inspection.PartialDependenceDisplay`; `permutation_importance`; `shap.Explainer`; compare explanations across train/test sets.
Week 70 — Phase 11 — End-to-End Mini Project
- 📖 Activities: Phase 11 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 11.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 12 · Elements of Statistical Learning — Weeks 71–81</b></summary>

Week 71 — ESL Ch. 1–3: Introduction & Linear Methods
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Review statistical learning framework; master linear regression theory (bias-variance decomposition, Gauss-Markov theorem); implement subset selection, ridge, and lasso from scratch; understand effective degrees of freedom.
- ✅ Pass: Derive bias-variance decomposition analytically; prove Gauss-Markov theorem; implement best subset selection via exhaustive search for p ≤ 10; implement ridge and lasso with coordinate descent; compute effective degrees of freedom `df(λ) = tr[X(XᵀX + λI)⁻¹Xᵀ]` and verify empirically; compare subset selection, ridge, and lasso on test error and coefficient paths.
- 🛠️ How: Bias-variance: `E[(Y-f̂)²] = Bias²(f̂) + Var(f̂) + σ²`; Gauss-Markov: show OLS has minimum variance among linear unbiased estimators; subset selection: iterate over all 2^p subsets; ridge: `β̂ = (XᵀX + λI)⁻¹Xᵀy`; lasso coordinate descent: soft thresholding `S(z,γ) = sign(z)(|z|-γ)₊`; effective df from hat matrix trace.

Week 72 — ESL Ch. 4–5: Linear Classification & Basis Expansions
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Master linear discriminant analysis (LDA), quadratic discriminant analysis (QDA), logistic regression, and separating hyperplanes; understand basis expansions (polynomial, splines, wavelets); implement natural cubic splines.
- ✅ Pass: Derive LDA decision boundary assuming equal covariance; implement QDA allowing separate covariances; fit logistic regression via Newton-Raphson (IRLS); implement linear separating hyperplane via perceptron algorithm; construct natural cubic spline basis manually and fit regression; compare polynomial vs spline fits showing boundary bias and variance; derive and verify degrees of freedom for smoothing splines.
- 🛠️ How: LDA: estimate class means μₖ and pooled covariance Σ, classify via `argmax_k log P(G=k) - ½(x-μₖ)ᵀΣ⁻¹(x-μₖ)`; QDA: separate Σₖ for each class; logistic IRLS: iterate `β := β + (XᵀWX)⁻¹Xᵀ(y-p)` where W=diag(p(1-p)); perceptron: `β := β + ηyᵢxᵢ` for misclassified points; natural spline: impose constraints for linearity beyond boundary knots.

Week 73 — ESL Ch. 6–7: Kernel Methods & Model Assessment
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Understand kernel smoothing and local regression; implement k-nearest neighbors, Nadaraya-Watson estimator, local polynomial regression; master cross-validation theory (GCV, leave-one-out shortcuts); understand bootstrap for model selection; derive and implement optimism estimators (Cp, AIC, BIC).
- ✅ Pass: Implement Nadaraya-Watson kernel regression with Gaussian kernel and bandwidth selection via CV; implement local linear regression (LOESS) and show boundary bias correction compared to Nadaraya-Watson; derive and implement leave-one-out CV shortcut for linear smoothers via hat matrix; implement 0.632 bootstrap estimator; compute Cp, AIC, BIC for nested models and verify consistency of BIC; compare all model selection criteria on a common dataset.
- 🛠️ How: Nadaraya-Watson: `f̂(x₀) = Σ K((xᵢ-x₀)/h)yᵢ / Σ K((xᵢ-x₀)/h)`; LOESS: weighted least squares in local neighborhood; LOO shortcut: `CV = (1/n)Σ(yᵢ-f̂(xᵢ))²/(1-hᵢᵢ)²` where hᵢᵢ is diagonal of hat matrix; Cp: `RSS/σ² + 2d`; AIC: `-2log-likelihood + 2d`; BIC: `-2log-likelihood + log(n)d`.

Week 74 — ESL Ch. 8–9: Model Inference & Additive Models
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Understand bootstrap for inference (standard errors, confidence intervals, percentile and BCa methods); implement permutation tests; master generalized additive models (GAMs) with backfitting algorithm; understand tree-based models and CART algorithm.
- ✅ Pass: Implement bootstrap confidence intervals (normal, percentile, BCa) and compare coverage on simulations; implement permutation test for independence and verify Type I error rate; implement GAM backfitting algorithm from scratch for additive model with spline components; fit and prune CART tree using cost-complexity pruning; compare tree to GAM on same dataset; prove backfitting convergence for additive models.
- 🛠️ How: BCa: bias-correction `z₀` and acceleration `a` from jackknife; percentile: 2.5% and 97.5% quantiles of bootstrap distribution; permutation: shuffle one variable, recompute test statistic; backfitting: iterate `f̂ⱼ := S_j[Y - Σₖ≠ⱼf̂ₖ]` where Sⱼ is smoother; CART: recursive binary splits minimizing RSS or Gini; cost-complexity: `min_T Σ(yᵢ-ŷₜ)² + α|T|`.

Week 75 — ESL Ch. 10: Boosting & Additive Models
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Understand AdaBoost algorithm and its connection to exponential loss; implement gradient boosting from scratch; understand forward stagewise additive modeling; derive L2Boosting and show connection to gradient descent in function space; implement shrinkage and early stopping for regularization.
- ✅ Pass: Implement AdaBoost with decision stumps from scratch; show connection to exponential loss by deriving weight updates; implement gradient boosting with squared loss and deviance loss; demonstrate that gradient boosting is steepest descent in function space; compare learning rates and early stopping for regularization; implement stochastic gradient boosting (subsampling); produce learning curves showing train/validation error vs boosting iterations.
- 🛠️ How: AdaBoost: iterate `err_m = Σw_i I(y_i≠G_m(x_i))/Σw_i`, `α_m = log((1-err_m)/err_m)`, `w_i := w_i exp(α_m I(y_i≠G_m))`, final: `G = sign(Σα_m G_m)`; gradient boosting: `f_m = f_{m-1} + ν·h_m` where `h_m` fits residuals `-∂L/∂f`; derive for squared loss: residuals are `y-f`; for deviance: residuals are gradients of log-likelihood.

Week 76 — ESL Ch. 11–12: Neural Networks & Support Vector Machines
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Implement feedforward neural network with backpropagation from scratch; understand universal approximation; derive and implement weight decay and early stopping; implement SVM via quadratic programming; understand kernel trick and mercer kernels; compare SVM to logistic regression and neural networks.
- ✅ Pass: Implement multi-layer perceptron with one hidden layer from scratch including backpropagation; verify gradient computation with finite differences; train on classification and regression tasks with weight decay; implement SVM dual problem and solve with quadratic programming; implement kernel SVM with RBF kernel; visualize decision boundaries; compare SVM, logistic regression, and neural network on nonlinearly separable data; demonstrate kernel trick equivalence.
- 🛠️ How: Backprop: forward pass compute activations, backward pass compute gradients via chain rule; weight update: `w := w - η∂L/∂w`; SVM dual: `max Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼxᵢᵀxⱼ` subject to `0≤α≤C`, `Σαᵢyᵢ=0`; kernel trick: replace `xᵢᵀxⱼ` with `K(xᵢ,xⱼ)`; use `cvxopt.solvers.qp` or `scipy.optimize.minimize` for QP.

Week 77 — ESL Ch. 13–14: Prototype Methods & Unsupervised Learning
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Implement k-means, k-medoids, and Gaussian mixture models (GMM) via EM; understand learning vector quantization (LVQ); implement hierarchical clustering with different linkages; understand self-organizing maps (SOM); derive EM algorithm for GMM from first principles.
- ✅ Pass: Implement k-means from scratch and prove convergence (monotonic decrease of objective); implement k-medoids (PAM algorithm); derive EM algorithm for GMM (E-step: compute responsibilities, M-step: update parameters); implement GMM-EM and compare to k-means; implement hierarchical clustering with single, complete, and average linkage; compute cophenetic correlation; implement LVQ and compare to k-means; visualize dendrograms and cluster quality metrics (silhouette, Davies-Bouldin).
- 🛠️ How: k-means: iterate assign-to-nearest-centroid, update-centroids; objective: `Σᵢ Σₖ rᵢₖ||xᵢ-μₖ||²`; EM for GMM: `γᵢₖ = πₖ N(xᵢ|μₖ,Σₖ) / Σⱼ πⱼ N(xᵢ|μⱼ,Σⱼ)`, update `πₖ = Σγᵢₖ/n`, `μₖ = Σγᵢₖxᵢ/Σγᵢₖ`, `Σₖ = Σγᵢₖ(xᵢ-μₖ)(xᵢ-μₖ)ᵀ/Σγᵢₖ`; hierarchical: `scipy.cluster.hierarchy`; cophenetic: correlation between pairwise distances and dendrogram heights.

Week 78 — ESL Ch. 15: Random Forests
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Understand bagging and its variance reduction; implement random forests from scratch with bootstrap sampling and feature subsampling; compute out-of-bag (OOB) error as unbiased test error estimate; understand variable importance measures (permutation, Gini); analyze effect of correlation between trees.
- ✅ Pass: Implement random forest from scratch (bootstrap samples, random feature subset at each split, majority vote/averaging); compute OOB error and compare to test error and cross-validation; implement variable importance via permutation (OOB samples) and Gini decrease; demonstrate variance reduction compared to single tree via bias-variance decomposition; analyze effect of number of features sampled (mtry) on correlation between trees and forest performance; produce partial dependence plots.
- 🛠️ How: Random forest: build B trees each on bootstrap sample with feature subsampling (√p for classification, p/3 for regression); OOB error: for each observation, average predictions from trees not containing it in bootstrap sample; permutation importance: shuffle feature j in OOB data, compute increase in OOB error; Gini importance: sum Gini decrease when splitting on feature across all trees; correlation: compute pairwise correlation of tree predictions.

Week 79 — ESL Ch. 16–17: Ensemble Learning & Graphical Models
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Master ensemble methods theory; understand stacking and super learner; implement Bayesian model averaging; explore undirected graphical models (Markov networks); understand conditional independence and the Hammersley-Clifford theorem; implement graphical lasso for sparse inverse covariance estimation.
- ✅ Pass: Implement stacked generalization (train meta-learner on out-of-fold predictions); implement super learner with cross-validation-based weighting; compute Bayesian model averaging weights using BIC approximation; implement graphical lasso (L1-penalized precision matrix estimation) and visualize resulting network; test conditional independence using partial correlations; compare ensemble methods (bagging, boosting, stacking) on same dataset with ≥5 base learners; produce detailed analysis of why and when each ensemble method excels.
- 🛠️ How: Stacking: train base learners, collect out-of-fold predictions, train meta-learner on these; super learner: non-negative weights minimizing CV error `min Σ(yᵢ - Σαₖf̂ₖ⁽⁻ⁱ⁾)²` subject to `α≥0`, `Σα=1`; BMA weights: `w_k ∝ exp(-BIC_k/2)`; graphical lasso: `max log det Θ - tr(SΘ) - λ||Θ||₁`; use `sklearn_glasso` or ADMM implementation; zero entries in Θ imply conditional independence.

Week 80 — ESL Ch. 18 & Integration: High-Dimensional Problems
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Understand challenges in high-dimensional settings (p >> n); implement elastic net combining L1 and L2 penalties; understand the Lasso path and LARS algorithm; implement fused lasso for spatial/temporal smoothing; master multiple testing correction (FDR, FWER); understand compressed sensing and restricted isometry property.
- ✅ Pass: Implement elastic net and demonstrate scenarios where it outperforms pure lasso or ridge; implement LARS algorithm and verify equivalence to lasso path; implement fused lasso for 1D signal denoising; apply multiple testing corrections (Bonferroni, Holm, Benjamini-Hochberg) and compare false discovery rates via simulation; demonstrate compressed sensing recovery with RIP-satisfying matrices; integrate ≥5 ESL techniques in comprehensive analysis comparing interpretability, prediction accuracy, computational cost, and theoretical guarantees.
- 🛠️ How: Elastic net: `min ||y-Xβ||² + λ₁||β||₁ + λ₂||β||²`; LARS: forward stagewise that adds most correlated predictor and moves in equiangular direction; fused lasso: `min ||y-β||² + λ₁||β||₁ + λ₂Σ|βᵢ-βᵢ₊₁|`; FDR: Benjamini-Hochberg procedure sorting p-values; compressed sensing: recover sparse signal from few measurements when sensing matrix satisfies RIP; compare convergence and solution paths.
Week 81 — Phase 12 — End-to-End Mini Project
- 📖 Activities: Phase 12 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 12.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 13 · Deep Learning — Weeks 82–85</b></summary>

Week 82 — D2L (Fundamentals)
- 📖 [D2L](https://d2l.ai)
- 🧪 Practice: Topic-specific small models exactly as covered (MLP, CNN, RNN; optimization; regularization; data pipelines)
- ✅ Pass (weekly): Train the chapter’s model variant on a toy dataset with fixed seeds and one controlled ablation (optimizer OR regularization) taught in D2L; log curves/metrics.
- 🛠️ How: Follow D2L’s PyTorch/MXNet examples; fix seeds; keep experiments minimal and reproducible.

Week 83 — The Illustrated Transformer (Bridge)
- 📖 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- 🧪 Practice: Understand the Transformer architecture: self-attention mechanism, multi-head attention, positional encoding, encoder-decoder structure.
- ✅ Pass: Implement self-attention from scratch; verify tensor shapes at each step; implement attention masking; write unit tests for: (1) output shape correctness, (2) masked positions get zero attention, (3) attention weights sum to 1.
- 🛠️ How: Use NumPy or PyTorch; implement Q, K, V projections; scaled dot-product attention; verify with `assert` statements and test cases.

Week 84 — Deep Learning Book (Complete)
- 📖 [Deep Learning Book](https://www.deeplearningbook.org/)
- 🧪 Practice: For each chapter, run a small experiment that demonstrates the chapter’s key concept using building blocks learned in D2L
- ✅ Pass (weekly): Provide a controlled comparison or demonstration plot showing the expected qualitative effect (e.g., different inits, L2 vs dropout, step-size schedules).
- 🛠️ How: Small synthetic or standard toy datasets; fixed seeds; log and compare curves cleanly.
Week 85 — Phase 13 — End-to-End Mini Project
- 📖 Activities: Phase 13 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 13.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 14 · R for Data Science — Weeks 86–87</b></summary>

Week 86 — R4DS (Complete)
- 📖 [R for Data Science (2e)](https://r4ds.hadley.nz)
- 🧪 Practice: Learn R and tidyverse progressively: data import, tidying (pivot_longer/wider), transformation (dplyr verbs), visualization (ggplot2), strings, factors, dates, functions, iteration, and communication (Quarto/RMarkdown).
- ✅ Pass (weekly): Complete a mini-analysis using only functions from chapters covered that week; produce a Quarto/RMarkdown report that renders end-to-end; include at least one visualization and one summary table.
- 🛠️ How: `library(tidyverse)`; `read_csv`; `dplyr` verbs (`filter`, `mutate`, `summarize`, `group_by`); `ggplot2`; `set.seed()` for reproducibility.
Week 87 — Phase 14 — End-to-End Mini Project
- 📖 Activities: Phase 14 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 14.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 15 · Econometrics, Time Series & Financial Econometrics — Weeks 88–91</b></summary>

Week 88 — Basic Econometrics (complete)
- 📖 [Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf)
- 🧪 Practice: Reproduce a worked example per chapter using methods from that chapter only (OLS basics; classical assumption diagnostics; heteroskedasticity/autocorrelation remedies; functional form; limited dependent variables as presented)
- ✅ Pass (weekly): Match the textbook example’s coefficients and standard errors (within rounding) and include one robustness check discussed in that chapter (e.g., robust/HAC SEs when appropriate).
- 🛠️ How: `statsmodels` OLS/GLM, `cov_type="HC3"` or HAC if the chapter addresses it; include diagnostic plots taught there.

Week 89 — Lütkepohl (complete)
- 📖 [Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)
- 🧪 Practice: Implement multivariate time series analysis: VAR model specification, estimation, lag order selection, stability analysis, impulse response functions, forecast error variance decomposition, and cointegration/VECM.
- ✅ Pass (weekly): Fit VAR/VECM to macroeconomic data; select lag order using information criteria; verify stability (roots inside unit circle); compute and plot IRFs with confidence bands; perform Johansen cointegration test when applicable.
- 🛠️ How: `statsmodels.tsa.api.VAR`; `statsmodels.tsa.vector_ar.vecm.VECM`; `irf()` for impulse responses; rolling-window forecasts for evaluation.

Week 90 — Financial Econometrics (complete)
- 📖 [Financial Econometrics (PDF)](https://bashtage.github.io/kevinsheppard.com/files/teaching/mfe/notes/financial-econometrics-2020-2021.pdf)
- 🧪 Practice: Master financial econometrics progressively: volatility modeling (ARCH/GARCH family), multivariate GARCH models, realized volatility and high-frequency data analysis, factor models for asset pricing, portfolio optimization, option pricing and risk management.
- ✅ Pass (weekly): Reproduce examples from the text using methods from each section; implement ARCH/GARCH models and forecast volatility; estimate multivariate GARCH (CCC, DCC, BEKK) and compute risk measures (VaR, ES); analyze high-frequency data and compute realized volatility; apply factor models (CAPM, Fama-French) and optimize portfolios; implement Black-Scholes pricing and calibrate volatility surfaces; verify model specifications using information criteria and diagnostic tests.
- 🛠️ How: `arch` package for GARCH models: `arch_model(returns, vol='GARCH', p=1, q=1).fit()`; multivariate models: `arch.multivariate`; realized volatility from intraday returns; factor regressions: `statsmodels.api.OLS`; portfolio optimization: `scipy.optimize.minimize` with constraints; VaR: `np.percentile(returns, alpha)`; Black-Scholes implementation; diagnostics: Ljung-Box test on residuals and squared residuals; model selection via AIC/BIC.
Week 91 — Phase 15 — End-to-End Mini Project
- 📖 Activities: Phase 15 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 15.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 16 · Causal Inference — Weeks 92–102</b></summary>

Week 92 — Properties of Regression, DAGs, Potential Outcomes
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand Simpson's paradox and collider bias; draw and analyze directed acyclic graphs (DAGs); master potential outcomes framework; understand Average Treatment Effect (ATE) and selection bias.
- ✅ Pass: Implement Simpson's paradox example showing reversal of association; construct ≥3 DAGs identifying confounders, mediators, and colliders; derive ATE under different selection mechanisms; demonstrate selection bias analytically and via simulation.
- 🛠️ How: Use `networkx` or `dagitty` for DAG visualization; simulate counterfactuals with fixed treatment assignments; compute `E[Y¹] - E[Y⁰]` vs observed difference-in-means; show bias = `E[Y⁰|D=1] - E[Y⁰|D=0]`.

Week 93 — Randomized Controlled Trials & Matching
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand randomization inference; implement exact matching, propensity score matching (PSM), and coarsened exact matching; check covariate balance; assess common support.
- ✅ Pass: Analyze an RCT dataset computing ATE with randomization inference (permutation test); implement PSM with ≥3 matching algorithms (nearest neighbor, caliper, kernel); produce balance tables and Love plots before/after matching; check common support with density plots; report treatment effects with bootstrapped standard errors.
- 🛠️ How: Permutation test: shuffle treatment vector 1000+ times, recompute difference-in-means; `sklearn.neighbors.NearestNeighbors` for matching; logistic regression for propensity scores; standardized mean differences for balance; `seaborn.kdeplot` for common support.

Week 94 — Regression Discontinuity Design (RDD)
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand sharp and fuzzy RDD; check continuity assumptions; select bandwidth using cross-validation and optimal methods; test for manipulation of running variable; implement local polynomial regression.
- ✅ Pass: Apply RDD to real or simulated data with a known cutoff; test for discontinuity at the threshold using local linear regression with ≥3 bandwidths; perform McCrary density test for manipulation; produce RDD plots showing outcome vs running variable with fitted lines; report local average treatment effect (LATE) with robust standard errors; conduct placebo tests at false cutoffs.
- 🛠️ How: Local linear regression within bandwidth h: `Y ~ D + (X-c) + D*(X-c)` for |X-c| < h; optimal bandwidth via `rdrobust` (R) or manual cross-validation; McCrary test: fit separate densities left/right of cutoff and test for jump; bootstrap for inference.

Week 95 — Instrumental Variables (IV)
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand endogeneity and IV identification; implement two-stage least squares (2SLS); test instrument relevance and exogeneity; understand weak instruments problem; compute local average treatment effect (LATE) with compliance types.
- ✅ Pass: Identify a valid instrument and justify exclusion restriction; implement 2SLS manually (first stage, second stage) and compare to built-in IV estimator; test instrument strength (F-stat > 10 rule of thumb, Cragg-Donald); perform overidentification test when multiple instruments available; compute LATE and interpret in terms of compliers; conduct sensitivity analysis for violation of exclusion restriction.
- 🛠️ How: Manual 2SLS: regress X on Z (first stage), predict X̂, regress Y on X̂ (second stage); `statsmodels.sandbox.regression.gmm.IV2SLS` or `linearmodels.iv.IV2SLS`; first-stage F-stat for relevance; Hansen J-stat for overidentification; bound analysis for exclusion restriction violations.

Week 96 — Panel Data & Fixed Effects
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand within-group variation; implement fixed effects (FE) and first differences (FD); test fixed vs random effects (Hausman test); handle time-varying treatments; understand parallel trends assumption.
- ✅ Pass: Estimate panel data model with entity and time fixed effects; compare pooled OLS, FE, and random effects; perform Hausman test; demean data manually and verify equivalence to FE estimator; produce event study plots for dynamic treatment effects; test parallel trends visually and formally; cluster standard errors at appropriate level.
- 🛠️ How: FE via demeaning: `Y_it - Ȳ_i = (X_it - X̄_i)β + (ε_it - ε̄_i)`; `linearmodels.panel.PanelOLS` with `entity_effects=True`; Hausman test compares FE vs RE; event study: include leads/lags of treatment; plot coefficients with 95% CIs; cluster SEs: `cov_type='clustered'`.

Week 97 — Difference-in-Differences (DiD)
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Implement canonical 2×2 DiD; test parallel trends assumption; handle staggered treatment adoption; understand two-way fixed effects (TWFE) issues with heterogeneous treatment effects; apply robust DiD estimators.
- ✅ Pass: Estimate 2×2 DiD with interaction term and verify equivalence to group-time means; test parallel trends with pre-treatment period placebo tests; visualize trends with event study; implement staggered DiD using TWFE and compare to Callaway-Sant'Anna or Sun-Abraham estimators to avoid bias from heterogeneous effects; report treatment effects with wild cluster bootstrap standard errors.
- 🛠️ How: DiD: `Y = β₀ + β₁·Treated + β₂·Post + β₃·(Treated×Post)`; parallel trends: plot group-specific trends pre-treatment; placebo DiD on earlier periods; for staggered adoption, never-treated as control group; decompose TWFE weights; wild bootstrap: `clustered_bootstrap` with Rademacher weights.

Week 98 — Synthetic Control Method
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand synthetic control as data-driven matching on pre-treatment outcomes; implement synthetic control optimization; conduct permutation-based inference; assess fit quality; handle multiple treated units.
- ✅ Pass: Apply synthetic control to a policy intervention; construct synthetic control by optimizing weights on donor pool to match pre-treatment outcomes; report weights and predictor balance; visualize treated vs synthetic trends; conduct placebo tests by reassigning treatment to each donor; compute p-values from permutation distribution; assess robustness by excluding donors iteratively; report pre/post-treatment RMSPE ratio.
- 🛠️ How: Synthetic control: minimize `||X₁ - X₀W||` subject to `W ≥ 0`, `∑W = 1`, where X₁ is treated unit pre-treatment outcomes, X₀ is donor matrix; use `scipy.optimize.minimize` with constraints or quadratic programming; permutation inference: apply method to each control unit, rank treatment effect; gap plot showing treated - synthetic over time; leave-one-out for robustness.

Week 99 — Regression Kink Design & Bunching
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand regression kink design (RKD) as derivative discontinuity; implement bunching estimator for detecting behavioral responses; test for slope changes; estimate elasticities.
- ✅ Pass: Apply RKD to a policy with kinked schedule (e.g., tax, subsidy); test for change in slope at kink point using local polynomial regression on subsamples; visualize kink with binned scatter plot; implement bunching estimator by comparing empirical distribution to counterfactual; estimate excess mass and implied elasticity; conduct robustness checks varying excluded region and polynomial order.
- 🛠️ How: RKD: estimate `dY/dX` separately left/right of kink, test equality; local linear separately each side: `Y ~ (X-k) + covariates` for X near k; binned scatter: equal-sized bins, plot means; bunching: integrate empirical density, fit counterfactual excluding region around kink (polynomial fit), excess mass = observed - counterfactual; elasticity from excess mass and tax change.

Week 100 — Regression Sensitivity & Bounds
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Assess robustness using omitted variable bias (OVB) formulas; implement Oster (2019) bounds; conduct sensitivity analysis for unobserved confounding; use Rosenbaum bounds for matching estimators; understand partial identification.
- ✅ Pass: Apply OVB formula to show direction/magnitude of bias from omitted confounder; implement Oster method computing δ (relative importance of unobservables) for null result; produce sensitivity plots showing treatment effect as function of confounder strength; apply Rosenbaum bounds to PSM estimates varying Γ; report identified set and discuss assumption needed for causal claim; compare naïve, conditional, and bounded estimates.
- 🛠️ How: OVB: `β̂ = β + γ·δ` where γ is effect of omitted U on Y, δ is coefficient from X ~ U; Oster δ: `δ = [R²max - R̃²]/[R̃² - R°²] · [β̃ - β*]/[β° - β̃]`; plot treatment effect vs confounding strength; Rosenbaum Γ: recompute p-value under assumption of hidden bias; identified set: report range of treatment effects consistent with assumptions.

Week 101 — Advanced Topics & Review
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Integrate multiple identification strategies; understand machine learning for causal inference (double/debiased ML, causal forests); review all methods; conduct sensitivity analysis across multiple methods.
- ✅ Pass: Apply ≥3 causal methods to the same research question; compare point estimates and confidence intervals; discuss relative credibility of each design; implement double ML for treatment effect estimation in high-dimensional setting; report model-averaged treatment effects and conduct multi-method sensitivity analysis; produce comprehensive writeup documenting identification assumptions, threats to validity, and robustness.
- 🛠️ How: Compare DiD, IV, RDD on same outcome; assess common support, parallel trends, instrument strength; double ML: use cross-fitting with `DoubleMLPLR` or manual implementation (Lasso for Y~X, D~X, residualize); causal forest: `grf` package (R) or `econml.dml.CausalForestDML` (Python); plot distribution of treatment effects; report heterogeneity by subgroups; synthesis table with all estimates.
Week 102 — Phase 16 — End-to-End Mini Project
- 📖 Activities: Phase 16 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 16.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 17 · MLOps & Data Engineering — Weeks 103–106</b></summary>

Week 103 — MLOps Zoomcamp
- 📖 [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- 🧪 Practice: Module-by-module implementation as taught (tracking, packaging, CI, serving, orchestration, monitoring)
- ✅ Pass (weekly): A runnable local pipeline from clean state to served endpoint with tests passing for that week’s scope.
- 🛠️ How: Docker/Compose; MLflow/W&B; `pytest`; minimal infra defined as per module.

Week 104 — Machine Learning Systems
- 📖 [ML Systems](https://mlsysbook.ai)
- 🧪 Practice: Write/extend a system design doc each week focusing only on that week’s concepts (SLA/SLOs; rollout/rollback; monitoring; data contracts; cost/reliability)
- ✅ Pass (weekly): The doc includes concrete metrics, failure scenarios, and operational procedures aligned to the chapter.
- 🛠️ How: ADR template; simple diagrams-as-code optional (e.g., Mermaid).

Week 105 — Data Engineering Zoomcamp
- 📖 [DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
- 🧪 Practice: Module-by-module pipeline work (ingestion, storage, batch/stream, orchestration, analytics eng, testing) as taught in the course
- ✅ Pass (weekly): Re-deployable pipeline from scratch with idempotent runs for that module’s scope.
- 🛠️ How: Terraform/Docker where required, dbt, Airflow/Prefect according to the module.
Week 106 — Phase 17 — End-to-End Mini Project
- 📖 Activities: Phase 17 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 17.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>

<details>
<summary><b>Phase 18 · Flow Matching and Diffusion Models — Weeks 107–112</b></summary>

Week 107 — Flow Matching Fundamentals
- 📖 Activities: [MIT Diffusion Course 2026](https://diffusion.csail.mit.edu/2026/index.html)
- 🧪 Practice: Understand continuous normalizing flows and probability paths.
- ✅ Pass: Implement a basic 2D flow matching objective.
- 🛠️ How: PyTorch implementation of vector fields.

Week 108 — Score-Based Generative Models
- 📖 Activities: [MIT Diffusion Course 2026](https://diffusion.csail.mit.edu/2026/index.html)
- 🧪 Practice: Connections between score matching and diffusion.
- ✅ Pass: Train a simple score network on toy data.
- 🛠️ How: Denoising score matching objective.

Week 109 — Classifier-Free Guidance & Samplers
- 📖 Activities: [MIT Diffusion Course 2026](https://diffusion.csail.mit.edu/2026/index.html)
- 🧪 Practice: SDE/ODE solvers, Euler vs. Heun, CFG implementation.
- ✅ Pass: Apply CFG to condition the toy score network.
- 🛠️ How: Combine conditional and unconditional score estimates.

Week 110 — Latent Diffusion Models (LDM)
- 📖 Activities: [MIT Diffusion Course 2026](https://diffusion.csail.mit.edu/2026/index.html)
- 🧪 Practice: Variational Autoencoders (VAEs), pushing diffusion to latent space.
- ✅ Pass: Hook up a pretrained VAE and train a diffusion model on its latents.
- 🛠️ How: PyTorch `AutoencoderKL` and a U-Net backbone.

Week 111 — Advanced Applications
- 📖 Activities: [MIT Diffusion Course 2026](https://diffusion.csail.mit.edu/2026/index.html)
- 🧪 Practice: Image editing, inpainting, and inversion.
- ✅ Pass: Perform image inpainting using the trained latent diffusion model.
- 🛠️ How: Masked latent updates during sampling steps.
Week 112 — Phase 18 — End-to-End Mini Project
- 📖 Activities: Phase 18 Project Synthesis
- 🧪 Practice: Execute an end-to-end task utilizing only the tools learned up to Phase 18.
- ✅ Pass: Implement a complete pipeline representing this phase's core concepts, documenting findings.
- 🛠️ How: Produce a standalone Jupyter Notebook or script demonstrating the concepts end-to-end.

</details>
