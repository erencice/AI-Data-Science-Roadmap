# AI & Data Science Weekly Plan — Activities, Practice, and Pass Criteria

![Duration](https://img.shields.io/badge/duration-~218_weeks-6f42c1)
![Pace](https://img.shields.io/badge/pace-8–10_h%2Fweek-0e8a16)
![Path](https://img.shields.io/badge/path-beginner%E2%86%92practitioner-0366d6)
![Style](https://img.shields.io/badge/style-cumulative%2C_concept%E2%86%92practice-555)

Zero prior knowledge is assumed. Learning order is strictly top-to-bottom. Each week includes a clear “Pass” requirement aligned to the primary resource.

— Quick jump —
- Phase 1 · Data Analysis Foundations
- Phase 2 · SQL
- Phase 3 · Mathematics for Machine Learning
- Phase 4 · Introduction to Probability
- Phase 5 · Convex Optimization
- Phase 6 · Statistics Fundamentals
- Phase 7 · Mathematical Statistics
- Phase 8 · Applied Multivariate Statistics
- Phase 9 · Bayesian Statistics & Missing Data
- Phase 10 · Statistical Learning with Python (ISLP)
- Phase 11 · Data Mining
- Phase 12 · Classical Machine Learning
- Phase 13 · Elements of Statistical Learning
- Phase 14 · Deep Learning
- Phase 15 · R for Data Science
- Phase 16 · Econometrics, Time Series & Financial Econometrics
- Phase 17 · Causal Inference
- Phase 18 · MLOps & Data Engineering

Legend
- 📖 Activities (primary source)
- 🧪 Practice (small tasks)
- ✅ Pass (weekly pass criterion)
- 🛠️ How (implementation hint)
- 🔁 Flex (catch-up, spaced review)

Duration and pacing
- Duration: ~218 weeks (≈4.2 years), 8–10 h/week
- Weekly output: small practical tasks only
- Frequent Flex Weeks between phases for consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 1 · Data Analysis Foundations — Weeks 1–8 (Complete Python for Data Analysis)</b></summary>

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
</details>

🔁 Flex — Consolidate EDA template and notes

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 2 · SQL — Weeks 9–11 (Complete SQL)</b></summary>
  
Week 9 — SQL Basics, Syntax & Data Manipulation
- 📖 Activities: [SQL Roadmap (GeeksforGeeks)](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- 🧪 **Practice**: Understand the foundation of databases (**SQL Basics**: RDBMS, SQL vs NoSQL). Master **Basic Syntax** including SQL data types and operators. Learn **Data Definition Language (DDL)** commands and syntax to define database structures. Execute **Data Manipulation Language (DML)** commands and SQL Clauses for day-to-day operations (inserting, updating, deleting, and retrieving data), and understand the difference between DML vs DDL and DML vs TCL.
- ✅ **Pass**: Install and configure an RDBMS; create a sample database demonstrating DDL commands; write queries covering basic syntax and operators; perform DML operations (INSERT, UPDATE, DELETE) and basic retrieval (SELECT) using appropriate clauses; document the differences between DDL and DML operations.
- 🛠️ **How**: Start with database theory (Relational vs NoSQL); use `CREATE TABLE`, `ALTER TABLE`, and `DROP` for DDL; use `INSERT INTO`, `UPDATE`, and `DELETE` for DML; practice basic `SELECT` queries with operators (WHERE, AND/OR, LIKE) to filter data.

Week 10 — Aggregate Queries, Constraints, JOINs & Subqueries
- 📖 Activities: [SQL Roadmap (GeeksforGeeks)](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- 🧪 **Practice**: Master **Aggregate Queries** and aggregate functions to perform calculations on grouped data. Apply **Data Constraints** to enforce data rules and maintain integrity. Work with **JOIN Queries** (Inner Join, Outer Join, Cartesian Join, and Self Join) to combine data from multiple tables. Utilize **Subqueries** (Correlated, Nested) for complex filtering, and analyze the use cases of SQL Join vs Subquery.
- ✅ **Pass**: Create a relational schema with constraints enforced (Primary Key, Foreign Key, Not Null); write queries using aggregate functions with GROUP BY and HAVING; write queries demonstrating all major JOIN types (Inner, Outer, Self, Cartesian); create complex queries using nested and correlated subqueries; document how constraints maintain data accuracy.
- 🛠️ **How**: Use `COUNT`, `SUM`, `AVG`, `MIN`, `MAX` for aggregations; enforce rules with constraints; combine tables using `INNER JOIN`, `LEFT/RIGHT OUTER JOIN`, and `CROSS JOIN`; write subqueries in `WHERE`, `FROM`, and `SELECT` clauses; compare the execution and readability of JOINs vs Subqueries.

Week 11 — Advanced Functions, Performance & Advanced SQL Features
- 📖 Activities: [SQL Roadmap (GeeksforGeeks)](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- 🧪 **Practice**: Explore **Advanced Functions** (String, Date and Time, Numeric). Create **Views** to simplify complex queries. Utilize **Indexes** for query optimization and best practices. Understand **Transactions** for data reliability and **Integrity Constraints**. Work with **Stored Procedures** and functions. Master **Performance Optimization** techniques and **Advanced SQL** features like Window Functions, Common Table Expressions (CTEs), Pivot and Unpivot, Dynamic SQL, and SQL Triggers.
- ✅ **Pass**: Build an optimized database schema; write queries using advanced string, numeric, and date functions; create and manage Views; demonstrate performance optimization by creating Indexes and analyzing queries; write a Transaction block ensuring consistency; create a Stored Procedure; implement Window Functions, CTEs, and Triggers for advanced data manipulation.
- 🛠️ **How**: Use specific functions like `CONCAT`, `EXTRACT`, `ROUND`; `CREATE VIEW view_name`; `CREATE INDEX` and apply query optimization best practices; manage transactions with `BEGIN`, `COMMIT`, `ROLLBACK`; write `CREATE PROCEDURE`; apply `OVER (PARTITION BY... ORDER BY...)` for Window Functions; structure queries with `WITH` (CTEs); implement `CREATE TRIGGER` and explore Pivot/Unpivot operations.

</details>

🔁 Flex — ETL mini-project

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 3 · Mathematics for ML — Weeks 12–21 (Complete MML)</b></summary>

Week 12 — Linear Algebra I
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) — Ch. 2, §2.1–2.5
- 🧪 Practice: Systems of linear equations and their geometric interpretation (§2.1); matrix addition, multiplication, and transpose (§2.2); Gaussian elimination and row echelon form (§2.3); vector spaces and subspaces (§2.4); linear independence and span (§2.5).
- ✅ Pass: Solve a 3×3 system Ax=b by hand using Gaussian elimination; confirm the solution via `np.linalg.solve`; determine whether three given vectors in ℝ³ are linearly independent using row reduction and rank check.
- 🛠️ How: `np.linalg.solve`, `np.linalg.matrix_rank`; `sympy.Matrix.rref()` for row echelon form; compare rank to number of columns to test independence.

Week 13 — Linear Algebra II
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) — Ch. 2, §2.6–2.8; Ch. 3
- 🧪 Practice: Basis and rank (§2.6); linear mappings and transformation matrices (§2.7); affine spaces (§2.8); norms and inner products (§3.1–3.2); orthogonality, orthonormal bases, and orthogonal projections (§3.4–3.8).
- ✅ Pass: Find a basis for the column space and null space of a matrix; perform a change of basis; project a vector onto a subspace using the projection formula; verify orthonormality of a Gram-Schmidt result.
- 🛠️ How: `scipy.linalg.null_space`, `np.linalg.matrix_rank`; implement Gram-Schmidt manually; projection formula P = A(AᵀA)⁻¹Aᵀ.

Week 14 — Matrix Decompositions
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) — Ch. 4
- 🧪 Practice: Determinant and trace (§4.1); eigenvalues and eigenvectors (§4.2); Cholesky decomposition (§4.3); eigendecomposition A=PDP⁻¹ and positive definiteness (§4.4); Singular Value Decomposition (§4.5); low-rank matrix approximation (§4.6).
- ✅ Pass: Compute eigendecomposition of a symmetric matrix; verify A=PDP⁻¹ by reconstruction; compute full SVD; reconstruct the matrix from its top-k singular values and plot Frobenius reconstruction error vs k.
- 🛠️ How: `np.linalg.eigh` for symmetric matrices, `np.linalg.eig` for general; `np.linalg.svd`; `np.linalg.cholesky`; Frobenius error `np.linalg.norm(A - Ak, 'fro')`.

Week 15 — Vector Calculus I
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) — Ch. 5, §5.1–5.4
- 🧪 Practice: Differentiation of univariate functions (§5.1); partial derivatives and gradients of scalar-valued functions (§5.2); Jacobians of vector-valued functions (§5.3); gradients of matrices (§5.4).
- ✅ Pass: Compute the gradient of f(x,y)=x²y+sin(y) analytically; verify each partial derivative with central differences; compute the Jacobian of g(x,y)=[x²+y, xy] analytically and compare each entry to the numerical Jacobian.
- 🛠️ How: Central differences `(f(x+h)−f(x−h))/(2h)` per component; `sympy.diff` for symbolic verification; compute each column of the numerical Jacobian by perturbing one input variable at a time and differencing each output component (∂gᵢ/∂xⱼ ≈ (gᵢ(x+heⱼ)−gᵢ(x−heⱼ))/(2h)).

Week 16 — Vector Calculus II
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) — Ch. 5, §5.5–5.8
- 🧪 Practice: Useful gradient identities (§5.5); backpropagation and automatic differentiation (§5.6); higher-order derivatives and Hessians (§5.7); linearization and multivariate Taylor series (§5.8).
- ✅ Pass: Derive gradients of composed functions using chain rule; compute the Hessian of a multivariate function and verify positive/negative definiteness; implement a 2-layer forward and backward pass and verify all gradients against central differences (max abs diff < 1e-4).
- 🛠️ How: Chain rule by hand; numerical Hessian via finite differences; `torch.autograd` or the `autograd` library for automatic differentiation verification.

Week 17 — Probability I
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) — Ch. 6, §6.1–6.4
- 🧪 Practice: Probability spaces, sample spaces, and events (§6.1); discrete and continuous probabilities (§6.2); sum rule, product rule, and Bayes' theorem (§6.3); expectation, variance, covariance, and correlation as summary statistics (§6.4).
- ✅ Pass: Apply Bayes' theorem to a disease-testing scenario; compute joint, marginal, and conditional probabilities from a 2×2 contingency table; verify both the Bayes' theorem result and the contingency table probabilities via simulation; confirm Law of Large Numbers by plotting sample mean convergence.
- 🛠️ How: `np.random`, `scipy.stats`; verify `P(A|B)=P(A∩B)/P(B)`; Bayes: `P(A|B)=P(B|A)P(A)/P(B)`; compare empirical moments to closed-form expressions.

Week 18 — Probability II
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) — Ch. 6, §6.5–6.7
- 🧪 Practice: Univariate and multivariate Gaussian distributions (§6.5); conjugacy and exponential family (§6.6); change of variables and the inverse-CDF method (§6.7).
- ✅ Pass: Sample from a multivariate Gaussian via Cholesky decomposition; recover the empirical covariance matrix and compare it to Σ; visualize 2D Gaussian contours; implement inverse-CDF sampling for an Exponential distribution and verify with a Q-Q plot.
- 🛠️ How: `L = np.linalg.cholesky(Sigma)`, `X = Z @ L.T`; `np.cov`; inverse CDF: `scipy.stats.expon.ppf`; `scipy.stats.probplot` for Q-Q verification.

Week 19 — Optimization I
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) — Ch. 7, §7.1–7.2
- 🧪 Practice: Gradient descent algorithm and step-size selection (§7.1); constrained optimization and Lagrange multipliers (§7.2).
- ✅ Pass: Implement gradient descent for f(x)=½xᵀQx+cᵀx; demonstrate monotone loss decrease; compare convergence (iterations to tolerance) for three step sizes; solve an equality-constrained problem analytically using Lagrange multipliers and verify KKT conditions.
- 🛠️ How: Analytic gradient Qx+c; fixed and backtracking line-search step sizes; plot loss vs iterations; verify stationarity of Lagrangian ∇L=0.

Week 20 — Optimization II
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) — Ch. 7, §7.3; Ch. 9–10
- 🧪 Practice: Convex sets and convex functions, second-order conditions (§7.3); linear regression as maximum likelihood estimation (Ch. 9); dimensionality reduction with PCA (Ch. 10).
- ✅ Pass: Verify convexity of a function using the second-order condition; derive the closed-form normal equation for linear regression and compare to gradient descent; implement PCA via SVD on a 2D dataset and plot variance explained by each principal component.
- 🛠️ How: `scipy.optimize.minimize`; normal equation `w=(XᵀX)⁻¹Xᵀy`; `np.linalg.svd` for PCA; compare to `sklearn.decomposition.PCA`.

Week 21 — Review
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) — All chapters
- 🧪 Practice: Connect all MML topics (Ch. 2 Linear Algebra → Ch. 3 Geometry → Ch. 4 Decompositions → Ch. 5 Calculus → Ch. 6 Probability → Ch. 7 Optimization → Ch. 8–12 ML Applications); identify how each foundational concept appears in a concrete ML algorithm.
- ✅ Pass: A one-page concept map with ≥10 explicit connections between math concepts and ML techniques, each labeled with the relevant MML chapter (e.g., SVD §4.5 ↔ PCA §10.2; multivariate Gaussian §6.5 ↔ GMMs §11.3; Lagrange multipliers §7.2 ↔ SVM §12.2; eigendecomposition §4.4 ↔ dimensionality reduction §10.3).
- 🛠️ How: Use mind-mapping tool or hand-drawn diagram; annotate each connection with the relevant MML section number; write one sentence per link explaining the relationship.
</details>

🔁 Flex — Retrieval practice and summaries

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 4 · Introduction to Probability — Weeks 22–25 (Complete Grinstead & Snell)</b></summary>

Week 22 — Probability Ch. 1–2 (Discrete Probability, Continuous Probability)
- 📖 [Probability (PDF)](https://math.dartmouth.edu/~prob/prob/prob.pdf)
- 🧪 Practice: Understand sample spaces, events, and probability axioms; work with discrete and continuous random variables; compute probabilities using counting methods; master conditional probability and independence; apply Bayes' theorem.
- ✅ Pass: Solve ≥10 probability problems covering: sample space construction, probability calculations using combinations/permutations, conditional probability, independence tests, and Bayes' theorem applications; verify solutions analytically and via simulation; demonstrate Law of Total Probability.
- 🛠️ How: Use combinatorics: `math.comb(n,k)`, `math.perm(n,k)`; simulate outcomes with `np.random.choice`; verify `P(A|B) = P(A∩B)/P(B)`; Bayes: `P(A|B) = P(B|A)P(A)/P(B)`; compare analytical vs empirical probabilities.

Week 23 — Probability Ch. 3–4 (Expectation, Distributions)
- 📖 [Probability (PDF)](https://math.dartmouth.edu/~prob/prob/prob.pdf)
- 🧪 Practice: Compute expected values and variance; work with common discrete distributions (Bernoulli, Binomial, Geometric, Poisson); understand continuous distributions (Uniform, Exponential, Normal); apply moment generating functions; explore distribution relationships.
- ✅ Pass: Compute expectations analytically for ≥5 distributions; derive variance from definition; generate samples and verify empirical moments match theoretical values (within 5% for n≥1000); use MGFs to derive moments; demonstrate Central Limit Theorem convergence with visualizations.
- 🛠️ How: `scipy.stats` for distributions; `np.random.binomial`, `np.random.poisson`, `np.random.normal`; empirical mean: `np.mean(samples)`; plot sampling distributions; CLT: plot standardized sample means for increasing n.

Week 24 — Probability Ch. 5–7 (Markov Chains, Random Walks)
- 📖 [Probability (PDF)](https://math.dartmouth.edu/~prob/prob/prob.pdf)
- 🧪 Practice: Understand Markov chain fundamentals: states, transitions, transition matrices; compute stationary distributions; classify states (transient, recurrent, absorbing); work with random walks; understand gambler's ruin problem; explore applications.
- ✅ Pass: Implement discrete-time Markov chain simulator; compute n-step transition probabilities via matrix powers; find stationary distribution by solving πP = π; classify states and compute expected hitting times; simulate random walks and verify theoretical properties (e.g., return probabilities); solve gambler's ruin analytically and verify via simulation.
- 🛠️ How: Transition matrix: `P = np.array([[p11, p12,...], [...]])`; n-step: `np.linalg.matrix_power(P, n)`; stationary: eigenvalue problem with `np.linalg.eig`, find eigenvector for λ=1; simulation: iterate `state = np.random.choice(states, p=P[state])`; random walk: `positions = np.cumsum(steps)`.

Week 25 — Probability Ch. 8–10 (Law of Large Numbers, Limit Theorems)
- 📖 [Probability (PDF)](https://math.dartmouth.edu/~prob/prob/prob.pdf)
- 🧪 Practice: Understand weak and strong law of large numbers; master Central Limit Theorem and its applications; work with generating functions for sums; understand convergence concepts; apply limit theorems to approximation problems.
- ✅ Pass: Demonstrate Law of Large Numbers: plot sample mean convergence to theoretical mean for increasing sample sizes; verify CLT: show standardized sum converges to Normal via QQ plots and hypothesis tests for n=[10,30,100,1000]; use generating functions to compute distribution of sums; apply continuity correction for Normal approximation to Binomial; compute confidence intervals using CLT.
- 🛠️ How: LLN: `running_mean = np.cumsum(samples)/np.arange(1, n+1)`; plot vs theoretical mean; CLT: `(sum(samples) - n*mu)/(sigma*sqrt(n))` should be N(0,1); `scipy.stats.normaltest` for normality; `scipy.stats.probplot` for QQ plots; confidence intervals: `mean ± z*SE`.

</details>

🔁 Flex — Probability consolidation

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 5 · Convex Optimization — Weeks 26–35 (Complete Boyd & Vandenberghe)</b></summary>

Week 26 — Mathematical Foundations & Convex Sets
- 📖 [Convex Optimization (PDF)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- 🧪 Practice: Understand vector spaces, norms, and topology; master convex sets (definition, operations, separating hyperplanes); work with cones (proper, dual); understand convex hulls and Carathéodory's theorem.
- ✅ Pass: Prove convexity of specific sets analytically; verify convexity numerically for given sets; implement separating hyperplane algorithm; compute convex hull of finite point set; visualize 2D/3D convex sets and their intersections; verify that intersection of convex sets is convex through examples.
- 🛠️ How: Check convexity: for x, y in set and θ ∈ [0,1], verify θx + (1-θ)y in set; `scipy.spatial.ConvexHull`; plot with `plt.fill` for 2D, `mpl_toolkits.mplot3d` for 3D; separating hyperplane via linear program or support vector methods.

Week 27 — Convex Functions & Operations
- 📖 [Convex Optimization (PDF)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- 🧪 Practice: Define and verify convex functions; understand epigraphs and sublevel sets; master operations preserving convexity (nonnegative weighted sum, composition, pointwise maximum, perspective); work with conjugate functions and Fenchel duality.
- ✅ Pass: Verify convexity via first-order condition (gradient) and second-order condition (Hessian PSD); compute epigraphs and sublevel sets; prove convexity of composed functions using composition rules; compute conjugate functions for common functions (norms, indicators, quadratics); visualize convex functions and their conjugates; implement perspective operation.
- 🛠️ How: First-order: `f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)` for all x,y; second-order: `∇²f(x) ⪰ 0`; check eigenvalues `np.linalg.eigvals(H) ≥ 0`; conjugate: `f*(y) = sup_x(yᵀx - f(x))`; 3D surface plots for visualization.

Week 28 — Convex Optimization Problems
- 📖 [Convex Optimization (PDF)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- 🧪 Practice: Formulate optimization problems in standard form; understand linear programming (LP), quadratic programming (QP), and second-order cone programming (SOCP); work with geometric programming; understand quasiconvex optimization.
- ✅ Pass: Formulate ≥5 real-world problems as convex programs (portfolio optimization, LP relaxation, robust optimization, etc.); solve using CVX/CVXPY; verify optimality conditions; convert non-convex problems to convex via transformation (log transform for GP); demonstrate equivalence of problem formulations.
- 🛠️ How: `cvxpy` for modeling: `cp.Variable`, `cp.Minimize/Maximize`, `cp.Problem(objective, constraints).solve()`; verify KKT conditions at solution; transformations: log-transform for geometric programs; compare solution time across formulations.

Week 29 — Duality Theory
- 📖 [Convex Optimization (PDF)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- 🧪 Practice: Derive Lagrange dual function and dual problem; understand weak and strong duality; apply Slater's condition for strong duality; work with KKT conditions for optimality; interpret dual variables as sensitivity (shadow prices).
- ✅ Pass: Formulate Lagrangian for ≥3 optimization problems; derive dual problem; verify weak duality (dual objective ≤ primal objective); check Slater's condition and confirm strong duality; solve primal and dual numerically and verify zero duality gap; interpret dual variables and verify sensitivity interpretation via perturbation analysis; verify KKT conditions at optimum.
- 🛠️ How: Lagrangian: `L(x,λ,ν) = f(x) + Σλᵢgᵢ(x) + Σνⱼhⱼ(x)`; dual function: `g(λ,ν) = inf_x L(x,λ,ν)`; solve primal/dual with `cvxpy`; access dual variables: `constraint.dual_value`; perturbation: resolve with modified constraint bounds, compare optimal values to dual variables.

Week 30 — Unconstrained Optimization Algorithms
- 📖 [Convex Optimization (PDF)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- 🧪 Practice: Implement gradient descent with exact and backtracking line search; implement Newton's method and analyze convergence rates; understand quasi-Newton methods (BFGS); compare first-order vs second-order methods.
- ✅ Pass: Implement gradient descent with backtracking line search from scratch; implement Newton's method with Hessian modification for non-convexity; compare convergence rates empirically (linear for GD, quadratic for Newton); implement BFGS and compare to exact Newton; plot objective value, gradient norm, and step size vs iterations; verify theoretical convergence rates on quadratic problems.
- 🛠️ How: GD with backtracking: start with step size t, while `f(x - t∇f) > f(x) - αt||∇f||²` do `t = βt` (α=0.3, β=0.8); Newton: `x := x - [∇²f(x)]⁻¹∇f(x)`; Hessian modification: add λI if not PD; BFGS: update inverse Hessian approximation; `scipy.optimize.minimize(method='BFGS')` for comparison.

Week 31 — Equality Constrained Optimization
- 📖 [Convex Optimization (PDF)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- 🧪 Practice: Solve equality-constrained problems via elimination and KKT system; implement Newton's method for equality-constrained problems; understand feasible and infeasible start methods.
- ✅ Pass: Solve equality-constrained QP by forming and solving KKT system directly; implement Newton step for equality constraints (compute search direction solving KKT system); compare elimination method (reduce dimensions) vs Lagrange multiplier method; implement feasible start Newton (project onto feasible set) and infeasible start Newton (minimize feasibility and optimality); verify that solution satisfies primal and dual feasibility.
- 🛠️ How: KKT system: `[H Aᵀ; A 0][Δx; Δν] = [-∇f; -h]` where Ax=b are equality constraints; solve with `np.linalg.solve`; elimination: express x = Fz + x₀ where Fx₀=b, AF=0, then minimize in z; feasibility measure: `||Ax-b||²`; verify solution: check `Ax=b` and `∇f + Aᵀν = 0`.

Week 32 — Interior-Point Methods
- 📖 [Convex Optimization (PDF)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- 🧪 Practice: Understand barrier methods and central path; implement log-barrier method for inequality constraints; understand primal-dual interior-point methods; analyze complexity and convergence.
- ✅ Pass: Implement log-barrier method for LP or QP with inequality constraints; track central path by solving sequence of problems for decreasing t; implement primal-dual interior-point method computing Newton steps in primal-dual space; compare to barrier method; plot duality gap vs iterations; verify polynomial-time complexity empirically; compare to simplex method for LP.
- 🛠️ How: Barrier function: `φ(x) = -Σ log(-fᵢ(x))`; minimize `t·f₀(x) + φ(x)` for increasing t; primal-dual: solve KKT system with perturbed complementarity `λᵢfᵢ(x) = -1/t`; Newton step: `[H+∇²φ Aᵀ; A 0][Δx;Δν] = [-t∇f-∇φ; -Ax+b]`; track `η = m/t` (duality gap upper bound).

Week 33 — Applications to Machine Learning
- 📖 [Convex Optimization (PDF)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- 🧪 Practice: Formulate ML problems as convex optimization: SVM (hinge loss, soft margin), logistic regression, Lasso and elastic net, matrix completion, robust PCA; understand regularization from optimization perspective.
- ✅ Pass: Formulate and solve SVM dual problem; implement coordinate descent for Lasso; formulate logistic regression as convex problem and solve with Newton's method; implement matrix completion via nuclear norm minimization; solve robust PCA (low-rank + sparse decomposition); compare custom implementations to sklearn baselines; visualize regularization paths and decision boundaries.
- 🛠️ How: SVM dual: `max Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)` subject to `0 ≤ α ≤ C`, `Σαᵢyᵢ=0`; Lasso coordinate descent: update one coefficient at a time with soft thresholding; nuclear norm: `||X||* = Σσᵢ`; robust PCA: `min ||L||* + λ||S||₁` subject to `L+S=M`; use `cvxpy` for verification.

Week 34 — Advanced Topics: Distributed & Stochastic Methods
- 📖 [Convex Optimization (PDF)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- 🧪 Practice: Understand decomposition methods (dual decomposition, ADMM); implement stochastic gradient methods; work with proximal operators and proximal gradient method; understand operator splitting methods.
- ✅ Pass: Implement ADMM for a separable problem (e.g., Lasso, consensus optimization); implement stochastic gradient descent with diminishing and constant step sizes; compare convergence to batch GD; implement proximal gradient method for composite objectives (smooth + nonsmooth); derive and implement proximal operators for common functions (L1 norm, indicator functions); demonstrate ADMM convergence to consensus.
- 🛠️ How: ADMM: iterate `x := argmin L_ρ(x,z,u)`, `z := argmin L_ρ(x,z,u)`, `u := u + ρ(Ax+Bz-c)` where `L_ρ = f(x)+g(z)+uᵀ(Ax+Bz-c)+ρ/2||Ax+Bz-c||²`; SGD: sample minibatch, update with gradient estimate; proximal operator: `prox_f(x) = argmin_u (f(u) + ½||u-x||²)`; proximal gradient: `x := prox_{tg}(x - t∇f(x))`.

Week 35 — Integration & Review
- 📖 [Convex Optimization (PDF)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- 🧪 Practice: Integrate all topics; formulate and solve complex real-world problems; understand when to use each algorithm; compare optimization formulations.
- ✅ Pass: Complete a comprehensive project applying convex optimization to a real problem; formulate in ≥2 different ways (primal/dual, different variables); solve with ≥3 algorithms comparing convergence and computation time; verify optimality via KKT conditions and duality gap; produce detailed report documenting problem formulation, algorithm selection rationale, convergence analysis, and sensitivity analysis; include visualizations of feasible set, level sets, and optimization trajectory.
- 🛠️ How: Select problem from application domain (portfolio, control, signal processing, ML); compare custom implementations to industrial solvers (CVXPY, Gurobi, MOSEK); profiling with `cProfile` or `line_profiler`; convergence plots (objective, constraint violation, KKT residual); sensitivity: perturb problem data and track optimal value.
</details>

🔁 Flex — Convex optimization consolidation

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 6 · Statistics Fundamentals — Weeks 36–41 (Complete Think Stats)</b></summary>

Week 36 — Think Stats Ch. 1
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Explore a dataset; compute summary statistics; build histograms and PMFs; construct ECDFs.
- ✅ Pass: Implement ECDF from scratch on real data; verify it is non-decreasing and ends at 1.0; overlay histogram and ECDF to compare distributional insights; interpret outliers.
- 🛠️ How: `np.sort`; `np.arange(1,n+1)/n`; `plt.step` for ECDF; `plt.hist` for histogram.

Week 37 — Think Stats Ch. 2
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Compute central tendency (mean, median, mode) and spread (variance, std, range, IQR); explore effect of outliers on these measures.
- ✅ Pass: Compare mean/SD vs median/MAD/IQR on 2 datasets (one symmetric, one skewed); explain when each measure is appropriate; show outlier impact graphically.
- 🛠️ How: `np.mean`, `np.median`, `np.std`; `scipy.stats.median_abs_deviation`; `np.percentile` for IQR.

Week 38 — Think Stats Ch. 3–4
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Work with CDFs and PMFs; model data with probability distributions; compare empirical vs theoretical distributions.
- ✅ Pass: Fit data to common distributions (Normal, Exponential); use CDF plots to assess fit; compute percentiles and quantiles; explain when to use PMF vs CDF.
- 🛠️ How: `scipy.stats.norm.fit`, `scipy.stats.expon`; `probplot` for QQ plots; CDF comparison plots.

Week 39 — Think Stats Ch. 5–6
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Model data with analytical distributions; explore relationships between variables; compute conditional probabilities.
- ✅ Pass: Fit a parametric model to real data; compute and interpret correlation and covariance; demonstrate conditional probability with a contingency table.
- 🛠️ How: `scipy.stats` distribution fitting; `np.corrcoef`; `pd.crosstab` for contingency tables.

Week 40 — Think Stats Ch. 7–8
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Estimate parameters with confidence intervals; perform hypothesis tests; understand p-values and statistical significance.
- ✅ Pass: Compute confidence intervals via bootstrap and analytical methods; run a hypothesis test; simulate to show Type I error ≈ α; produce a power curve for detecting effect sizes.
- 🛠️ How: Bootstrap resampling; `scipy.stats.ttest_ind`; simulation to count rejections under H₀ and H₁.

Week 41 — Think Stats Ch. 9–10 (+wrap)
- 📖 [Think Stats](https://allendowney.github.io/ThinkStats/)
- 🧪 Practice: Explore linear relationships; fit simple and multiple regression; interpret coefficients; check regression assumptions.
- ✅ Pass: Fit OLS regression; interpret R², coefficients, and p-values; produce diagnostic plots (residuals vs fitted, QQ plot); compute VIFs and flag multicollinearity.
- 🛠️ How: `statsmodels.api.OLS`; `statsmodels.stats.outliers_influence.variance_inflation_factor`; diagnostic plots.
</details>

🔁 Flex — Stats recap

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 7 · Mathematical Statistics — Weeks 42–51 (Complete Freund's Mathematical Statistics)</b></summary>

Week 42 — Freund's Ch. 1–2: Introduction & Probability Distributions
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Review probability foundations for statistical inference; master probability distributions (discrete and continuous); understand distribution functions, expectation, variance, and moment-generating functions; work with common distributions (binomial, Poisson, normal, exponential, gamma).
- ✅ Pass: Solve ≥15 problems from Chapters 1-2 covering: probability axioms, conditional probability, distribution functions (PMF, PDF, CDF), expectation and variance calculations, moment-generating functions, and distribution properties; verify ≥5 solutions analytically and compare with simulation; compute MGFs and use them to derive moments for ≥3 distributions.
- 🛠️ How: Use `scipy.stats` for distribution verification; simulate samples to verify theoretical moments; plot empirical vs theoretical distributions; derive MGFs by hand and verify moment formulas; `np.mean`, `np.var` for empirical moments.

Week 43 — Freund's Ch. 3–4: Multivariate Distributions & Sampling Distributions
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Master joint, marginal, and conditional distributions; compute covariance and correlation; understand independence; derive sampling distributions of sample mean and sample variance; master chi-square, t, and F distributions; understand Central Limit Theorem applications.
- ✅ Pass: Solve ≥20 problems from Chapters 3-4 covering: joint and marginal distributions, conditional distributions, covariance and correlation computations, independence tests, sampling distribution derivations; derive sampling distribution of sample mean for normal populations; prove and verify that (n-1)S²/σ² follows chi-square distribution; demonstrate CLT convergence with simulations showing standardized sample means converge to N(0,1) for n ≥ 30.
- 🛠️ How: Compute joint probabilities via integration/summation; `scipy.stats.chi2`, `scipy.stats.t`, `scipy.stats.f` for theoretical distributions; simulate samples and compute sample statistics; plot sampling distributions; verify theoretical results via simulation with n=1000 iterations.

Week 44 — Freund's Ch. 5: Point Estimation
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Understand properties of point estimators (unbiasedness, consistency, efficiency, sufficiency); master method of moments and maximum likelihood estimation; compute Fisher information and Cramér-Rao lower bound; understand sufficient and complete statistics; apply Rao-Blackwell theorem.
- ✅ Pass: Solve ≥15 problems from Chapter 5 covering: method of moments estimation for ≥3 distributions, maximum likelihood estimation with analytic solutions, Fisher information computation, verification of unbiasedness via expectation, efficiency comparisons using CRLB, sufficiency verification using factorization theorem; derive MLE for ≥5 parametric families; compute Fisher information and verify CRLB for variance of unbiased estimators; implement Rao-Blackwell improvement for an estimator.
- 🛠️ How: Solve likelihood equations analytically; compute `E[θ̂]` to check unbiasedness; Fisher information: `I(θ) = -E[∂²log L/∂θ²]`; CRLB: `Var(θ̂) ≥ 1/I(θ)`; numerical verification via simulation; implement factorization to find sufficient statistics.

Week 45 — Freund's Ch. 6: Interval Estimation
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Construct confidence intervals for means (known and unknown variance), proportions, variances, and differences; understand confidence coefficient interpretation; apply large-sample approximations; construct confidence intervals for parameters of common distributions; understand relationship between confidence intervals and hypothesis tests.
- ✅ Pass: Solve ≥15 problems from Chapter 6 constructing: confidence intervals for population mean (σ known and unknown), confidence intervals for proportions using normal approximation, confidence intervals for variance using chi-square distribution, confidence intervals for difference of means (independent and paired samples), two-sided and one-sided intervals; verify coverage probability via simulation (generate 1000 samples, construct CIs, verify ~95% contain true parameter); demonstrate duality between CIs and hypothesis tests.
- 🛠️ How: Normal CI: `x̄ ± z*σ/√n` (σ known); t-based: `x̄ ± t*s/√n` (σ unknown); proportion: `p̂ ± z*√(p̂(1-p̂)/n)`; variance: `[(n-1)s²/χ²_upper, (n-1)s²/χ²_lower]`; simulate samples with `np.random.normal`, compute CIs, check coverage; `scipy.stats.t.ppf`, `scipy.stats.chi2.ppf` for critical values.

Week 46 — Freund's Ch. 7: Hypothesis Testing
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Master hypothesis testing framework (null and alternative hypotheses, Type I and II errors, power); perform z-tests and t-tests for means; chi-square tests for variance; understand p-values and significance levels; compute power functions; understand Neyman-Pearson lemma and likelihood ratio tests.
- ✅ Pass: Solve ≥20 problems from Chapter 7 covering: one-sample and two-sample tests for means, one-sided and two-sided tests, tests for proportions, tests for variance; compute Type I error probability (α) and Type II error probability (β); derive and plot power functions showing power vs true parameter value; apply Neyman-Pearson lemma to find most powerful tests; conduct likelihood ratio tests; verify Type I error rate via simulation (generate data under H₀, compute rejection rate ≈ α).
- 🛠️ How: Test statistic: `z = (x̄-μ₀)/(σ/√n)` or `t = (x̄-μ₀)/(s/√n)`; p-value: `P(|Z|>|z_obs|)` for two-sided; power: `P(reject H₀ | H₁ true)`; simulate under H₀ and H₁ to compute α and power; plot power curve; `scipy.stats` for test statistics and p-values; implement LRT: `λ = L(θ₀)/L(θ̂_MLE)`.

Week 47 — Freund's Ch. 8: Inferences About Means
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Conduct inferences about single means (t-test), difference of means (independent samples), paired samples; understand pooled vs unpooled variance; check assumptions (normality, equal variance); apply Welch's t-test for unequal variances; understand one-way ANOVA for comparing multiple means.
- ✅ Pass: Solve ≥15 problems from Chapter 8 including: one-sample t-tests with confidence intervals, two-sample t-tests (pooled and unpooled), paired t-tests, one-way ANOVA with ≥3 groups; verify assumptions using normality tests (Shapiro-Wilk) and equal variance tests (Levene's, Bartlett's); conduct post-hoc comparisons after ANOVA; compute effect sizes (Cohen's d); reproduce ANOVA F-test result manually from sum of squares decomposition.
- 🛠️ How: `scipy.stats.ttest_1samp`, `ttest_ind`, `ttest_rel`; pooled variance: `s²_p = ((n₁-1)s₁² + (n₂-1)s₂²)/(n₁+n₂-2)`; Welch's t-test for unequal variances; `scipy.stats.f_oneway` for ANOVA; manual ANOVA: compute SSB (between), SSW (within), MSB, MSW, F = MSB/MSW; `scipy.stats.shapiro`, `scipy.stats.levene`; post-hoc: Tukey HSD or Bonferroni correction.

Week 48 — Freund's Ch. 9–10: Inferences About Proportions & Variances
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Conduct inferences about single proportions and differences; perform chi-square tests for variance; test equality of two variances (F-test); apply goodness-of-fit tests; understand contingency table analysis and tests of independence; compute relative risk and odds ratios.
- ✅ Pass: Solve ≥15 problems from Chapters 9-10 covering: confidence intervals and hypothesis tests for single proportions, two-proportion z-tests, chi-square goodness-of-fit tests, chi-square tests of independence with contingency tables, F-tests for equality of variances; verify normal approximation validity (np ≥ 5, n(1-p) ≥ 5); conduct Fisher's exact test when cell counts are small; compute and interpret odds ratios and relative risk; verify chi-square test Type I error via simulation.
- 🛠️ How: Proportion test: `z = (p̂-p₀)/√(p₀(1-p₀)/n)`; two-proportion: pooled estimate; chi-square GOF: `χ² = Σ(O-E)²/E`; independence: `χ² = Σ(O_ij - E_ij)²/E_ij` where `E_ij = (row_i × col_j)/n`; F-test: `F = s₁²/s₂²`; `scipy.stats.chi2_contingency`, `fisher_exact`; odds ratio: `OR = (a×d)/(b×c)` for 2×2 table.

Week 49 — Freund's Ch. 11: Regression & Correlation
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Understand simple linear regression model assumptions; derive least squares estimators; conduct inference on slope and intercept; test for significance of regression; understand correlation coefficient and its inference; distinguish correlation from causation; compute residuals and check assumptions.
- ✅ Pass: Solve ≥15 problems from Chapter 11 including: fitting simple linear regression using least squares, deriving normal equations, computing confidence intervals for slope and intercept, testing H₀: β₁=0, constructing confidence and prediction intervals for Y, computing correlation coefficient and testing its significance; verify regression assumptions via residual plots (residuals vs fitted, QQ plot, scale-location); derive least squares estimators from first principles; demonstrate Gauss-Markov theorem via simulation; compute R² and interpret as proportion of variance explained.
- 🛠️ How: Least squares: `β̂₁ = Σ(xᵢ-x̄)(yᵢ-ȳ)/Σ(xᵢ-x̄)²`, `β̂₀ = ȳ - β̂₁x̄`; `statsmodels.api.OLS` for inference; confidence interval for β₁: `β̂₁ ± t*SE(β̂₁)`; prediction interval wider than confidence interval; correlation test: `t = r√(n-2)/√(1-r²)`; residual diagnostics with `plt.scatter`; verify homoscedasticity and normality.

Week 50 — Freund's Ch. 12: Analysis of Variance (ANOVA)
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Master one-way and two-way ANOVA; understand completely randomized designs and randomized block designs; perform multiple comparisons; understand ANOVA assumptions and diagnostics; compute effect sizes; understand relationship between ANOVA and regression.
- ✅ Pass: Solve ≥15 problems from Chapter 12 including: one-way ANOVA with ≥3 groups, two-way ANOVA with interaction, randomized complete block design, multiple comparison procedures (Tukey HSD, Bonferroni), effect size calculations (η², ω²); manually compute ANOVA table (SS, df, MS, F) and verify with software; check assumptions (normality, homogeneity of variance, independence); demonstrate equivalence of one-way ANOVA F-test to two-sample t-test when k=2 groups; interpret interaction plots for two-way ANOVA.
- 🛠️ How: One-way ANOVA: `F = MSB/MSW` where `MSB = SSB/(k-1)`, `MSW = SSW/(n-k)`; two-way: include main effects and interaction; `scipy.stats.f_oneway`; `statsmodels.stats.anova.anova_lm` for two-way; `statsmodels.stats.multicomp.pairwise_tukeyhsd`; effect size: `η² = SSB/SST`; check assumptions same as Week 47; visualize interactions with `sns.pointplot`.

Week 51 — Freund's Ch. 13–14: Review & Nonparametric Methods
- 📖 Activities: [John E. Freund's Mathematical Statistics with Applications](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- 🧪 Practice: Review all statistical inference concepts; master distribution-free nonparametric methods: sign test, Wilcoxon signed-rank test, Mann-Whitney U test, Kruskal-Wallis test, runs test; understand when to use parametric vs nonparametric tests; compare power of parametric vs nonparametric tests.
- ✅ Pass: Solve ≥15 problems covering: sign test for median, Wilcoxon signed-rank test for paired data, Mann-Whitney U test for two independent samples, Kruskal-Wallis test for multiple groups, Spearman rank correlation, runs test for randomness; compare results of parametric vs nonparametric tests on same data; conduct power analysis via simulation showing power loss of nonparametric tests under normality and power gain under non-normality; verify test assumptions and justify method selection; integrate all inference concepts from Chapters 1-14 in a comprehensive analysis.
- 🛠️ How: `scipy.stats.wilcoxon`, `mannwhitneyu`, `kruskal`, `spearmanr`; sign test: compare median to hypothesized value using binomial; runs test for independence; simulate data from normal and heavy-tailed distributions, apply both parametric and nonparametric tests, compare power; decision tree for test selection based on assumptions.
</details>

🔁 Flex — Mathematical Statistics consolidation

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 8 · Applied Multivariate Statistics — Weeks 52–66 (Complete PSU STAT 505)</b></summary>

Week 52 — Measures of Central Tendency, Dispersion and Association (Lesson 1)
- 📖 Activities: [PSU STAT 505 Lesson 1](https://online.stat.psu.edu/stat505/lesson/1)
- 🧪 Practice: Compute multivariate measures of central tendency (mean vectors); understand dispersion (covariance matrices, generalized variance); compute association measures (correlation matrices); interpret relationships between variables.
- ✅ Pass: Calculate mean vectors and covariance matrices for multivariate data; compute correlation matrices and interpret relationships; calculate generalized variance and total variation; compare variance-covariance structures across groups. Complete PSU STAT 505 Lesson 1.
- 🛠️ How: `np.mean(axis=0)` for mean vectors; `np.cov` for covariance matrices; `np.corrcoef` for correlation; `np.linalg.det` for generalized variance; visualize with heatmaps.

Week 53 — Linear Combinations of Random Variables (Lesson 2)
- 📖 Activities: [PSU STAT 505 Lesson 2](https://online.stat.psu.edu/stat505/lesson/2)
- 🧪 Practice: Understand properties of linear combinations of random vectors; compute means and covariances of linear combinations; work with linear transformations; understand independence and correlation.
- ✅ Pass: Compute mean and covariance of linear combinations; verify properties of linear transformations; demonstrate how linear combinations preserve or change correlation structure; apply to dimensionality reduction scenarios. Complete PSU STAT 505 Lesson 2.
- 🛠️ How: Matrix operations for linear combinations `Y = AX + b`; compute `E[Y] = AE[X] + b` and `Cov(Y) = A Cov(X) A^T`; verify independence conditions; visualize transformations.

Week 54 — Graphical Display of Multivariate Data (Lesson 3)
- 📖 Activities: [PSU STAT 505 Lesson 3](https://online.stat.psu.edu/stat505/lesson/3)
- 🧪 Practice: Create various multivariate visualizations; produce scatter plot matrices, star plots, profile plots; understand 3D plots and contour plots; interpret patterns and relationships visually.
- ✅ Pass: Create comprehensive visualization suite: scatter plot matrix with correlations, 3D scatter plots, profile plots for repeated measures, star plots for multivariate observations, contour plots for bivariate distributions; identify patterns, outliers, and relationships. Complete PSU STAT 505 Lesson 3.
- 🛠️ How: `pd.plotting.scatter_matrix`; `mpl_toolkits.mplot3d` for 3D plots; `plt.plot` for profile plots; `seaborn.pairplot`; custom star/radar plots with `plt.subplot(projection='polar')`.

Week 55 — Multivariate Normal Distribution (Lesson 4)
- 📖 Activities: [PSU STAT 505 Lesson 4](https://online.stat.psu.edu/stat505/lesson/4)
- 🧪 Practice: Understand multivariate normal distribution properties; compute Mahalanobis distance; generate samples from MVN; test for multivariate normality; understand conditional and marginal distributions.
- ✅ Pass: Generate samples from multivariate normal; compute and interpret Mahalanobis distance vs Euclidean distance; perform Mardia's test for multivariate normality; compute marginal and conditional distributions; visualize MVN with contour plots. Complete PSU STAT 505 Lesson 4.
- 🛠️ How: `scipy.stats.multivariate_normal`; `scipy.spatial.distance.mahalanobis`; Mardia's skewness and kurtosis tests; compute conditionals using partitioned covariance matrices.

Week 56 — Sample Mean Vector and Sample Correlation (Lesson 5)
- 📖 Activities: [PSU STAT 505 Lesson 5](https://online.stat.psu.edu/stat505/lesson/5)
- 🧪 Practice: Estimate mean vectors and covariance matrices from samples; understand sampling distributions; perform inference on mean vectors; test hypotheses about correlations; construct confidence regions.
- ✅ Pass: Estimate mean vectors and covariance matrices; derive sampling distributions; construct confidence ellipses for mean vectors; test hypotheses about population means; compute standard errors for correlations. Complete PSU STAT 505 Lesson 5.
- 🛠️ How: Sample statistics with `np.mean`, `np.cov`; Wishart distribution for covariance; confidence ellipses using eigenvalues/eigenvectors; bootstrap for inference.

Week 57 — Multivariate Conditional Distribution and Partial Correlation (Lesson 6)
- 📖 Activities: [PSU STAT 505 Lesson 6](https://online.stat.psu.edu/stat505/lesson/6)
- 🧪 Practice: Compute conditional distributions from joint multivariate normal; calculate partial correlations; understand the difference between marginal and partial correlation; interpret conditional independence.
- ✅ Pass: Partition covariance matrices to compute conditional distributions; calculate and interpret partial correlations; compare partial vs marginal correlations; test for conditional independence; visualize relationships controlling for other variables. Complete PSU STAT 505 Lesson 6.
- 🛠️ How: Use partitioned covariance matrices `Σ = [[Σ11, Σ12], [Σ21, Σ22]]`; conditional mean `μ1 + Σ12 Σ22^-1 (x2 - μ2)`; conditional covariance `Σ11 - Σ12 Σ22^-1 Σ21`; `pingouin.partial_corr` for partial correlations.

Week 58 — Inferences Regarding Multivariate Population Mean (Lesson 7)
- 📖 Activities: [PSU STAT 505 Lesson 7](https://online.stat.psu.edu/stat505/lesson/7)
- 🧪 Practice: Perform Hotelling's T² tests for one-sample and two-sample problems; construct simultaneous confidence intervals; understand multivariate hypothesis testing; compare with univariate t-tests.
- ✅ Pass: Conduct one-sample Hotelling's T² test; perform two-sample Hotelling's T² test; construct simultaneous confidence intervals using Bonferroni correction; compare multivariate vs univariate approaches; interpret test statistics, p-values, and effect sizes. Complete PSU STAT 505 Lesson 7.
- 🛠️ How: Implement `T² = n(x̄ - μ0)^T S^-1 (x̄ - μ0)`; convert to F-statistic: `F = (n-p)T²/((n-1)p)`; `scipy.stats.f` for p-values; Bonferroni intervals: `t_(α/2p, n-1)`.

Week 59 — Multivariate Analysis of Variance (MANOVA) (Lesson 8)
- 📖 Activities: [PSU STAT 505 Lesson 8](https://online.stat.psu.edu/stat505/lesson/8)
- 🧪 Practice: Perform one-way and two-way MANOVA; understand Wilks' Lambda, Pillai's trace, and other test statistics; conduct post-hoc tests; check MANOVA assumptions; compare to univariate ANOVA.
- ✅ Pass: Run MANOVA with ≥2 dependent variables and ≥3 groups; report test statistics (Wilks' Lambda, Pillai's trace, Hotelling-Lawley trace, Roy's largest root); perform follow-up univariate ANOVAs and discriminant analysis; check assumptions (Box's M test, multivariate normality). Complete PSU STAT 505 Lesson 8.
- 🛠️ How: `statsmodels.multivariate.manova.MANOVA`; interpret output; visualize group centroids; check assumptions before interpretation; compare effect sizes across responses.

Week 60 — Repeated Measures Analysis (Lesson 9)
- 📖 Activities: [PSU STAT 505 Lesson 9](https://online.stat.psu.edu/stat505/lesson/9)
- 🧪 Practice: Analyze repeated measures data using multivariate approach; understand sphericity and compound symmetry; perform profile analysis; test for parallelism, coincidence, and flatness; handle within-subject factors.
- ✅ Pass: Analyze repeated measures design with multivariate approach; test sphericity assumption (Mauchly's test); perform profile analysis testing parallelism, levels, and flatness hypotheses; compare multivariate vs univariate repeated measures ANOVA; interpret within-subject and between-subject effects. Complete PSU STAT 505 Lesson 9.
- 🛠️ How: `statsmodels` for repeated measures; test sphericity; profile plots with error bars; Greenhouse-Geisser correction when sphericity violated; contrast matrices for specific comparisons.

Week 61 — Discriminant Analysis (Lesson 10)
- 📖 Activities: [PSU STAT 505 Lesson 10](https://online.stat.psu.edu/stat505/lesson/10)
- 🧪 Practice: Perform linear and quadratic discriminant analysis; understand Fisher's linear discriminant; classify observations; evaluate classification performance; understand relationship to MANOVA; compare LDA/QDA assumptions.
- ✅ Pass: Apply LDA and QDA to classification problem; compute discriminant functions and classify held-out observations; report confusion matrix and misclassification rates; visualize decision boundaries; compare LDA/QDA to logistic regression; verify equal covariance assumption. Complete PSU STAT 505 Lesson 10.
- 🛠️ How: `sklearn.discriminant_analysis.LinearDiscriminantAnalysis/QuadraticDiscriminantAnalysis`; `classification_report`; ROC curves; cross-validation for error estimation; Box's M test for covariance equality.

Week 62 — Principal Components Analysis (Lesson 11)
- 📖 Activities: [PSU STAT 505 Lesson 11](https://online.stat.psu.edu/stat505/lesson/11)
- 🧪 Practice: Perform PCA on correlation and covariance matrices; understand eigenvalues/eigenvectors interpretation; determine number of components; compute component scores; interpret loadings; create biplots; understand variance explained.
- ✅ Pass: Apply PCA to dataset with ≥6 variables; create scree plot; select components using Kaiser criterion (eigenvalue > 1) and cumulative variance (80%); interpret loadings for first 2-3 PCs; create biplot; reconstruct data; compare PCA on correlation vs covariance. Complete PSU STAT 505 Lesson 11.
- 🛠️ How: `sklearn.decomposition.PCA`; standardize with `StandardScaler`; `explained_variance_ratio_`; scree plot; biplot with `plt.arrow`; verify reconstruction error.

Week 63 — Factor Analysis (Lesson 12)
- 📖 Activities: [PSU STAT 505 Lesson 12](https://online.stat.psu.edu/stat505/lesson/12)
- 🧪 Practice: Perform exploratory factor analysis; understand factor model and common vs specific variance; estimate communalities and uniqueness; perform factor rotations (varimax, promax); determine number of factors; interpret factor loadings.
- ✅ Pass: Conduct factor analysis; determine number of factors using parallel analysis and scree plot; extract factors using maximum likelihood or principal axis factoring; perform varimax and promax rotations; interpret and name factors; report communalities and variance explained; compare to PCA. Complete PSU STAT 505 Lesson 12.
- 🛠️ How: `sklearn.decomposition.FactorAnalysis`; `factor_analyzer` package for rotations; parallel analysis comparing eigenvalues to random data; factor loading interpretation with cutoff |loading| > 0.3.

Week 64 — Canonical Correlation Analysis (Lesson 13)
- 📖 Activities: [PSU STAT 505 Lesson 13](https://online.stat.psu.edu/stat505/lesson/13)
- 🧪 Practice: Perform canonical correlation analysis between two sets of variables; compute canonical correlations and canonical variates; test significance; interpret canonical loadings and cross-loadings; assess redundancy.
- ✅ Pass: Apply CCA to dataset with two variable sets (≥3 variables each); compute all canonical correlations and test significance; interpret first 2-3 canonical variate pairs; compute canonical loadings (structure correlations); perform redundancy analysis; visualize canonical variates. Complete PSU STAT 505 Lesson 13.
- 🛠️ How: `sklearn.cross_decomposition.CCA`; Wilks' Lambda test: `Λ = ∏(1 - r²)`; canonical loadings as correlations between original variables and canonical variates; redundancy index.

Week 65 — Cluster Analysis (Lesson 14)
- 📖 Activities: [PSU STAT 505 Lesson 14](https://online.stat.psu.edu/stat505/lesson/14)
- 🧪 Practice: Apply hierarchical clustering with different linkage methods; perform k-means clustering; understand distance measures and similarity metrics; determine optimal number of clusters; validate clustering solutions; compare clustering methods.
- ✅ Pass: Perform hierarchical clustering with ≥3 linkage methods (single, complete, average, Ward); create dendrograms; apply k-means with multiple k values; determine optimal k using elbow method, silhouette analysis, and gap statistic; validate with silhouette scores; visualize clusters using PCA; compare hierarchical vs partitioning methods. Complete PSU STAT 505 Lesson 14.
- 🛠️ How: `scipy.cluster.hierarchy` for hierarchical clustering; `sklearn.cluster.KMeans`; distance metrics: Euclidean, Manhattan, Mahalanobis; silhouette analysis; dendrogram interpretation; standardize data before clustering.

Week 66 — Integration and Review
- 📖 Activities: Review all PSU STAT 505 lessons
- 🧪 Practice: Integrate multivariate methods in comprehensive analysis; understand when to use each technique; compare and contrast methods; apply multiple techniques to same dataset.
- ✅ Pass: Complete end-to-end multivariate analysis applying ≥5 techniques from course; write comprehensive report connecting methods; explain method selection rationale; interpret results in context; discuss assumptions and limitations; compare insights from different methods.
- 🛠️ How: Choose appropriate methods for research question; check assumptions; compare complementary analyses (e.g., PCA then cluster analysis; MANOVA then discriminant analysis); synthesize findings.
</details>

🔁 Flex — Multivariate stats consolidation

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 9 · Bayesian Statistics & Missing Data — Weeks 67–78 (Complete Think Bayes, FIMD)</b></summary>

Weeks 67–74 — Think Bayes (Ch. 1–14, paced)
- 📖 [Think Bayes](https://allendowney.github.io/ThinkBayes2/)
- 🧪 Practice: Apply Bayes' theorem to update beliefs; implement conjugate prior models (Beta-Binomial, Gamma-Poisson, Normal-Normal); perform posterior predictive checks; compare models.
- ✅ Pass (weekly): Implement a Bayesian model aligned with the chapter's topic; show prior sensitivity analysis (vary prior parameters and observe posterior changes); generate posterior predictive samples and compare to observed data using a suitable test statistic.
- 🛠️ How: Use analytical posteriors when available; for PPC, draw samples from posterior, then from likelihood, and compare summary stats to data.

Weeks 75–78 — Flexible Imputation of Missing Data (complete)
- 📖 [FIMD](https://stefvanbuuren.name/fimd/)
- 🧪 Practice: Missingness mechanisms; MICE; sensitivity (as in book)
- ✅ Pass (weekly): Run MICE (m≥5) on a dataset; report pooled estimates per Rubin’s rules; compare to complete-case; perform delta-adjustment sensitivity where relevant.
- 🛠️ How: use a MICE implementation (e.g., statsmodels/impyute/sklearn-iterative as proxy) consistent with book procedures.
</details>

🔁 Flex — Consolidate Bayesian + MI

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 10 · Statistical Learning with Python (ISLP) — Weeks 79–88 (Complete ISLP)</b></summary>

Week 79 — ISLP Ch. 1–2 (Intro + Statistical Learning)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand the statistical learning framework; implement train/test splits; explore the bias-variance trade-off with KNN at different k values.
- ✅ Pass: On a dataset, demonstrate how training error decreases with model complexity while test error shows U-shape; implement 5-fold CV and compare to hold-out estimate; discuss flexibility vs interpretability.
- 🛠️ How: `train_test_split`; `KFold`/`cross_val_score`; vary KNN's k parameter; plot training vs test error curves.

Week 80 — ISLP Ch. 3 (Linear Regression)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Fit simple and multiple linear regression; interpret coefficients; add interaction and polynomial terms; assess model fit with residual diagnostics.
- ✅ Pass: Fit OLS with and without interaction/polynomial terms; compare R² vs adjusted R²; produce residual plots; select optimal polynomial degree via CV; interpret coefficient confidence intervals.
- 🛠️ How: `LinearRegression`; `PolynomialFeatures`; `cross_val_score`; `statsmodels` for CIs; residual diagnostics.

Week 81 — ISLP Ch. 4 (Classification)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Implement logistic regression; understand LDA/QDA assumptions; apply KNN for classification; explore classification metrics beyond accuracy.
- ✅ Pass: Compare logistic regression, LDA, QDA, and KNN using stratified 5-fold CV; report confusion matrix, precision, recall, and ROC-AUC; select optimal classification threshold based on problem context.
- 🛠️ How: `LogisticRegression`; `LinearDiscriminantAnalysis`; `QuadraticDiscriminantAnalysis`; `KNeighborsClassifier`; `roc_curve` for threshold selection.

Week 82 — ISLP Ch. 5 (Resampling Methods)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Compare validation approaches: hold-out, LOOCV, k-fold CV; use bootstrap for uncertainty estimation; understand variance-bias trade-off in resampling.
- ✅ Pass: Compare test error estimates from LOOCV vs 5-fold vs 10-fold CV; implement bootstrap to estimate coefficient standard errors; compare bootstrap SEs to analytic SEs.
- 🛠️ How: `LeaveOneOut`; `KFold`; implement bootstrap loop with `np.random.choice`; fix seeds for reproducibility.

Week 83 — ISLP Ch. 6 (Model Selection & Regularization)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand the motivation for regularization; implement ridge and lasso regression; interpret coefficient shrinkage and sparsity; tune regularization parameter via CV.
- ✅ Pass: Plot ridge and lasso coefficient paths as λ varies; select optimal λ via CV; compare test error of OLS vs ridge vs lasso; explain when lasso produces sparse solutions.
- 🛠️ How: `Ridge`; `Lasso`; `RidgeCV`; `LassoCV`; `StandardScaler` (scale features first); `lasso_path` for path plots.

Week 84 — ISLP Ch. 7 (Beyond Linearity)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Move beyond linearity with polynomial regression, step functions, and splines; understand degrees of freedom; fit GAM-style models.
- ✅ Pass: Fit polynomial, step function, and spline models; compare their flexibility and test errors; produce partial dependence plots; select appropriate number of knots/degrees via CV.
- 🛠️ How: `PolynomialFeatures`; `SplineTransformer`; `pd.cut` for step functions; compare MSE on held-out data.

Week 85 — ISLP Ch. 8 (Tree-Based Methods)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Fit decision trees; understand bagging and the random forest algorithm; implement gradient boosting; interpret tree-based models.
- ✅ Pass: Fit and prune a decision tree; compare single tree vs random forest vs gradient boosting on test error; show OOB error for RF; plot feature importances and partial dependence plots.
- 🛠️ How: `DecisionTreeClassifier/Regressor`; `RandomForestClassifier/Regressor`; `GradientBoostingClassifier/Regressor`; `permutation_importance`; `plot_partial_dependence`.

Week 86 — ISLP Ch. 9 (Support Vector Machines)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand maximal margin classifiers and support vectors; fit SVMs with linear and non-linear kernels; tune hyperparameters (C, gamma).
- ✅ Pass: Fit SVM with linear and RBF kernels; tune C and gamma via grid search with CV; visualize decision boundaries on 2D data; identify and highlight support vectors; compare to logistic regression.
- 🛠️ How: `SVC`; `GridSearchCV`; `plt.contourf` for decision boundaries; access `support_vectors_` attribute.

Week 87 — ISLP Ch. 10 (Unsupervised Learning)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Perform dimensionality reduction with PCA; apply k-means and hierarchical clustering; understand the importance of scaling; evaluate clustering quality.
- ✅ Pass: Apply PCA and plot cumulative explained variance; choose number of components; cluster with k-means (elbow method for k) and hierarchical clustering (dendrogram); evaluate with silhouette score and compare cluster stability across random seeds.
- 🛠️ How: `StandardScaler` (always scale first); `PCA`; `KMeans` with inertia plots; `AgglomerativeClustering`; `dendrogram`; `silhouette_score`.

Week 88 — ISLP Labs/Wrap-up
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Complete an end-to-end ML project using techniques from all ISLP chapters: EDA, preprocessing, model selection, hyperparameter tuning, evaluation, and interpretation.
- ✅ Pass: Deliver a reproducible notebook with proper train/test split, cross-validation, model comparison, hyperparameter tuning, error analysis, and a 1-page summary documenting decisions, limitations, and risks.
- 🛠️ How: `Pipeline`; `ColumnTransformer` for mixed feature types; `GridSearchCV`/`RandomizedSearchCV`; fixed `random_state` throughout; clean documentation.
</details>

🔁 Flex — Validation basics consolidation

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 11 · Data Mining — Weeks 89–97 (Complete DM 3e)</b></summary>

Weeks 89–97 — Data Mining 3e (Ch. 1–12)
- 📖 [Data Mining 3e (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- 🧪 Practice: Per-chapter algorithmic work strictly matching the chapter (e.g., data preprocessing tasks; Apriori/FP-Growth; decision trees; k-means/DBSCAN; outlier detection)
- ✅ Pass (weekly): Implement a minimal working version for the chapter’s focal algorithm OR replicate results using a library; verify correctness on a deterministic toy and compare performance on a small real dataset.
- 🛠️ How: construct small synthetic datasets with known ground truth (fixed seeds); assert counts/clusters/rules match expectation.
</details>

🔁 Flex — Mining recap

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 12 · Classical Machine Learning — Weeks 98–116 (Complete PRML, Interpretable ML)</b></summary>

Weeks 98–111 — PRML (Ch. 1–13 + review)
- 📖 [PRML (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- 🧪 Practice: Implement core algorithms from each chapter from scratch: probability distributions, linear models, neural networks, kernel methods, graphical models, mixture models, EM algorithm, approximate inference, and sampling methods.
- ✅ Pass (weekly): Implement the chapter's focal algorithm from scratch; verify correctness by comparing to sklearn/scipy baseline (within 2-5% accuracy); document mathematical derivations; use fixed seeds for reproducibility.
- 🛠️ How: Use NumPy for implementations; sklearn only as verification oracle; work on toy datasets; keep detailed notes linking code to book equations.

Weeks 112–116 — Interpretable ML (complete)
- 📖 [Interpretable ML](https://christophm.github.io/interpretable-ml-book/)
- 🧪 Practice: Apply model-agnostic interpretation methods: PDP, ICE, permutation importance, LIME, SHAP; understand intrinsically interpretable models; explore feature interaction methods.
- ✅ Pass (weekly): For a trained model, produce PDP/ICE plots for top features; compute permutation importance; generate SHAP values for individual predictions; write a 1-page analysis comparing methods' stability across 3 bootstrap resamples.
- 🛠️ How: `sklearn.inspection.PartialDependenceDisplay`; `permutation_importance`; `shap.Explainer`; compare explanations across train/test sets.
</details>

🔁 Flex — Validation & interpretation synthesis

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 13 · Elements of Statistical Learning — Weeks 117–126 (Complete ESL)</b></summary>

Week 117 — ESL Ch. 1–3: Introduction & Linear Methods
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Review statistical learning framework; master linear regression theory (bias-variance decomposition, Gauss-Markov theorem); implement subset selection, ridge, and lasso from scratch; understand effective degrees of freedom.
- ✅ Pass: Derive bias-variance decomposition analytically; prove Gauss-Markov theorem; implement best subset selection via exhaustive search for p ≤ 10; implement ridge and lasso with coordinate descent; compute effective degrees of freedom `df(λ) = tr[X(XᵀX + λI)⁻¹Xᵀ]` and verify empirically; compare subset selection, ridge, and lasso on test error and coefficient paths.
- 🛠️ How: Bias-variance: `E[(Y-f̂)²] = Bias²(f̂) + Var(f̂) + σ²`; Gauss-Markov: show OLS has minimum variance among linear unbiased estimators; subset selection: iterate over all 2^p subsets; ridge: `β̂ = (XᵀX + λI)⁻¹Xᵀy`; lasso coordinate descent: soft thresholding `S(z,γ) = sign(z)(|z|-γ)₊`; effective df from hat matrix trace.

Week 118 — ESL Ch. 4–5: Linear Classification & Basis Expansions
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Master linear discriminant analysis (LDA), quadratic discriminant analysis (QDA), logistic regression, and separating hyperplanes; understand basis expansions (polynomial, splines, wavelets); implement natural cubic splines.
- ✅ Pass: Derive LDA decision boundary assuming equal covariance; implement QDA allowing separate covariances; fit logistic regression via Newton-Raphson (IRLS); implement linear separating hyperplane via perceptron algorithm; construct natural cubic spline basis manually and fit regression; compare polynomial vs spline fits showing boundary bias and variance; derive and verify degrees of freedom for smoothing splines.
- 🛠️ How: LDA: estimate class means μₖ and pooled covariance Σ, classify via `argmax_k log P(G=k) - ½(x-μₖ)ᵀΣ⁻¹(x-μₖ)`; QDA: separate Σₖ for each class; logistic IRLS: iterate `β := β + (XᵀWX)⁻¹Xᵀ(y-p)` where W=diag(p(1-p)); perceptron: `β := β + ηyᵢxᵢ` for misclassified points; natural spline: impose constraints for linearity beyond boundary knots.

Week 119 — ESL Ch. 6–7: Kernel Methods & Model Assessment
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Understand kernel smoothing and local regression; implement k-nearest neighbors, Nadaraya-Watson estimator, local polynomial regression; master cross-validation theory (GCV, leave-one-out shortcuts); understand bootstrap for model selection; derive and implement optimism estimators (Cp, AIC, BIC).
- ✅ Pass: Implement Nadaraya-Watson kernel regression with Gaussian kernel and bandwidth selection via CV; implement local linear regression (LOESS) and show boundary bias correction compared to Nadaraya-Watson; derive and implement leave-one-out CV shortcut for linear smoothers via hat matrix; implement 0.632 bootstrap estimator; compute Cp, AIC, BIC for nested models and verify consistency of BIC; compare all model selection criteria on a common dataset.
- 🛠️ How: Nadaraya-Watson: `f̂(x₀) = Σ K((xᵢ-x₀)/h)yᵢ / Σ K((xᵢ-x₀)/h)`; LOESS: weighted least squares in local neighborhood; LOO shortcut: `CV = (1/n)Σ(yᵢ-f̂(xᵢ))²/(1-hᵢᵢ)²` where hᵢᵢ is diagonal of hat matrix; Cp: `RSS/σ² + 2d`; AIC: `-2log-likelihood + 2d`; BIC: `-2log-likelihood + log(n)d`.

Week 120 — ESL Ch. 8–9: Model Inference & Additive Models
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Understand bootstrap for inference (standard errors, confidence intervals, percentile and BCa methods); implement permutation tests; master generalized additive models (GAMs) with backfitting algorithm; understand tree-based models and CART algorithm.
- ✅ Pass: Implement bootstrap confidence intervals (normal, percentile, BCa) and compare coverage on simulations; implement permutation test for independence and verify Type I error rate; implement GAM backfitting algorithm from scratch for additive model with spline components; fit and prune CART tree using cost-complexity pruning; compare tree to GAM on same dataset; prove backfitting convergence for additive models.
- 🛠️ How: BCa: bias-correction `z₀` and acceleration `a` from jackknife; percentile: 2.5% and 97.5% quantiles of bootstrap distribution; permutation: shuffle one variable, recompute test statistic; backfitting: iterate `f̂ⱼ := S_j[Y - Σₖ≠ⱼf̂ₖ]` where Sⱼ is smoother; CART: recursive binary splits minimizing RSS or Gini; cost-complexity: `min_T Σ(yᵢ-ŷₜ)² + α|T|`.

Week 121 — ESL Ch. 10: Boosting & Additive Models
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Understand AdaBoost algorithm and its connection to exponential loss; implement gradient boosting from scratch; understand forward stagewise additive modeling; derive L2Boosting and show connection to gradient descent in function space; implement shrinkage and early stopping for regularization.
- ✅ Pass: Implement AdaBoost with decision stumps from scratch; show connection to exponential loss by deriving weight updates; implement gradient boosting with squared loss and deviance loss; demonstrate that gradient boosting is steepest descent in function space; compare learning rates and early stopping for regularization; implement stochastic gradient boosting (subsampling); produce learning curves showing train/validation error vs boosting iterations.
- 🛠️ How: AdaBoost: iterate `err_m = Σw_i I(y_i≠G_m(x_i))/Σw_i`, `α_m = log((1-err_m)/err_m)`, `w_i := w_i exp(α_m I(y_i≠G_m))`, final: `G = sign(Σα_m G_m)`; gradient boosting: `f_m = f_{m-1} + ν·h_m` where `h_m` fits residuals `-∂L/∂f`; derive for squared loss: residuals are `y-f`; for deviance: residuals are gradients of log-likelihood.

Week 122 — ESL Ch. 11–12: Neural Networks & Support Vector Machines
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Implement feedforward neural network with backpropagation from scratch; understand universal approximation; derive and implement weight decay and early stopping; implement SVM via quadratic programming; understand kernel trick and mercer kernels; compare SVM to logistic regression and neural networks.
- ✅ Pass: Implement multi-layer perceptron with one hidden layer from scratch including backpropagation; verify gradient computation with finite differences; train on classification and regression tasks with weight decay; implement SVM dual problem and solve with quadratic programming; implement kernel SVM with RBF kernel; visualize decision boundaries; compare SVM, logistic regression, and neural network on nonlinearly separable data; demonstrate kernel trick equivalence.
- 🛠️ How: Backprop: forward pass compute activations, backward pass compute gradients via chain rule; weight update: `w := w - η∂L/∂w`; SVM dual: `max Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼxᵢᵀxⱼ` subject to `0≤α≤C`, `Σαᵢyᵢ=0`; kernel trick: replace `xᵢᵀxⱼ` with `K(xᵢ,xⱼ)`; use `cvxopt.solvers.qp` or `scipy.optimize.minimize` for QP.

Week 123 — ESL Ch. 13–14: Prototype Methods & Unsupervised Learning
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Implement k-means, k-medoids, and Gaussian mixture models (GMM) via EM; understand learning vector quantization (LVQ); implement hierarchical clustering with different linkages; understand self-organizing maps (SOM); derive EM algorithm for GMM from first principles.
- ✅ Pass: Implement k-means from scratch and prove convergence (monotonic decrease of objective); implement k-medoids (PAM algorithm); derive EM algorithm for GMM (E-step: compute responsibilities, M-step: update parameters); implement GMM-EM and compare to k-means; implement hierarchical clustering with single, complete, and average linkage; compute cophenetic correlation; implement LVQ and compare to k-means; visualize dendrograms and cluster quality metrics (silhouette, Davies-Bouldin).
- 🛠️ How: k-means: iterate assign-to-nearest-centroid, update-centroids; objective: `Σᵢ Σₖ rᵢₖ||xᵢ-μₖ||²`; EM for GMM: `γᵢₖ = πₖ N(xᵢ|μₖ,Σₖ) / Σⱼ πⱼ N(xᵢ|μⱼ,Σⱼ)`, update `πₖ = Σγᵢₖ/n`, `μₖ = Σγᵢₖxᵢ/Σγᵢₖ`, `Σₖ = Σγᵢₖ(xᵢ-μₖ)(xᵢ-μₖ)ᵀ/Σγᵢₖ`; hierarchical: `scipy.cluster.hierarchy`; cophenetic: correlation between pairwise distances and dendrogram heights.

Week 124 — ESL Ch. 15: Random Forests
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Understand bagging and its variance reduction; implement random forests from scratch with bootstrap sampling and feature subsampling; compute out-of-bag (OOB) error as unbiased test error estimate; understand variable importance measures (permutation, Gini); analyze effect of correlation between trees.
- ✅ Pass: Implement random forest from scratch (bootstrap samples, random feature subset at each split, majority vote/averaging); compute OOB error and compare to test error and cross-validation; implement variable importance via permutation (OOB samples) and Gini decrease; demonstrate variance reduction compared to single tree via bias-variance decomposition; analyze effect of number of features sampled (mtry) on correlation between trees and forest performance; produce partial dependence plots.
- 🛠️ How: Random forest: build B trees each on bootstrap sample with feature subsampling (√p for classification, p/3 for regression); OOB error: for each observation, average predictions from trees not containing it in bootstrap sample; permutation importance: shuffle feature j in OOB data, compute increase in OOB error; Gini importance: sum Gini decrease when splitting on feature across all trees; correlation: compute pairwise correlation of tree predictions.

Week 125 — ESL Ch. 16–17: Ensemble Learning & Graphical Models
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Master ensemble methods theory; understand stacking and super learner; implement Bayesian model averaging; explore undirected graphical models (Markov networks); understand conditional independence and the Hammersley-Clifford theorem; implement graphical lasso for sparse inverse covariance estimation.
- ✅ Pass: Implement stacked generalization (train meta-learner on out-of-fold predictions); implement super learner with cross-validation-based weighting; compute Bayesian model averaging weights using BIC approximation; implement graphical lasso (L1-penalized precision matrix estimation) and visualize resulting network; test conditional independence using partial correlations; compare ensemble methods (bagging, boosting, stacking) on same dataset with ≥5 base learners; produce detailed analysis of why and when each ensemble method excels.
- 🛠️ How: Stacking: train base learners, collect out-of-fold predictions, train meta-learner on these; super learner: non-negative weights minimizing CV error `min Σ(yᵢ - Σαₖf̂ₖ⁽⁻ⁱ⁾)²` subject to `α≥0`, `Σα=1`; BMA weights: `w_k ∝ exp(-BIC_k/2)`; graphical lasso: `max log det Θ - tr(SΘ) - λ||Θ||₁`; use `sklearn_glasso` or ADMM implementation; zero entries in Θ imply conditional independence.

Week 126 — ESL Ch. 18 & Integration: High-Dimensional Problems
- 📖 [ESL](https://hastie.su.domains/ElemStatLearn/)
- 🧪 Practice: Understand challenges in high-dimensional settings (p >> n); implement elastic net combining L1 and L2 penalties; understand the Lasso path and LARS algorithm; implement fused lasso for spatial/temporal smoothing; master multiple testing correction (FDR, FWER); understand compressed sensing and restricted isometry property.
- ✅ Pass: Implement elastic net and demonstrate scenarios where it outperforms pure lasso or ridge; implement LARS algorithm and verify equivalence to lasso path; implement fused lasso for 1D signal denoising; apply multiple testing corrections (Bonferroni, Holm, Benjamini-Hochberg) and compare false discovery rates via simulation; demonstrate compressed sensing recovery with RIP-satisfying matrices; integrate ≥5 ESL techniques in comprehensive analysis comparing interpretability, prediction accuracy, computational cost, and theoretical guarantees.
- 🛠️ How: Elastic net: `min ||y-Xβ||² + λ₁||β||₁ + λ₂||β||²`; LARS: forward stagewise that adds most correlated predictor and moves in equiangular direction; fused lasso: `min ||y-β||² + λ₁||β||₁ + λ₂Σ|βᵢ-βᵢ₊₁|`; FDR: Benjamini-Hochberg procedure sorting p-values; compressed sensing: recover sparse signal from few measurements when sensing matrix satisfies RIP; compare convergence and solution paths.
</details>

🔁 Flex — Statistical learning theory consolidation

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 14 · Deep Learning — Weeks 127–146 (Complete D2L fundamentals, Goodfellow DL)</b></summary>

Weeks 127–134 — D2L (Fundamentals)
- 📖 [D2L](https://d2l.ai)
- 🧪 Practice: Topic-specific small models exactly as covered (MLP, CNN, RNN; optimization; regularization; data pipelines)
- ✅ Pass (weekly): Train the chapter’s model variant on a toy dataset with fixed seeds and one controlled ablation (optimizer OR regularization) taught in D2L; log curves/metrics.
- 🛠️ How: Follow D2L’s PyTorch/MXNet examples; fix seeds; keep experiments minimal and reproducible.

Week 135 — The Illustrated Transformer (Bridge)
- 📖 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- 🧪 Practice: Understand the Transformer architecture: self-attention mechanism, multi-head attention, positional encoding, encoder-decoder structure.
- ✅ Pass: Implement self-attention from scratch; verify tensor shapes at each step; implement attention masking; write unit tests for: (1) output shape correctness, (2) masked positions get zero attention, (3) attention weights sum to 1.
- 🛠️ How: Use NumPy or PyTorch; implement Q, K, V projections; scaled dot-product attention; verify with `assert` statements and test cases.

Weeks 136–146 — Deep Learning Book (Complete)
- 📖 [Deep Learning Book](https://www.deeplearningbook.org/)
- 🧪 Practice: For each chapter, run a small experiment that demonstrates the chapter’s key concept using building blocks learned in D2L
- ✅ Pass (weekly): Provide a controlled comparison or demonstration plot showing the expected qualitative effect (e.g., different inits, L2 vs dropout, step-size schedules).
- 🛠️ How: Small synthetic or standard toy datasets; fixed seeds; log and compare curves cleanly.
</details>

🔁 Flex — DL recap + tracked mini project

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 15 · R for Data Science — Weeks 147–156 (Complete R4DS 2e)</b></summary>

Weeks 147–156 — R4DS (Complete)
- 📖 [R for Data Science (2e)](https://r4ds.hadley.nz)
- 🧪 Practice: Learn R and tidyverse progressively: data import, tidying (pivot_longer/wider), transformation (dplyr verbs), visualization (ggplot2), strings, factors, dates, functions, iteration, and communication (Quarto/RMarkdown).
- ✅ Pass (weekly): Complete a mini-analysis using only functions from chapters covered that week; produce a Quarto/RMarkdown report that renders end-to-end; include at least one visualization and one summary table.
- 🛠️ How: `library(tidyverse)`; `read_csv`; `dplyr` verbs (`filter`, `mutate`, `summarize`, `group_by`); `ggplot2`; `set.seed()` for reproducibility.
</details>

🔁 Flex — R consolidation

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 16 · Econometrics, Time Series & Financial Econometrics — Weeks 157–184 (Complete Gujarati, Lütkepohl, Financial Econometrics)</b></summary>

Weeks 157–168 — Basic Econometrics (complete)
- 📖 [Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf)
- 🧪 Practice: Reproduce a worked example per chapter using methods from that chapter only (OLS basics; classical assumption diagnostics; heteroskedasticity/autocorrelation remedies; functional form; limited dependent variables as presented)
- ✅ Pass (weekly): Match the textbook example’s coefficients and standard errors (within rounding) and include one robustness check discussed in that chapter (e.g., robust/HAC SEs when appropriate).
- 🛠️ How: `statsmodels` OLS/GLM, `cov_type="HC3"` or HAC if the chapter addresses it; include diagnostic plots taught there.

Weeks 169–178 — Lütkepohl (complete)
- 📖 [Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)
- 🧪 Practice: Implement multivariate time series analysis: VAR model specification, estimation, lag order selection, stability analysis, impulse response functions, forecast error variance decomposition, and cointegration/VECM.
- ✅ Pass (weekly): Fit VAR/VECM to macroeconomic data; select lag order using information criteria; verify stability (roots inside unit circle); compute and plot IRFs with confidence bands; perform Johansen cointegration test when applicable.
- 🛠️ How: `statsmodels.tsa.api.VAR`; `statsmodels.tsa.vector_ar.vecm.VECM`; `irf()` for impulse responses; rolling-window forecasts for evaluation.

Weeks 179–184 — Financial Econometrics (complete)
- 📖 [Financial Econometrics (PDF)](https://bashtage.github.io/kevinsheppard.com/files/teaching/mfe/notes/financial-econometrics-2020-2021.pdf)
- 🧪 Practice: Master financial econometrics progressively: volatility modeling (ARCH/GARCH family), multivariate GARCH models, realized volatility and high-frequency data analysis, factor models for asset pricing, portfolio optimization, option pricing and risk management.
- ✅ Pass (weekly): Reproduce examples from the text using methods from each section; implement ARCH/GARCH models and forecast volatility; estimate multivariate GARCH (CCC, DCC, BEKK) and compute risk measures (VaR, ES); analyze high-frequency data and compute realized volatility; apply factor models (CAPM, Fama-French) and optimize portfolios; implement Black-Scholes pricing and calibrate volatility surfaces; verify model specifications using information criteria and diagnostic tests.
- 🛠️ How: `arch` package for GARCH models: `arch_model(returns, vol='GARCH', p=1, q=1).fit()`; multivariate models: `arch.multivariate`; realized volatility from intraday returns; factor regressions: `statsmodels.api.OLS`; portfolio optimization: `scipy.optimize.minimize` with constraints; VaR: `np.percentile(returns, alpha)`; Black-Scholes implementation; diagnostics: Ljung-Box test on residuals and squared residuals; model selection via AIC/BIC.
</details>

🔁 Flex — Econometrics/time-series consolidation

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 17 · Causal Inference — Weeks 185–194 (Complete The Mixtape)</b></summary>

Week 185 — Properties of Regression, DAGs, Potential Outcomes
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand Simpson's paradox and collider bias; draw and analyze directed acyclic graphs (DAGs); master potential outcomes framework; understand Average Treatment Effect (ATE) and selection bias.
- ✅ Pass: Implement Simpson's paradox example showing reversal of association; construct ≥3 DAGs identifying confounders, mediators, and colliders; derive ATE under different selection mechanisms; demonstrate selection bias analytically and via simulation.
- 🛠️ How: Use `networkx` or `dagitty` for DAG visualization; simulate counterfactuals with fixed treatment assignments; compute `E[Y¹] - E[Y⁰]` vs observed difference-in-means; show bias = `E[Y⁰|D=1] - E[Y⁰|D=0]`.

Week 186 — Randomized Controlled Trials & Matching
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand randomization inference; implement exact matching, propensity score matching (PSM), and coarsened exact matching; check covariate balance; assess common support.
- ✅ Pass: Analyze an RCT dataset computing ATE with randomization inference (permutation test); implement PSM with ≥3 matching algorithms (nearest neighbor, caliper, kernel); produce balance tables and Love plots before/after matching; check common support with density plots; report treatment effects with bootstrapped standard errors.
- 🛠️ How: Permutation test: shuffle treatment vector 1000+ times, recompute difference-in-means; `sklearn.neighbors.NearestNeighbors` for matching; logistic regression for propensity scores; standardized mean differences for balance; `seaborn.kdeplot` for common support.

Week 187 — Regression Discontinuity Design (RDD)
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand sharp and fuzzy RDD; check continuity assumptions; select bandwidth using cross-validation and optimal methods; test for manipulation of running variable; implement local polynomial regression.
- ✅ Pass: Apply RDD to real or simulated data with a known cutoff; test for discontinuity at the threshold using local linear regression with ≥3 bandwidths; perform McCrary density test for manipulation; produce RDD plots showing outcome vs running variable with fitted lines; report local average treatment effect (LATE) with robust standard errors; conduct placebo tests at false cutoffs.
- 🛠️ How: Local linear regression within bandwidth h: `Y ~ D + (X-c) + D*(X-c)` for |X-c| < h; optimal bandwidth via `rdrobust` (R) or manual cross-validation; McCrary test: fit separate densities left/right of cutoff and test for jump; bootstrap for inference.

Week 188 — Instrumental Variables (IV)
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand endogeneity and IV identification; implement two-stage least squares (2SLS); test instrument relevance and exogeneity; understand weak instruments problem; compute local average treatment effect (LATE) with compliance types.
- ✅ Pass: Identify a valid instrument and justify exclusion restriction; implement 2SLS manually (first stage, second stage) and compare to built-in IV estimator; test instrument strength (F-stat > 10 rule of thumb, Cragg-Donald); perform overidentification test when multiple instruments available; compute LATE and interpret in terms of compliers; conduct sensitivity analysis for violation of exclusion restriction.
- 🛠️ How: Manual 2SLS: regress X on Z (first stage), predict X̂, regress Y on X̂ (second stage); `statsmodels.sandbox.regression.gmm.IV2SLS` or `linearmodels.iv.IV2SLS`; first-stage F-stat for relevance; Hansen J-stat for overidentification; bound analysis for exclusion restriction violations.

Week 189 — Panel Data & Fixed Effects
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand within-group variation; implement fixed effects (FE) and first differences (FD); test fixed vs random effects (Hausman test); handle time-varying treatments; understand parallel trends assumption.
- ✅ Pass: Estimate panel data model with entity and time fixed effects; compare pooled OLS, FE, and random effects; perform Hausman test; demean data manually and verify equivalence to FE estimator; produce event study plots for dynamic treatment effects; test parallel trends visually and formally; cluster standard errors at appropriate level.
- 🛠️ How: FE via demeaning: `Y_it - Ȳ_i = (X_it - X̄_i)β + (ε_it - ε̄_i)`; `linearmodels.panel.PanelOLS` with `entity_effects=True`; Hausman test compares FE vs RE; event study: include leads/lags of treatment; plot coefficients with 95% CIs; cluster SEs: `cov_type='clustered'`.

Week 190 — Difference-in-Differences (DiD)
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Implement canonical 2×2 DiD; test parallel trends assumption; handle staggered treatment adoption; understand two-way fixed effects (TWFE) issues with heterogeneous treatment effects; apply robust DiD estimators.
- ✅ Pass: Estimate 2×2 DiD with interaction term and verify equivalence to group-time means; test parallel trends with pre-treatment period placebo tests; visualize trends with event study; implement staggered DiD using TWFE and compare to Callaway-Sant'Anna or Sun-Abraham estimators to avoid bias from heterogeneous effects; report treatment effects with wild cluster bootstrap standard errors.
- 🛠️ How: DiD: `Y = β₀ + β₁·Treated + β₂·Post + β₃·(Treated×Post)`; parallel trends: plot group-specific trends pre-treatment; placebo DiD on earlier periods; for staggered adoption, never-treated as control group; decompose TWFE weights; wild bootstrap: `clustered_bootstrap` with Rademacher weights.

Week 191 — Synthetic Control Method
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand synthetic control as data-driven matching on pre-treatment outcomes; implement synthetic control optimization; conduct permutation-based inference; assess fit quality; handle multiple treated units.
- ✅ Pass: Apply synthetic control to a policy intervention; construct synthetic control by optimizing weights on donor pool to match pre-treatment outcomes; report weights and predictor balance; visualize treated vs synthetic trends; conduct placebo tests by reassigning treatment to each donor; compute p-values from permutation distribution; assess robustness by excluding donors iteratively; report pre/post-treatment RMSPE ratio.
- 🛠️ How: Synthetic control: minimize `||X₁ - X₀W||` subject to `W ≥ 0`, `∑W = 1`, where X₁ is treated unit pre-treatment outcomes, X₀ is donor matrix; use `scipy.optimize.minimize` with constraints or quadratic programming; permutation inference: apply method to each control unit, rank treatment effect; gap plot showing treated - synthetic over time; leave-one-out for robustness.

Week 192 — Regression Kink Design & Bunching
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Understand regression kink design (RKD) as derivative discontinuity; implement bunching estimator for detecting behavioral responses; test for slope changes; estimate elasticities.
- ✅ Pass: Apply RKD to a policy with kinked schedule (e.g., tax, subsidy); test for change in slope at kink point using local polynomial regression on subsamples; visualize kink with binned scatter plot; implement bunching estimator by comparing empirical distribution to counterfactual; estimate excess mass and implied elasticity; conduct robustness checks varying excluded region and polynomial order.
- 🛠️ How: RKD: estimate `dY/dX` separately left/right of kink, test equality; local linear separately each side: `Y ~ (X-k) + covariates` for X near k; binned scatter: equal-sized bins, plot means; bunching: integrate empirical density, fit counterfactual excluding region around kink (polynomial fit), excess mass = observed - counterfactual; elasticity from excess mass and tax change.

Week 193 — Regression Sensitivity & Bounds
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Assess robustness using omitted variable bias (OVB) formulas; implement Oster (2019) bounds; conduct sensitivity analysis for unobserved confounding; use Rosenbaum bounds for matching estimators; understand partial identification.
- ✅ Pass: Apply OVB formula to show direction/magnitude of bias from omitted confounder; implement Oster method computing δ (relative importance of unobservables) for null result; produce sensitivity plots showing treatment effect as function of confounder strength; apply Rosenbaum bounds to PSM estimates varying Γ; report identified set and discuss assumption needed for causal claim; compare naïve, conditional, and bounded estimates.
- 🛠️ How: OVB: `β̂ = β + γ·δ` where γ is effect of omitted U on Y, δ is coefficient from X ~ U; Oster δ: `δ = [R²max - R̃²]/[R̃² - R°²] · [β̃ - β*]/[β° - β̃]`; plot treatment effect vs confounding strength; Rosenbaum Γ: recompute p-value under assumption of hidden bias; identified set: report range of treatment effects consistent with assumptions.

Week 194 — Advanced Topics & Review
- 📖 [The Mixtape](https://mixtape.scunning.com)
- 🧪 Practice: Integrate multiple identification strategies; understand machine learning for causal inference (double/debiased ML, causal forests); review all methods; conduct sensitivity analysis across multiple methods.
- ✅ Pass: Apply ≥3 causal methods to the same research question; compare point estimates and confidence intervals; discuss relative credibility of each design; implement double ML for treatment effect estimation in high-dimensional setting; report model-averaged treatment effects and conduct multi-method sensitivity analysis; produce comprehensive writeup documenting identification assumptions, threats to validity, and robustness.
- 🛠️ How: Compare DiD, IV, RDD on same outcome; assess common support, parallel trends, instrument strength; double ML: use cross-fitting with `DoubleMLPLR` or manual implementation (Lasso for Y~X, D~X, residualize); causal forest: `grf` package (R) or `econml.dml.CausalForestDML` (Python); plot distribution of treatment effects; report heterogeneity by subgroups; synthesis table with all estimates.
</details>

🔁 Flex — Causal inference consolidation

---------------------------------------------------------------------

---------------------------------------------------------------------

<details>
<summary><b>Phase 18 · MLOps & Data Engineering — Weeks 195–218 (Complete Zoomcamps, ML Systems)</b></summary>

Weeks 195–202 — MLOps Zoomcamp
- 📖 [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- 🧪 Practice: Module-by-module implementation as taught (tracking, packaging, CI, serving, orchestration, monitoring)
- ✅ Pass (weekly): A runnable local pipeline from clean state to served endpoint with tests passing for that week’s scope.
- 🛠️ How: Docker/Compose; MLflow/W&B; `pytest`; minimal infra defined as per module.

Weeks 203–210 — Machine Learning Systems
- 📖 [ML Systems](https://mlsysbook.ai)
- 🧪 Practice: Write/extend a system design doc each week focusing only on that week’s concepts (SLA/SLOs; rollout/rollback; monitoring; data contracts; cost/reliability)
- ✅ Pass (weekly): The doc includes concrete metrics, failure scenarios, and operational procedures aligned to the chapter.
- 🛠️ How: ADR template; simple diagrams-as-code optional (e.g., Mermaid).

Weeks 211–218 — Data Engineering Zoomcamp
- 📖 [DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
- 🧪 Practice: Module-by-module pipeline work (ingestion, storage, batch/stream, orchestration, analytics eng, testing) as taught in the course
- ✅ Pass (weekly): Re-deployable pipeline from scratch with idempotent runs for that module’s scope.
- 🛠️ How: Terraform/Docker where required, dbt, Airflow/Prefect according to the module.
</details>

🔁 Flex — Ops/engineering consolidation

---------------------------------------------------------------------


---------------------------------------------------------------------

Resource-to-Week Completion Map (cover-to-cover)
- Python for Data Analysis — Weeks 1–8 — [Python for Data Analysis](https://wesmckinney.com/book/)
- SQL Roadmap (GeeksforGeeks) — Weeks 9–11 — [SQL Roadmap](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- Mathematics for Machine Learning — Weeks 12–21 — [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- Introduction to Probability (Grinstead & Snell) — Weeks 22–25 — [Probability (PDF)](https://math.dartmouth.edu/~prob/prob/prob.pdf)
- Convex Optimization (Boyd & Vandenberghe) — Weeks 26–35 — [Convex Optimization (PDF)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- Think Stats — Weeks 36–41 — [Think Stats](https://allendowney.github.io/ThinkStats/)
- John E. Freund's Mathematical Statistics with Applications — Weeks 42–51 — [Freund's Mathematical Statistics](https://archive.org/details/johnefreundsmath0008mill), [Solutions Manual](https://archive.org/details/instructors-solutions-manual-for-john-e.-freunds-mathematical-statistics-with-ap/)
- PSU STAT 505 (Applied Multivariate Statistics) — Weeks 52–66 — [PSU STAT 505](https://online.stat.psu.edu/stat505)
- Think Bayes — Weeks 67–78 — [Think Bayes](https://allendowney.github.io/ThinkBayes2/)
- Flexible Imputation of Missing Data — Weeks 75–78 — [FIMD](https://stefvanbuuren.name/fimd/)
- ISLP (Statistical Learning with Python) — Weeks 79–88 — [ISLP](https://www.statlearning.com/)
- Data Mining: Concepts and Techniques (3e) — Weeks 89–97 — [Data Mining 3e (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- PRML (Bishop) — Weeks 98–111 — [PRML (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- Interpretable Machine Learning — Weeks 112–116 — [Interpretable ML](https://christophm.github.io/interpretable-ml-book/)
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman) — Weeks 117–126 — [ESL](https://hastie.su.domains/ElemStatLearn/)
- Dive into Deep Learning — Weeks 127–134 — [D2L](https://d2l.ai)
- The Illustrated Transformer — Week 135 — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Deep Learning — Weeks 136–146 — [Deep Learning Book](https://www.deeplearningbook.org/)
- R for Data Science (2e) — Weeks 147–156 — [R for Data Science (2e)](https://r4ds.hadley.nz)
- Basic Econometrics (Gujarati) — Weeks 157–168 — [Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf)
- New Introduction to Multiple Time Series (Lütkepohl) — Weeks 169–178 — [Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)
- Financial Econometrics (Sheppard) — Weeks 179–184 — [Financial Econometrics (PDF)](https://bashtage.github.io/kevinsheppard.com/files/teaching/mfe/notes/financial-econometrics-2020-2021.pdf)
- Causal Inference: The Mixtape (Cunningham) — Weeks 185–194 — [The Mixtape](https://mixtape.scunning.com)
- MLOps Zoomcamp — Weeks 195–202 — [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- Machine Learning Systems — Weeks 203–210 — [ML Systems](https://mlsysbook.ai)
- Data Engineering Zoomcamp — Weeks 211–218 — [DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)

Notes
- Keep work in any format; seed randomness for reproducibility.
- Use Flex Weeks to finish pass items, review tricky parts, and add spaced-repetition cards (optional).

**Important Limitation Regarding Theoretical Econometrics**

This roadmap includes foundational econometric resources (Gujarati for Basic Econometrics, Lütkepohl for Time Series, Cunningham for Causal Inference) that provide essential intuition and practical implementation skills. However, **these resources have limited theoretical rigor and depth compared to advanced theoretical econometrics textbooks** such as:
- Greene's *Econometric Analysis*
- Hayashi's *Econometrics* - Hansen's *Econometrics*
- Hamilton's *Time Series Analysis*
- Wooldridge's graduate texts

- Back to top ↑
