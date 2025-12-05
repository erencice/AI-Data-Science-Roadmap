# AI & Data Science Weekly Plan — Activities, Practice, and Pass Criteria

![Duration](https://img.shields.io/badge/duration-~179_weeks-6f42c1)
![Pace](https://img.shields.io/badge/pace-8–10_h%2Fweek-0e8a16)
![Path](https://img.shields.io/badge/path-beginner%E2%86%92practitioner-0366d6)
![Style](https://img.shields.io/badge/style-cumulative%2C_concept%E2%86%92practice-555)

Zero prior knowledge is assumed. Learning order is strictly top-to-bottom. Each week includes a clear “Pass” requirement aligned to the primary resource.

— Quick jump —
- Phase 1 · Data Analysis Foundations
- Phase 2 · Mathematics for ML
- Phase 3 · Statistics Fundamentals
- Phase 4 · Applied Multivariate Statistics
- Phase 5 · Bayesian Statistics & Missing Data
- Phase 6 · Statistical Learning with Python (ISLP)
- Phase 7 · Classical ML
- Phase 8 · Data Mining
- Phase 9 · Econometrics & Time Series
- Phase 10 · R for Data Science
- Phase 11 · Web Scraping & SQL
- Phase 12 · Deep Learning
- Phase 13 · MLOps & Data Engineering
- Phase 14 · LLMs & Open-Source AI
- Phase 15 · Consolidation & Capstone

Legend
- 📖 Activities (primary source)
- 🧪 Practice (small tasks)
- ✅ Pass (weekly pass criterion)
- 🛠️ How (implementation hint)
- 🔁 Flex (catch-up, spaced review)

Duration and pacing
- Duration: ~179 weeks (≈3.4 years), 8–10 h/week
- Weekly output: small practical tasks only
- Frequent Flex Weeks between phases for consolidation

Main resources (cover-to-cover completion)
- Python for Data Analysis — Wes McKinney — [Python for Data Analysis](https://wesmckinney.com/book/)
- Mathematics for Machine Learning — Deisenroth, Faisal, Ong — [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- Think Stats — Allen B. Downey — [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- Think Bayes — Allen B. Downey — [Think Bayes](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple)
- Flexible Imputation of Missing Data — van Buuren — [FIMD](https://stefvanbuuren.name/fimd/)
- An Introduction to Statistical Learning with Applications in Python — James, Witten, Hastie, Tibshirani — [ISLP](https://www.statlearning.com/)
- Pattern Recognition and Machine Learning — Bishop — [PRML (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- Interpretable Machine Learning — Molnar — [Interpretable ML](https://christophm.github.io/interpretable-ml-book/)
- Data Mining: Concepts and Techniques (3e) — Han, Kamber, Pei — [Data Mining 3e (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- Basic Econometrics — Gujarati — [Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf)
- New Introduction to Multiple Time Series — Lütkepohl — [Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)
- R for Data Science (2e) — Wickham, Çetinkaya-Rundel, Grolemund — [R for Data Science (2e)](https://r4ds.hadley.nz)
- Beautiful Soup docs — [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- Selenium (Python) docs — [Selenium (Python)](https://selenium-python.readthedocs.io/index.html)
- SQL Roadmap — [SQL Roadmap (GeeksforGeeks)](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- Dive into Deep Learning — Zhang et al. — [D2L](https://d2l.ai)
- Deep Learning — Goodfellow, Bengio, Courville — [Deep Learning Book](https://www.deeplearningbook.org/)
- MLOps Zoomcamp — DataTalksClub — [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- Machine Learning Systems — Symeonidis et al. — [ML Systems](https://mlsysbook.ai)
- Data Engineering Zoomcamp — DataTalksClub — [DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
- Hugging Face Course — [HF Course](https://huggingface.co/course/chapter1)
- HF Agents Course — [HF Agents](https://huggingface.co/learn/agents-course/unit0/introduction)

Supporting references (selective)
- Trigonometric Cheat Sheet — [Trig Sheet (PDF)](https://tutorial.math.lamar.edu/pdf/Trig_Cheat_Sheet.pdf)
- Python Crash Course — [Video](https://www.youtube.com/watch?v=rfscVS0vtbw)
- Kevin Sheppard Python Notes — [Notes (PDF)](https://www.kevinsheppard.com/files/teaching/python/notes/python_introduction_2021.pdf)
- PSU STAT — [STAT portal](https://online.stat.psu.edu)
- scikit-learn docs — [scikit-learn](https://scikit-learn.org/stable/index.html)
- statsmodels docs — [statsmodels](https://www.statsmodels.org/stable/index.html)

---------------------------------------------------------------------

<details>
<summary><b>Phase 1 · Data Analysis Foundations — Weeks 1–8 (Complete Python for Data Analysis)</b></summary>

Week 1 — P4DA Ch. 1–2
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Set up Python environment (conda/pip); run IPython/Jupyter; practice Python basics (variables, control flow, functions); understand the data analysis ecosystem.
- ✅ Pass: Create a notebook demonstrating Python fundamentals: define 3 functions, use list/dict comprehensions, write a simple script that reads command-line arguments, and explain the role of NumPy/pandas/matplotlib in the data stack.
- 🛠️ How: Install Anaconda or miniconda; launch Jupyter; experiment with built-in types and control structures; skim the library overview in Ch.1.

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

<details>
<summary><b>Phase 2 · Mathematics for ML — Weeks 9–18 (Complete MML)</b></summary>

Week 9 — Linear Algebra I
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: Vectors (addition, scalar multiplication, norms); matrix operations (transpose, multiplication); linear independence and basis.
- ✅ Pass: Implement vector/matrix operations from scratch; verify linear independence of a set of vectors; compute and interpret different vector norms (L1, L2, Linf).
- 🛠️ How: `np.dot`, `np.linalg.norm`, `np.linalg.matrix_rank`; manually verify independence via row reduction.

Week 10 — Linear Algebra II
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: Eigenvalues and eigenvectors; matrix diagonalization; positive definiteness; condition number.
- ✅ Pass: Compute eigendecomposition of symmetric matrices; verify diagonalization A = PDP⁻¹; check positive definiteness via eigenvalues; interpret condition number for numerical stability.
- 🛠️ How: `np.linalg.eig`, `np.linalg.eigh` for symmetric; `np.linalg.cond`; verify reconstruction.

Week 11 — Decompositions & Geometry
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: SVD and its applications; orthogonal projections; analytic geometry (distances, angles, hyperplanes).
- ✅ Pass: Compute SVD; reconstruct matrix from top-k components and plot reconstruction error vs k; project points onto a subspace; compute distances to hyperplanes.
- 🛠️ How: `np.linalg.svd`; projection formula; `np.linalg.lstsq` for least squares via normal equations and QR.

Week 12 — Vector Calculus I
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: Partial derivatives; gradients of scalar functions; Jacobians of vector functions.
- ✅ Pass: Compute gradients analytically for multivariate functions; verify with numerical finite differences; visualize gradient field on a contour plot.
- 🛠️ How: Derive gradient by hand; implement central differences `(f(x+h)-f(x-h))/(2h)`; `plt.contour` with `plt.quiver`.

Week 13 — Vector Calculus II
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: Chain rule for composed functions; backpropagation intuition; Hessians and second-order derivatives.
- ✅ Pass: Derive gradients of composed functions using chain rule; compute Hessian matrix; verify gradient computation with central-difference check (max abs diff < 1e-4).
- 🛠️ How: Symbolic differentiation by hand; numerical Hessian via finite differences; check gradient correctness.

Week 14 — Probability I
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: Random variables; probability distributions (discrete and continuous); expectation and variance; common distributions (Bernoulli, Binomial, Gaussian).
- ✅ Pass: Simulate samples from common distributions; compute empirical vs theoretical mean/variance; verify Law of Large Numbers by plotting sample mean convergence.
- 🛠️ How: `np.random`, `scipy.stats`; compare empirical moments to closed-form expressions.

Week 15 — Probability II
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: Joint and marginal distributions; covariance and correlation; multivariate Gaussian; Gaussian conditioning and marginalization.
- ✅ Pass: Generate correlated Normals via Cholesky decomposition; recover empirical covariance matrix; visualize 2D Gaussian contours; demonstrate conditioning a multivariate Gaussian.
- 🛠️ How: `L = np.linalg.cholesky(Sigma)`; `X = Z @ L.T`; `np.cov`; contour plots for bivariate Gaussian.

Week 16 — Optimization I
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: Convex functions and sets; gradient descent algorithm; learning rate selection; convergence analysis.
- ✅ Pass: Implement gradient descent for a convex quadratic f(x)=½x^TQx+c^Tx; show monotone loss decrease; experiment with different step sizes and plot convergence curves.
- 🛠️ How: Analytic gradient Qx+c; fixed and adaptive step sizes; plot loss vs iterations.

Week 17 — Optimization II
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: Newton's method; constrained optimization concepts; regularization and its geometric interpretation.
- ✅ Pass: Implement Newton's method using Hessian; compare convergence (iterations to tolerance) with gradient descent; solve ridge regression and visualize how λ affects the solution.
- 🛠️ How: Newton step: x_new = x - H⁻¹∇f; `scipy.optimize.minimize`; compare first-order vs second-order methods.

Week 18 — Review
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: Create concept map linking all MML topics; write summary notes connecting math foundations to ML applications.
- ✅ Pass: A one-page concept map with ≥10 explicit connections between math concepts and ML techniques (e.g., eigenvalues ↔ PCA, gradient descent ↔ neural network training, condition number ↔ numerical stability).
- 🛠️ How: Use mind-mapping tool or hand-drawn diagram; include concrete examples for each link.
</details>

🔁 Flex — Retrieval practice and summaries

---------------------------------------------------------------------

<details>
<summary><b>Phase 3 · Statistics Fundamentals — Weeks 19–24 (Complete Think Stats)</b></summary>

Week 19 — Think Stats Ch. 1
- 📖 [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- 🧪 Practice: Explore a dataset; compute summary statistics; build histograms and PMFs; construct ECDFs.
- ✅ Pass: Implement ECDF from scratch on real data; verify it is non-decreasing and ends at 1.0; overlay histogram and ECDF to compare distributional insights; interpret outliers.
- 🛠️ How: `np.sort`; `np.arange(1,n+1)/n`; `plt.step` for ECDF; `plt.hist` for histogram.

Week 20 — Think Stats Ch. 2
- 📖 [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- 🧪 Practice: Compute central tendency (mean, median, mode) and spread (variance, std, range, IQR); explore effect of outliers on these measures.
- ✅ Pass: Compare mean/SD vs median/MAD/IQR on 2 datasets (one symmetric, one skewed); explain when each measure is appropriate; show outlier impact graphically.
- 🛠️ How: `np.mean`, `np.median`, `np.std`; `scipy.stats.median_abs_deviation`; `np.percentile` for IQR.

Week 21 — Think Stats Ch. 3–4
- 📖 [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- 🧪 Practice: Work with CDFs and PMFs; model data with probability distributions; compare empirical vs theoretical distributions.
- ✅ Pass: Fit data to common distributions (Normal, Exponential); use CDF plots to assess fit; compute percentiles and quantiles; explain when to use PMF vs CDF.
- 🛠️ How: `scipy.stats.norm.fit`, `scipy.stats.expon`; `probplot` for QQ plots; CDF comparison plots.

Week 22 — Think Stats Ch. 5–6
- 📖 [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- 🧪 Practice: Model data with analytical distributions; explore relationships between variables; compute conditional probabilities.
- ✅ Pass: Fit a parametric model to real data; compute and interpret correlation and covariance; demonstrate conditional probability with a contingency table.
- 🛠️ How: `scipy.stats` distribution fitting; `np.corrcoef`; `pd.crosstab` for contingency tables.

Week 23 — Think Stats Ch. 7–8
- 📖 [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- 🧪 Practice: Estimate parameters with confidence intervals; perform hypothesis tests; understand p-values and statistical significance.
- ✅ Pass: Compute confidence intervals via bootstrap and analytical methods; run a hypothesis test; simulate to show Type I error ≈ α; produce a power curve for detecting effect sizes.
- 🛠️ How: Bootstrap resampling; `scipy.stats.ttest_ind`; simulation to count rejections under H₀ and H₁.

Week 24 — Think Stats Ch. 9–10 (+wrap)
- 📖 [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- 🧪 Practice: Explore linear relationships; fit simple and multiple regression; interpret coefficients; check regression assumptions.
- ✅ Pass: Fit OLS regression; interpret R², coefficients, and p-values; produce diagnostic plots (residuals vs fitted, QQ plot); compute VIFs and flag multicollinearity.
- 🛠️ How: `statsmodels.api.OLS`; `statsmodels.stats.outliers_influence.variance_inflation_factor`; diagnostic plots.
</details>

🔁 Flex — Stats recap

---------------------------------------------------------------------

<details>
<summary><b>Phase 4 · Applied Multivariate Statistics — Weeks 25–39 (Complete PSU STAT 505)</b></summary>

Week 25 — Measures of Central Tendency, Dispersion and Association (Lesson 1)
- 📖 Activities: [PSU STAT 505 Lesson 1](https://online.stat.psu.edu/stat505/lesson/1)
- 🧪 Practice: Compute multivariate measures of central tendency (mean vectors); understand dispersion (covariance matrices, generalized variance); compute association measures (correlation matrices); interpret relationships between variables.
- ✅ Pass: Calculate mean vectors and covariance matrices for multivariate data; compute correlation matrices and interpret relationships; calculate generalized variance and total variation; compare variance-covariance structures across groups. Complete PSU STAT 505 Lesson 1.
- 🛠️ How: `np.mean(axis=0)` for mean vectors; `np.cov` for covariance matrices; `np.corrcoef` for correlation; `np.linalg.det` for generalized variance; visualize with heatmaps.

Week 26 — Linear Combinations of Random Variables (Lesson 2)
- 📖 Activities: [PSU STAT 505 Lesson 2](https://online.stat.psu.edu/stat505/lesson/2)
- 🧪 Practice: Understand properties of linear combinations of random vectors; compute means and covariances of linear combinations; work with linear transformations; understand independence and correlation.
- ✅ Pass: Compute mean and covariance of linear combinations; verify properties of linear transformations; demonstrate how linear combinations preserve or change correlation structure; apply to dimensionality reduction scenarios. Complete PSU STAT 505 Lesson 2.
- 🛠️ How: Matrix operations for linear combinations `Y = AX + b`; compute `E[Y] = AE[X] + b` and `Cov(Y) = A Cov(X) A^T`; verify independence conditions; visualize transformations.

Week 27 — Graphical Display of Multivariate Data (Lesson 3)
- 📖 Activities: [PSU STAT 505 Lesson 3](https://online.stat.psu.edu/stat505/lesson/3)
- 🧪 Practice: Create various multivariate visualizations; produce scatter plot matrices, star plots, profile plots; understand 3D plots and contour plots; interpret patterns and relationships visually.
- ✅ Pass: Create comprehensive visualization suite: scatter plot matrix with correlations, 3D scatter plots, profile plots for repeated measures, star plots for multivariate observations, contour plots for bivariate distributions; identify patterns, outliers, and relationships. Complete PSU STAT 505 Lesson 3.
- 🛠️ How: `pd.plotting.scatter_matrix`; `mpl_toolkits.mplot3d` for 3D plots; `plt.plot` for profile plots; `seaborn.pairplot`; custom star/radar plots with `plt.subplot(projection='polar')`.

Week 28 — Multivariate Normal Distribution (Lesson 4)
- 📖 Activities: [PSU STAT 505 Lesson 4](https://online.stat.psu.edu/stat505/lesson/4)
- 🧪 Practice: Understand multivariate normal distribution properties; compute Mahalanobis distance; generate samples from MVN; test for multivariate normality; understand conditional and marginal distributions.
- ✅ Pass: Generate samples from multivariate normal; compute and interpret Mahalanobis distance vs Euclidean distance; perform Mardia's test for multivariate normality; compute marginal and conditional distributions; visualize MVN with contour plots. Complete PSU STAT 505 Lesson 4.
- 🛠️ How: `scipy.stats.multivariate_normal`; `scipy.spatial.distance.mahalanobis`; Mardia's skewness and kurtosis tests; compute conditionals using partitioned covariance matrices.

Week 29 — Sample Mean Vector and Sample Correlation (Lesson 5)
- 📖 Activities: [PSU STAT 505 Lesson 5](https://online.stat.psu.edu/stat505/lesson/5)
- 🧪 Practice: Estimate mean vectors and covariance matrices from samples; understand sampling distributions; perform inference on mean vectors; test hypotheses about correlations; construct confidence regions.
- ✅ Pass: Estimate mean vectors and covariance matrices; derive sampling distributions; construct confidence ellipses for mean vectors; test hypotheses about population means; compute standard errors for correlations. Complete PSU STAT 505 Lesson 5.
- 🛠️ How: Sample statistics with `np.mean`, `np.cov`; Wishart distribution for covariance; confidence ellipses using eigenvalues/eigenvectors; bootstrap for inference.

Week 30 — Multivariate Conditional Distribution and Partial Correlation (Lesson 6)
- 📖 Activities: [PSU STAT 505 Lesson 6](https://online.stat.psu.edu/stat505/lesson/6)
- 🧪 Practice: Compute conditional distributions from joint multivariate normal; calculate partial correlations; understand the difference between marginal and partial correlation; interpret conditional independence.
- ✅ Pass: Partition covariance matrices to compute conditional distributions; calculate and interpret partial correlations; compare partial vs marginal correlations; test for conditional independence; visualize relationships controlling for other variables. Complete PSU STAT 505 Lesson 6.
- 🛠️ How: Use partitioned covariance matrices `Σ = [[Σ11, Σ12], [Σ21, Σ22]]`; conditional mean `μ1 + Σ12 Σ22^-1 (x2 - μ2)`; conditional covariance `Σ11 - Σ12 Σ22^-1 Σ21`; `pingouin.partial_corr` for partial correlations.

Week 31 — Inferences Regarding Multivariate Population Mean (Lesson 7)
- 📖 Activities: [PSU STAT 505 Lesson 7](https://online.stat.psu.edu/stat505/lesson/7)
- 🧪 Practice: Perform Hotelling's T² tests for one-sample and two-sample problems; construct simultaneous confidence intervals; understand multivariate hypothesis testing; compare with univariate t-tests.
- ✅ Pass: Conduct one-sample Hotelling's T² test; perform two-sample Hotelling's T² test; construct simultaneous confidence intervals using Bonferroni correction; compare multivariate vs univariate approaches; interpret test statistics, p-values, and effect sizes. Complete PSU STAT 505 Lesson 7.
- 🛠️ How: Implement `T² = n(x̄ - μ0)^T S^-1 (x̄ - μ0)`; convert to F-statistic: `F = (n-p)T²/((n-1)p)`; `scipy.stats.f` for p-values; Bonferroni intervals: `t_(α/2p, n-1)`.

Week 32 — Multivariate Analysis of Variance (MANOVA) (Lesson 8)
- 📖 Activities: [PSU STAT 505 Lesson 8](https://online.stat.psu.edu/stat505/lesson/8)
- 🧪 Practice: Perform one-way and two-way MANOVA; understand Wilks' Lambda, Pillai's trace, and other test statistics; conduct post-hoc tests; check MANOVA assumptions; compare to univariate ANOVA.
- ✅ Pass: Run MANOVA with ≥2 dependent variables and ≥3 groups; report test statistics (Wilks' Lambda, Pillai's trace, Hotelling-Lawley trace, Roy's largest root); perform follow-up univariate ANOVAs and discriminant analysis; check assumptions (Box's M test, multivariate normality). Complete PSU STAT 505 Lesson 8.
- 🛠️ How: `statsmodels.multivariate.manova.MANOVA`; interpret output; visualize group centroids; check assumptions before interpretation; compare effect sizes across responses.

Week 33 — Repeated Measures Analysis (Lesson 9)
- 📖 Activities: [PSU STAT 505 Lesson 9](https://online.stat.psu.edu/stat505/lesson/9)
- 🧪 Practice: Analyze repeated measures data using multivariate approach; understand sphericity and compound symmetry; perform profile analysis; test for parallelism, coincidence, and flatness; handle within-subject factors.
- ✅ Pass: Analyze repeated measures design with multivariate approach; test sphericity assumption (Mauchly's test); perform profile analysis testing parallelism, levels, and flatness hypotheses; compare multivariate vs univariate repeated measures ANOVA; interpret within-subject and between-subject effects. Complete PSU STAT 505 Lesson 9.
- 🛠️ How: `statsmodels` for repeated measures; test sphericity; profile plots with error bars; Greenhouse-Geisser correction when sphericity violated; contrast matrices for specific comparisons.

Week 34 — Discriminant Analysis (Lesson 10)
- 📖 Activities: [PSU STAT 505 Lesson 10](https://online.stat.psu.edu/stat505/lesson/10)
- 🧪 Practice: Perform linear and quadratic discriminant analysis; understand Fisher's linear discriminant; classify observations; evaluate classification performance; understand relationship to MANOVA; compare LDA/QDA assumptions.
- ✅ Pass: Apply LDA and QDA to classification problem; compute discriminant functions and classify held-out observations; report confusion matrix and misclassification rates; visualize decision boundaries; compare LDA/QDA to logistic regression; verify equal covariance assumption. Complete PSU STAT 505 Lesson 10.
- 🛠️ How: `sklearn.discriminant_analysis.LinearDiscriminantAnalysis/QuadraticDiscriminantAnalysis`; `classification_report`; ROC curves; cross-validation for error estimation; Box's M test for covariance equality.

Week 35 — Principal Components Analysis (Lesson 11)
- 📖 Activities: [PSU STAT 505 Lesson 11](https://online.stat.psu.edu/stat505/lesson/11)
- 🧪 Practice: Perform PCA on correlation and covariance matrices; understand eigenvalues/eigenvectors interpretation; determine number of components; compute component scores; interpret loadings; create biplots; understand variance explained.
- ✅ Pass: Apply PCA to dataset with ≥6 variables; create scree plot; select components using Kaiser criterion (eigenvalue > 1) and cumulative variance (80%); interpret loadings for first 2-3 PCs; create biplot; reconstruct data; compare PCA on correlation vs covariance. Complete PSU STAT 505 Lesson 11.
- 🛠️ How: `sklearn.decomposition.PCA`; standardize with `StandardScaler`; `explained_variance_ratio_`; scree plot; biplot with `plt.arrow`; verify reconstruction error.

Week 36 — Factor Analysis (Lesson 12)
- 📖 Activities: [PSU STAT 505 Lesson 12](https://online.stat.psu.edu/stat505/lesson/12)
- 🧪 Practice: Perform exploratory factor analysis; understand factor model and common vs specific variance; estimate communalities and uniqueness; perform factor rotations (varimax, promax); determine number of factors; interpret factor loadings.
- ✅ Pass: Conduct factor analysis; determine number of factors using parallel analysis and scree plot; extract factors using maximum likelihood or principal axis factoring; perform varimax and promax rotations; interpret and name factors; report communalities and variance explained; compare to PCA. Complete PSU STAT 505 Lesson 12.
- 🛠️ How: `sklearn.decomposition.FactorAnalysis`; `factor_analyzer` package for rotations; parallel analysis comparing eigenvalues to random data; factor loading interpretation with cutoff |loading| > 0.3.

Week 37 — Canonical Correlation Analysis (Lesson 13)
- 📖 Activities: [PSU STAT 505 Lesson 13](https://online.stat.psu.edu/stat505/lesson/13)
- 🧪 Practice: Perform canonical correlation analysis between two sets of variables; compute canonical correlations and canonical variates; test significance; interpret canonical loadings and cross-loadings; assess redundancy.
- ✅ Pass: Apply CCA to dataset with two variable sets (≥3 variables each); compute all canonical correlations and test significance; interpret first 2-3 canonical variate pairs; compute canonical loadings (structure correlations); perform redundancy analysis; visualize canonical variates. Complete PSU STAT 505 Lesson 13.
- 🛠️ How: `sklearn.cross_decomposition.CCA`; Wilks' Lambda test: `Λ = ∏(1 - r²)`; canonical loadings as correlations between original variables and canonical variates; redundancy index.

Week 38 — Cluster Analysis (Lesson 14)
- 📖 Activities: [PSU STAT 505 Lesson 14](https://online.stat.psu.edu/stat505/lesson/14)
- 🧪 Practice: Apply hierarchical clustering with different linkage methods; perform k-means clustering; understand distance measures and similarity metrics; determine optimal number of clusters; validate clustering solutions; compare clustering methods.
- ✅ Pass: Perform hierarchical clustering with ≥3 linkage methods (single, complete, average, Ward); create dendrograms; apply k-means with multiple k values; determine optimal k using elbow method, silhouette analysis, and gap statistic; validate with silhouette scores; visualize clusters using PCA; compare hierarchical vs partitioning methods. Complete PSU STAT 505 Lesson 14.
- 🛠️ How: `scipy.cluster.hierarchy` for hierarchical clustering; `sklearn.cluster.KMeans`; distance metrics: Euclidean, Manhattan, Mahalanobis; silhouette analysis; dendrogram interpretation; standardize data before clustering.

Week 39 — Integration and Review
- 📖 Activities: Review all PSU STAT 505 lessons
- 🧪 Practice: Integrate multivariate methods in comprehensive analysis; understand when to use each technique; compare and contrast methods; apply multiple techniques to same dataset.
- ✅ Pass: Complete end-to-end multivariate analysis applying ≥5 techniques from course; write comprehensive report connecting methods; explain method selection rationale; interpret results in context; discuss assumptions and limitations; compare insights from different methods.
- 🛠️ How: Choose appropriate methods for research question; check assumptions; compare complementary analyses (e.g., PCA then cluster analysis; MANOVA then discriminant analysis); synthesize findings.
</details>

🔁 Flex — Multivariate stats consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 5 · Bayesian & Missing Data — Weeks 40–51 (Complete Think Bayes, FIMD)</b></summary>

Weeks 40–47 — Think Bayes (Ch. 1–14, paced)
- 📖 [Think Bayes](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple)
- 🧪 Practice: Apply Bayes' theorem to update beliefs; implement conjugate prior models (Beta-Binomial, Gamma-Poisson, Normal-Normal); perform posterior predictive checks; compare models.
- ✅ Pass (weekly): Implement a Bayesian model aligned with the chapter's topic; show prior sensitivity analysis (vary prior parameters and observe posterior changes); generate posterior predictive samples and compare to observed data using a suitable test statistic.
- 🛠️ How: Use analytical posteriors when available; for PPC, draw samples from posterior, then from likelihood, and compare summary stats to data.

Weeks 48–51 — Flexible Imputation of Missing Data (complete)
- 📖 [FIMD](https://stefvanbuuren.name/fimd/)
- 🧪 Practice: Missingness mechanisms; MICE; sensitivity (as in book)
- ✅ Pass (weekly): Run MICE (m≥5) on a dataset; report pooled estimates per Rubin’s rules; compare to complete-case; perform delta-adjustment sensitivity where relevant.
- 🛠️ How: use a MICE implementation (e.g., statsmodels/impyute/sklearn-iterative as proxy) consistent with book procedures.
</details>

🔁 Flex — Consolidate Bayesian + MI

---------------------------------------------------------------------

<details>
<summary><b>Phase 6 · Statistical Learning with Python — Weeks 52–61 (Complete ISLP)</b></summary>

Week 52 — ISLP Ch. 1–2 (Intro + Statistical Learning)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand the statistical learning framework; implement train/test splits; explore the bias-variance trade-off with KNN at different k values.
- ✅ Pass: On a dataset, demonstrate how training error decreases with model complexity while test error shows U-shape; implement 5-fold CV and compare to hold-out estimate; discuss flexibility vs interpretability.
- 🛠️ How: `train_test_split`; `KFold`/`cross_val_score`; vary KNN's k parameter; plot training vs test error curves.

Week 53 — ISLP Ch. 3 (Linear Regression)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Fit simple and multiple linear regression; interpret coefficients; add interaction and polynomial terms; assess model fit with residual diagnostics.
- ✅ Pass: Fit OLS with and without interaction/polynomial terms; compare R² vs adjusted R²; produce residual plots; select optimal polynomial degree via CV; interpret coefficient confidence intervals.
- 🛠️ How: `LinearRegression`; `PolynomialFeatures`; `cross_val_score`; `statsmodels` for CIs; residual diagnostics.

Week 54 — ISLP Ch. 4 (Classification)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Implement logistic regression; understand LDA/QDA assumptions; apply KNN for classification; explore classification metrics beyond accuracy.
- ✅ Pass: Compare logistic regression, LDA, QDA, and KNN using stratified 5-fold CV; report confusion matrix, precision, recall, and ROC-AUC; select optimal classification threshold based on problem context.
- 🛠️ How: `LogisticRegression`; `LinearDiscriminantAnalysis`; `QuadraticDiscriminantAnalysis`; `KNeighborsClassifier`; `roc_curve` for threshold selection.

Week 55 — ISLP Ch. 5 (Resampling Methods)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Compare validation approaches: hold-out, LOOCV, k-fold CV; use bootstrap for uncertainty estimation; understand variance-bias trade-off in resampling.
- ✅ Pass: Compare test error estimates from LOOCV vs 5-fold vs 10-fold CV; implement bootstrap to estimate coefficient standard errors; compare bootstrap SEs to analytic SEs.
- 🛠️ How: `LeaveOneOut`; `KFold`; implement bootstrap loop with `np.random.choice`; fix seeds for reproducibility.

Week 56 — ISLP Ch. 6 (Model Selection & Regularization)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand the motivation for regularization; implement ridge and lasso regression; interpret coefficient shrinkage and sparsity; tune regularization parameter via CV.
- ✅ Pass: Plot ridge and lasso coefficient paths as λ varies; select optimal λ via CV; compare test error of OLS vs ridge vs lasso; explain when lasso produces sparse solutions.
- 🛠️ How: `Ridge`; `Lasso`; `RidgeCV`; `LassoCV`; `StandardScaler` (scale features first); `lasso_path` for path plots.

Week 57 — ISLP Ch. 7 (Beyond Linearity)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Move beyond linearity with polynomial regression, step functions, and splines; understand degrees of freedom; fit GAM-style models.
- ✅ Pass: Fit polynomial, step function, and spline models; compare their flexibility and test errors; produce partial dependence plots; select appropriate number of knots/degrees via CV.
- 🛠️ How: `PolynomialFeatures`; `SplineTransformer`; `pd.cut` for step functions; compare MSE on held-out data.

Week 58 — ISLP Ch. 8 (Tree-Based Methods)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Fit decision trees; understand bagging and the random forest algorithm; implement gradient boosting; interpret tree-based models.
- ✅ Pass: Fit and prune a decision tree; compare single tree vs random forest vs gradient boosting on test error; show OOB error for RF; plot feature importances and partial dependence plots.
- 🛠️ How: `DecisionTreeClassifier/Regressor`; `RandomForestClassifier/Regressor`; `GradientBoostingClassifier/Regressor`; `permutation_importance`; `plot_partial_dependence`.

Week 59 — ISLP Ch. 9 (Support Vector Machines)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand maximal margin classifiers and support vectors; fit SVMs with linear and non-linear kernels; tune hyperparameters (C, gamma).
- ✅ Pass: Fit SVM with linear and RBF kernels; tune C and gamma via grid search with CV; visualize decision boundaries on 2D data; identify and highlight support vectors; compare to logistic regression.
- 🛠️ How: `SVC`; `GridSearchCV`; `plt.contourf` for decision boundaries; access `support_vectors_` attribute.

Week 60 — ISLP Ch. 10 (Unsupervised Learning)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Perform dimensionality reduction with PCA; apply k-means and hierarchical clustering; understand the importance of scaling; evaluate clustering quality.
- ✅ Pass: Apply PCA and plot cumulative explained variance; choose number of components; cluster with k-means (elbow method for k) and hierarchical clustering (dendrogram); evaluate with silhouette score and compare cluster stability across random seeds.
- 🛠️ How: `StandardScaler` (always scale first); `PCA`; `KMeans` with inertia plots; `AgglomerativeClustering`; `dendrogram`; `silhouette_score`.

Week 61 — ISLP Labs/Wrap-up
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Complete an end-to-end ML project using techniques from all ISLP chapters: EDA, preprocessing, model selection, hyperparameter tuning, evaluation, and interpretation.
- ✅ Pass: Deliver a reproducible notebook with proper train/test split, cross-validation, model comparison, hyperparameter tuning, error analysis, and a 1-page summary documenting decisions, limitations, and risks.
- 🛠️ How: `Pipeline`; `ColumnTransformer` for mixed feature types; `GridSearchCV`/`RandomizedSearchCV`; fixed `random_state` throughout; clean documentation.
</details>

🔁 Flex — Validation basics consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 7 · Classical ML — Weeks 62–80 (Complete PRML, Interpretable ML)</b></summary>

Weeks 62–75 — PRML (Ch. 1–13 + review)
- 📖 [PRML (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- 🧪 Practice: Implement core algorithms from each chapter from scratch: probability distributions, linear models, neural networks, kernel methods, graphical models, mixture models, EM algorithm, approximate inference, and sampling methods.
- ✅ Pass (weekly): Implement the chapter's focal algorithm from scratch; verify correctness by comparing to sklearn/scipy baseline (within 2-5% accuracy); document mathematical derivations; use fixed seeds for reproducibility.
- 🛠️ How: Use NumPy for implementations; sklearn only as verification oracle; work on toy datasets; keep detailed notes linking code to book equations.

Weeks 76–80 — Interpretable ML (complete)
- 📖 [Interpretable ML](https://christophm.github.io/interpretable-ml-book/)
- 🧪 Practice: Apply model-agnostic interpretation methods: PDP, ICE, permutation importance, LIME, SHAP; understand intrinsically interpretable models; explore feature interaction methods.
- ✅ Pass (weekly): For a trained model, produce PDP/ICE plots for top features; compute permutation importance; generate SHAP values for individual predictions; write a 1-page analysis comparing methods' stability across 3 bootstrap resamples.
- 🛠️ How: `sklearn.inspection.PartialDependenceDisplay`; `permutation_importance`; `shap.Explainer`; compare explanations across train/test sets.
</details>

🔁 Flex — Validation & interpretation synthesis

---------------------------------------------------------------------

<details>
<summary><b>Phase 8 · Data Mining — Weeks 81–89 (Complete DM 3e)</b></summary>

Weeks 81–89 — Data Mining 3e (Ch. 1–12)
- 📖 [Data Mining 3e (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- 🧪 Practice: Per-chapter algorithmic work strictly matching the chapter (e.g., data preprocessing tasks; Apriori/FP-Growth; decision trees; k-means/DBSCAN; outlier detection)
- ✅ Pass (weekly): Implement a minimal working version for the chapter’s focal algorithm OR replicate results using a library; verify correctness on a deterministic toy and compare performance on a small real dataset.
- 🛠️ How: construct small synthetic datasets with known ground truth (fixed seeds); assert counts/clusters/rules match expectation.
</details>

🔁 Flex — Mining recap

---------------------------------------------------------------------

<details>
<summary><b>Phase 9 · Econometrics & Time Series — Weeks 90–111 (Complete Gujarati, Lütkepohl)</b></summary>

Weeks 90–101 — Basic Econometrics (complete)
- 📖 [Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf)
- 🧪 Practice: Reproduce a worked example per chapter using methods from that chapter only (OLS basics; classical assumption diagnostics; heteroskedasticity/autocorrelation remedies; functional form; limited dependent variables as presented)
- ✅ Pass (weekly): Match the textbook example’s coefficients and standard errors (within rounding) and include one robustness check discussed in that chapter (e.g., robust/HAC SEs when appropriate).
- 🛠️ How: `statsmodels` OLS/GLM, `cov_type="HC3"` or HAC if the chapter addresses it; include diagnostic plots taught there.

Weeks 102–111 — Lütkepohl (complete)
- 📖 [Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)
- 🧪 Practice: Implement multivariate time series analysis: VAR model specification, estimation, lag order selection, stability analysis, impulse response functions, forecast error variance decomposition, and cointegration/VECM.
- ✅ Pass (weekly): Fit VAR/VECM to macroeconomic data; select lag order using information criteria; verify stability (roots inside unit circle); compute and plot IRFs with confidence bands; perform Johansen cointegration test when applicable.
- 🛠️ How: `statsmodels.tsa.api.VAR`; `statsmodels.tsa.vector_ar.vecm.VECM`; `irf()` for impulse responses; rolling-window forecasts for evaluation.
</details>

🔁 Flex — Econometrics/time-series consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 10 · R for Data Science — Weeks 112–121 (Complete R4DS 2e)</b></summary>

Weeks 112–121 — R4DS (Complete)
- 📖 [R for Data Science (2e)](https://r4ds.hadley.nz)
- 🧪 Practice: Learn R and tidyverse progressively: data import, tidying (pivot_longer/wider), transformation (dplyr verbs), visualization (ggplot2), strings, factors, dates, functions, iteration, and communication (Quarto/RMarkdown).
- ✅ Pass (weekly): Complete a mini-analysis using only functions from chapters covered that week; produce a Quarto/RMarkdown report that renders end-to-end; include at least one visualization and one summary table.
- 🛠️ How: `library(tidyverse)`; `read_csv`; `dplyr` verbs (`filter`, `mutate`, `summarize`, `group_by`); `ggplot2`; `set.seed()` for reproducibility.
</details>

🔁 Flex — R consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 11 · Web Scraping & SQL — Weeks 122–127 (Complete BeautifulSoup, Selenium, SQL)</b></summary>

Week 122 — BeautifulSoup
- 📖 [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- 🧪 Practice: Scrape static HTML pages: fetch with requests, parse with BeautifulSoup, navigate the DOM, extract data using CSS selectors and tag methods.
- ✅ Pass: Scrape a static website and extract structured data; save as CSV/JSON with documented schema; check robots.txt before scraping; implement polite delays to avoid rate limiting (no HTTP 429 errors).
- 🛠️ How: `requests.get(url)`; `BeautifulSoup(html, "lxml")`; `.select()` for CSS selectors; `.find_all()` for tag-based search; `time.sleep()` between requests.

Weeks 123–124 — Selenium
- 📖 [Selenium (Python)](https://selenium-python.readthedocs.io/index.html)
- 🧪 Practice: Automate browser interactions for dynamic websites: handle JavaScript-rendered content, implement explicit waits, manage pagination and infinite scroll, fill forms.
- ✅ Pass (weekly): Scrape a JavaScript-rendered page (e.g., infinite scroll or content behind clicks); implement proper waits and error handling; save timestamped data with retry/timeout logs; handle at least one failure scenario gracefully.
- 🛠️ How: `webdriver.Chrome()`; `WebDriverWait` with `expected_conditions`; CSS/XPath selectors; `execute_script()` for scrolling; consistent viewport settings.

Week 125 — SQL Basics & Data Retrieval
- 📖 [SQL Roadmap (GeeksforGeeks)](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- 🧪 Practice: Set up SQL environment (SQLite/PostgreSQL/MySQL); understand database structure (tables, rows, columns); master SELECT statement; use WHERE clause with comparison operators (=, !=, <, >, <=, >=); apply logical operators (AND, OR, NOT); sort results with ORDER BY; limit results with LIMIT/TOP; use DISTINCT to remove duplicates; practice basic string matching with LIKE and wildcards (%, _).
- ✅ Pass: Install and configure a SQL database system; create a sample database with at least 3 tables; write ≥20 SELECT queries demonstrating: simple selection, filtering with WHERE, multiple conditions with AND/OR, sorting ascending/descending, limiting results, removing duplicates, and pattern matching with LIKE; document each query with its purpose and expected row count.
- 🛠️ How: Install SQLite (lightweight) or PostgreSQL (production-grade); use `CREATE TABLE`, `INSERT INTO` for sample data; practice `SELECT * FROM table`, `SELECT col1, col2 FROM table WHERE condition`, `ORDER BY col ASC/DESC`, `LIMIT n`, `DISTINCT col`, `WHERE col LIKE 'pattern%'`; use a SQL client (DBeaver, pgAdmin, or command line).

Week 126 — SQL Joins, Aggregations & Subqueries
- 📖 [SQL Roadmap (GeeksforGeeks)](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- 🧪 Practice: Master different types of joins (INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL OUTER JOIN, CROSS JOIN, SELF JOIN); understand when to use each join type; work with aggregate functions (COUNT, SUM, AVG, MIN, MAX); use GROUP BY for grouping data; filter grouped data with HAVING clause; write subqueries in SELECT, WHERE, and FROM clauses; understand correlated vs non-correlated subqueries; practice set operations (UNION, UNION ALL, INTERSECT, EXCEPT); work with NULL values (IS NULL, IS NOT NULL, COALESCE, NULLIF).
- ✅ Pass: Create a relational schema with ≥3 related tables (e.g., customers, orders, products); write ≥25 queries demonstrating: all join types with explanations, aggregate functions with and without GROUP BY, HAVING clause filters, nested subqueries (at least 2 levels deep), correlated subqueries, set operations combining multiple queries, NULL handling in various contexts; include at least one complex multi-join query involving 3+ tables; document the business question each query answers.
- 🛠️ How: Design schema with foreign key relationships; `INNER JOIN table2 ON table1.id = table2.fk_id`, `LEFT JOIN` for including unmatched rows; `SELECT COUNT(*), AVG(col) FROM table GROUP BY category`, `HAVING COUNT(*) > 5`; `WHERE col IN (SELECT...)`, `SELECT (SELECT...) AS subquery_col`; `UNION` to combine results; `COALESCE(col, 'default')` for NULL handling; visualize join results with Venn diagrams.

Week 127 — Advanced SQL: Window Functions, CTEs & Performance
- 📖 [SQL Roadmap (GeeksforGeeks)](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- 🧪 Practice: Master window functions (ROW_NUMBER, RANK, DENSE_RANK, NTILE); use aggregate window functions (SUM, AVG, COUNT over partitions); apply LEAD and LAG for accessing adjacent rows; work with PARTITION BY and ORDER BY in window functions; write Common Table Expressions (CTEs) for readable complex queries; use recursive CTEs for hierarchical data; understand query execution plans and optimization; create and use indexes for performance; practice transactions (BEGIN, COMMIT, ROLLBACK); work with views and stored procedures; implement data manipulation (INSERT, UPDATE, DELETE) with complex conditions; understand constraints (PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK); explore date/time functions and string manipulation functions.
- ✅ Pass: Build a complete analytics database schema with ≥4 tables and proper constraints; write ≥30 advanced queries including: ≥10 window function queries (ranking, running totals, moving averages, lag/lead analysis), ≥5 CTEs (including at least 1 recursive CTE for hierarchical data like org charts or category trees), ≥5 queries with EXPLAIN/ANALYZE showing index usage, ≥3 transactions demonstrating ACID properties, ≥3 views encapsulating complex logic, ≥5 DML operations (INSERT/UPDATE/DELETE with subqueries), date calculations (date differences, date parts, date formatting), and complex string manipulations; create indexes and demonstrate query performance improvement; document each query with execution time before/after optimization where applicable.
- 🛠️ How: Window functions: `ROW_NUMBER() OVER (PARTITION BY col ORDER BY col2)`, `SUM(col) OVER (PARTITION BY category ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` for running totals; CTEs: `WITH cte_name AS (SELECT...) SELECT * FROM cte_name`; Recursive CTE: `WITH RECURSIVE cte AS (base_case UNION ALL recursive_case) SELECT * FROM cte`; Indexes: `CREATE INDEX idx_name ON table(col)`; check execution plan with `EXPLAIN` (PostgreSQL) or `EXPLAIN QUERY PLAN` (SQLite); transactions: `BEGIN; UPDATE...; COMMIT;`; views: `CREATE VIEW view_name AS SELECT...`; use `EXTRACT`, `DATE_TRUNC`, `AGE` for dates; `CONCAT`, `SUBSTRING`, `REGEXP_REPLACE` for strings; optimize with proper WHERE clause ordering and index hints.

</details>

🔁 Flex — ETL mini-project

---------------------------------------------------------------------

<details>
<summary><b>Phase 12 · Deep Learning — Weeks 128–147 (Complete D2L fundamentals, Goodfellow DL)</b></summary>

Weeks 128–135 — D2L (Fundamentals)
- 📖 [D2L](https://d2l.ai)
- 🧪 Practice: Topic-specific small models exactly as covered (MLP, CNN, RNN; optimization; regularization; data pipelines)
- ✅ Pass (weekly): Train the chapter’s model variant on a toy dataset with fixed seeds and one controlled ablation (optimizer OR regularization) taught in D2L; log curves/metrics.
- 🛠️ How: Follow D2L’s PyTorch/MXNet examples; fix seeds; keep experiments minimal and reproducible.

Week 136 — The Illustrated Transformer (Bridge)
- 📖 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- 🧪 Practice: Understand the Transformer architecture: self-attention mechanism, multi-head attention, positional encoding, encoder-decoder structure.
- ✅ Pass: Implement self-attention from scratch; verify tensor shapes at each step; implement attention masking; write unit tests for: (1) output shape correctness, (2) masked positions get zero attention, (3) attention weights sum to 1.
- 🛠️ How: Use NumPy or PyTorch; implement Q, K, V projections; scaled dot-product attention; verify with `assert` statements and test cases.

Weeks 137–147 — Deep Learning Book (Complete)
- 📖 [Deep Learning Book](https://www.deeplearningbook.org/)
- 🧪 Practice: For each chapter, run a small experiment that demonstrates the chapter’s key concept using building blocks learned in D2L
- ✅ Pass (weekly): Provide a controlled comparison or demonstration plot showing the expected qualitative effect (e.g., different inits, L2 vs dropout, step-size schedules).
- 🛠️ How: Small synthetic or standard toy datasets; fixed seeds; log and compare curves cleanly.
</details>

🔁 Flex — DL recap + tracked mini project

---------------------------------------------------------------------

<details>
<summary><b>Phase 13 · MLOps & Data Engineering — Weeks 148–171 (Complete Zoomcamps, ML Systems)</b></summary>

Weeks 148–155 — MLOps Zoomcamp
- 📖 [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- 🧪 Practice: Module-by-module implementation as taught (tracking, packaging, CI, serving, orchestration, monitoring)
- ✅ Pass (weekly): A runnable local pipeline from clean state to served endpoint with tests passing for that week’s scope.
- 🛠️ How: Docker/Compose; MLflow/W&B; `pytest`; minimal infra defined as per module.

Weeks 156–163 — Machine Learning Systems
- 📖 [ML Systems](https://mlsysbook.ai)
- 🧪 Practice: Write/extend a system design doc each week focusing only on that week’s concepts (SLA/SLOs; rollout/rollback; monitoring; data contracts; cost/reliability)
- ✅ Pass (weekly): The doc includes concrete metrics, failure scenarios, and operational procedures aligned to the chapter.
- 🛠️ How: ADR template; simple diagrams-as-code optional (e.g., Mermaid).

Weeks 164–171 — Data Engineering Zoomcamp
- 📖 [DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
- 🧪 Practice: Module-by-module pipeline work (ingestion, storage, batch/stream, orchestration, analytics eng, testing) as taught in the course
- ✅ Pass (weekly): Re-deployable pipeline from scratch with idempotent runs for that module’s scope.
- 🛠️ How: Terraform/Docker where required, dbt, Airflow/Prefect according to the module.
</details>

🔁 Flex — Ops/engineering consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 14 · LLMs & Agents — Weeks 172–175 (Complete HF Course + Agents)</b></summary>

Weeks 172–174 — Hugging Face Course
- 📖 [HF Course](https://huggingface.co/course/chapter1)
- 🧪 Practice: Learn the Hugging Face ecosystem: load and preprocess datasets, understand tokenizers, fine-tune pretrained models, run inference, evaluate with appropriate metrics.
- ✅ Pass (weekly): Complete the course exercises for that week's chapters; fine-tune a small transformer on a downstream task (e.g., text classification, NER); evaluate with task-appropriate metrics (accuracy, F1, etc.); log all configurations.
- 🛠️ How: `transformers` library for models; `datasets` for data loading; `Trainer` API for fine-tuning; `accelerate` for distributed training; Weights & Biases or TensorBoard for logging.

Week 175 — HF Agents
- 📖 [HF Agents](https://huggingface.co/learn/agents-course/unit0/introduction)
- 🧪 Practice: Build AI agents that use tools: understand agent architectures, implement tool calling, handle errors and timeouts, implement safety guardrails.
- ✅ Pass: Build an agent that completes a multi-step task using external tools; implement proper timeout handling; test with an injected failure scenario and verify graceful degradation; document safety checks and limitations.
- 🛠️ How: Use Hugging Face agents framework; implement `Tool` classes; set timeouts with `asyncio.timeout` or similar; log all tool calls and responses; implement input validation.
</details>

---------------------------------------------------------------------

<details>
<summary><b>Phase 15 · Consolidation, Capstone, Portfolio — Weeks 176–179</b></summary>

Week 176 — statsmodels deep dive
- 📖 [statsmodels](https://www.statsmodels.org/stable/index.html)
- 🧪 Practice: Master statsmodels by reproducing analyses from earlier phases: OLS with diagnostics, GLMs, time series models (ARIMA, VAR), hypothesis testing.
- ✅ Pass: Reproduce two econometric analyses matching original coefficients and standard errors; include full diagnostic suite (heteroskedasticity, autocorrelation tests); apply robust SEs where violations exist.
- 🛠️ How: `statsmodels.api.OLS/GLM`; `statsmodels.tsa` for time series; `het_breuschpagan`, `acorr_ljungbox` for diagnostics; `cov_type="HC3"` for robust SEs.

Week 177 — scikit-learn deep dive
- 📖 [scikit-learn](https://scikit-learn.org/stable/index.html)
- 🧪 Practice: Create a production-ready ML pipeline template: preprocessing (scaling, encoding), feature selection, model training with CV, hyperparameter tuning, probability calibration.
- ✅ Pass: Build a complete Pipeline with ColumnTransformer for mixed types; implement nested CV for unbiased evaluation; apply probability calibration (Platt scaling or isotonic); ensure deterministic results with fixed seeds.
- 🛠️ How: `Pipeline`; `ColumnTransformer`; `GridSearchCV`/`RandomizedSearchCV`; `CalibratedClassifierCV`; fixed `random_state` throughout.

Weeks 178–179 — Capstone & Portfolio
- 📖 Integrate end-to-end skills only from prior phases
- 🧪 Practice: Complete a capstone project demonstrating: problem framing, data pipeline, modeling with uncertainty quantification, model interpretation, rigorous evaluation, and stakeholder communication.
- ✅ Pass: Deliver a fully reproducible project (single command to run); include README documenting problem, approach, assumptions, limitations, and risks; provide model interpretation (SHAP/PDP); write a 1-page non-technical summary for stakeholders.
- 🛠️ How: Use Git for version control; Docker for reproducibility; include uncertainty estimates (bootstrap CIs or Bayesian); create visualizations for non-technical audience; document all decisions.
</details>

---------------------------------------------------------------------

Resource-to-Week Completion Map (cover-to-cover)
- Python for Data Analysis — Weeks 1–8 — [Python for Data Analysis](https://wesmckinney.com/book/)
- Mathematics for Machine Learning — Weeks 9–18 — [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- Think Stats — Weeks 19–24 — [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- PSU STAT 505 (Applied Multivariate Statistics) — Weeks 25–39 — [PSU STAT 505](https://online.stat.psu.edu/stat505)
- Think Bayes — Weeks 40–47 — [Think Bayes](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple)
- Flexible Imputation of Missing Data — Weeks 48–51 — [FIMD](https://stefvanbuuren.name/fimd/)
- ISLP (Statistical Learning with Python) — Weeks 52–61 — [ISLP](https://www.statlearning.com/)
- PRML (Bishop) — Weeks 62–75 — [PRML (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- Interpretable Machine Learning — Weeks 76–80 — [Interpretable ML](https://christophm.github.io/interpretable-ml-book/)
- Data Mining: Concepts and Techniques (3e) — Weeks 81–89 — [Data Mining 3e (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- Basic Econometrics (Gujarati) — Weeks 90–101 — [Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf)
- New Introduction to Multiple Time Series (Lütkepohl) — Weeks 102–111 — [Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)
- R for Data Science (2e) — Weeks 112–121 — [R for Data Science (2e)](https://r4ds.hadley.nz)
- Beautiful Soup — Week 122 — [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- Selenium (Python) — Weeks 123–124 — [Selenium (Python)](https://selenium-python.readthedocs.io/index.html)
- SQL Roadmap (GeeksforGeeks) — Weeks 125–127 — [SQL Roadmap](https://www.geeksforgeeks.org/blogs/sql-roadmap/)
- Dive into Deep Learning — Weeks 128–135 — [D2L](https://d2l.ai)
- The Illustrated Transformer — Week 136 — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Deep Learning — Weeks 137–147 — [Deep Learning Book](https://www.deeplearningbook.org/)
- MLOps Zoomcamp — Weeks 148–155 — [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- Machine Learning Systems — Weeks 156–163 — [ML Systems](https://mlsysbook.ai)
- Data Engineering Zoomcamp — Weeks 164–171 — [DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
- HF Course + HF Agents — Weeks 172–175 — [HF Course](https://huggingface.co/course/chapter1), [HF Agents](https://huggingface.co/learn/agents-course/unit0/introduction)

Notes
- Keep work in any format; seed randomness for reproducibility.
- Use Flex Weeks to finish pass items, review tricky parts, and add spaced-repetition cards (optional).
- Back to top ↑
