# AI & Data Science Weekly Plan — Activities, Practice, and Pass Criteria

![Duration](https://img.shields.io/badge/duration-~164_weeks-6f42c1)
![Pace](https://img.shields.io/badge/pace-8–10_h%2Fweek-0e8a16)
![Path](https://img.shields.io/badge/path-beginner%E2%86%92practitioner-0366d6)
![Style](https://img.shields.io/badge/style-cumulative%2C_concept%E2%86%92practice-555)

Zero prior knowledge is assumed. Learning order is strictly top-to-bottom. Each week includes a clear “Pass” requirement aligned to the primary resource.

— Quick jump —
- Phase 1 · Data Analysis Foundations
- Phase 2 · Mathematics for ML
- Phase 3 · Statistics Fundamentals
- Phase 4 · Bayesian Statistics & Missing Data
- Phase 5 · Statistical Learning with Python (ISLP)
- Phase 6 · Classical ML
- Phase 7 · Data Mining
- Phase 8 · Econometrics & Time Series
- Phase 9 · R for Data Science
- Phase 10 · Web Scraping & SQL
- Phase 11 · Deep Learning
- Phase 12 · MLOps & Data Engineering
- Phase 13 · LLMs & Open-Source AI
- Phase 14 · Consolidation & Capstone

Legend
- 📖 Activities (primary source)
- 🧪 Practice (small tasks)
- ✅ Pass (weekly pass criterion)
- 🛠️ How (implementation hint)
- 🔁 Flex (catch-up, spaced review)

Duration and pacing
- Duration: ~164 weeks (≈3.1 years), 8–10 h/week
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
- SQL Tutorial — [SQL Tutorial](https://www.sqltutorial.org/)
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
<summary><b>Phase 4 · Bayesian & Missing Data — Weeks 25–36 (Complete Think Bayes, FIMD)</b></summary>

Weeks 25–32 — Think Bayes (Ch. 1–14, paced)
- 📖 [Think Bayes](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple)
- 🧪 Practice: Apply Bayes' theorem to update beliefs; implement conjugate prior models (Beta-Binomial, Gamma-Poisson, Normal-Normal); perform posterior predictive checks; compare models.
- ✅ Pass (weekly): Implement a Bayesian model aligned with the chapter's topic; show prior sensitivity analysis (vary prior parameters and observe posterior changes); generate posterior predictive samples and compare to observed data using a suitable test statistic.
- 🛠️ How: Use analytical posteriors when available; for PPC, draw samples from posterior, then from likelihood, and compare summary stats to data.

Weeks 33–36 — Flexible Imputation of Missing Data (complete)
- 📖 [FIMD](https://stefvanbuuren.name/fimd/)
- 🧪 Practice: Missingness mechanisms; MICE; sensitivity (as in book)
- ✅ Pass (weekly): Run MICE (m≥5) on a dataset; report pooled estimates per Rubin’s rules; compare to complete-case; perform delta-adjustment sensitivity where relevant.
- 🛠️ How: use a MICE implementation (e.g., statsmodels/impyute/sklearn-iterative as proxy) consistent with book procedures.
</details>

🔁 Flex — Consolidate Bayesian + MI

---------------------------------------------------------------------

<details>
<summary><b>Phase 5 · Statistical Learning with Python — Weeks 37–46 (Complete ISLP)</b></summary>

Week 37 — ISLP Ch. 1–2 (Intro + Statistical Learning)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand the statistical learning framework; implement train/test splits; explore the bias-variance trade-off with KNN at different k values.
- ✅ Pass: On a dataset, demonstrate how training error decreases with model complexity while test error shows U-shape; implement 5-fold CV and compare to hold-out estimate; discuss flexibility vs interpretability.
- 🛠️ How: `train_test_split`; `KFold`/`cross_val_score`; vary KNN's k parameter; plot training vs test error curves.

Week 38 — ISLP Ch. 3 (Linear Regression)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Fit simple and multiple linear regression; interpret coefficients; add interaction and polynomial terms; assess model fit with residual diagnostics.
- ✅ Pass: Fit OLS with and without interaction/polynomial terms; compare R² vs adjusted R²; produce residual plots; select optimal polynomial degree via CV; interpret coefficient confidence intervals.
- 🛠️ How: `LinearRegression`; `PolynomialFeatures`; `cross_val_score`; `statsmodels` for CIs; residual diagnostics.

Week 39 — ISLP Ch. 4 (Classification)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Implement logistic regression; understand LDA/QDA assumptions; apply KNN for classification; explore classification metrics beyond accuracy.
- ✅ Pass: Compare logistic regression, LDA, QDA, and KNN using stratified 5-fold CV; report confusion matrix, precision, recall, and ROC-AUC; select optimal classification threshold based on problem context.
- 🛠️ How: `LogisticRegression`; `LinearDiscriminantAnalysis`; `QuadraticDiscriminantAnalysis`; `KNeighborsClassifier`; `roc_curve` for threshold selection.

Week 40 — ISLP Ch. 5 (Resampling Methods)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Compare validation approaches: hold-out, LOOCV, k-fold CV; use bootstrap for uncertainty estimation; understand variance-bias trade-off in resampling.
- ✅ Pass: Compare test error estimates from LOOCV vs 5-fold vs 10-fold CV; implement bootstrap to estimate coefficient standard errors; compare bootstrap SEs to analytic SEs.
- 🛠️ How: `LeaveOneOut`; `KFold`; implement bootstrap loop with `np.random.choice`; fix seeds for reproducibility.

Week 41 — ISLP Ch. 6 (Model Selection & Regularization)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand the motivation for regularization; implement ridge and lasso regression; interpret coefficient shrinkage and sparsity; tune regularization parameter via CV.
- ✅ Pass: Plot ridge and lasso coefficient paths as λ varies; select optimal λ via CV; compare test error of OLS vs ridge vs lasso; explain when lasso produces sparse solutions.
- 🛠️ How: `Ridge`; `Lasso`; `RidgeCV`; `LassoCV`; `StandardScaler` (scale features first); `lasso_path` for path plots.

Week 42 — ISLP Ch. 7 (Beyond Linearity)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Move beyond linearity with polynomial regression, step functions, and splines; understand degrees of freedom; fit GAM-style models.
- ✅ Pass: Fit polynomial, step function, and spline models; compare their flexibility and test errors; produce partial dependence plots; select appropriate number of knots/degrees via CV.
- 🛠️ How: `PolynomialFeatures`; `SplineTransformer`; `pd.cut` for step functions; compare MSE on held-out data.

Week 43 — ISLP Ch. 8 (Tree-Based Methods)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Fit decision trees; understand bagging and the random forest algorithm; implement gradient boosting; interpret tree-based models.
- ✅ Pass: Fit and prune a decision tree; compare single tree vs random forest vs gradient boosting on test error; show OOB error for RF; plot feature importances and partial dependence plots.
- 🛠️ How: `DecisionTreeClassifier/Regressor`; `RandomForestClassifier/Regressor`; `GradientBoostingClassifier/Regressor`; `permutation_importance`; `plot_partial_dependence`.

Week 44 — ISLP Ch. 9 (Support Vector Machines)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Understand maximal margin classifiers and support vectors; fit SVMs with linear and non-linear kernels; tune hyperparameters (C, gamma).
- ✅ Pass: Fit SVM with linear and RBF kernels; tune C and gamma via grid search with CV; visualize decision boundaries on 2D data; identify and highlight support vectors; compare to logistic regression.
- 🛠️ How: `SVC`; `GridSearchCV`; `plt.contourf` for decision boundaries; access `support_vectors_` attribute.

Week 45 — ISLP Ch. 10 (Unsupervised Learning)
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Perform dimensionality reduction with PCA; apply k-means and hierarchical clustering; understand the importance of scaling; evaluate clustering quality.
- ✅ Pass: Apply PCA and plot cumulative explained variance; choose number of components; cluster with k-means (elbow method for k) and hierarchical clustering (dendrogram); evaluate with silhouette score and compare cluster stability across random seeds.
- 🛠️ How: `StandardScaler` (always scale first); `PCA`; `KMeans` with inertia plots; `AgglomerativeClustering`; `dendrogram`; `silhouette_score`.

Week 46 — ISLP Labs/Wrap-up
- 📖 Activities: [ISLP](https://www.statlearning.com/)
- 🧪 Practice: Complete an end-to-end ML project using techniques from all ISLP chapters: EDA, preprocessing, model selection, hyperparameter tuning, evaluation, and interpretation.
- ✅ Pass: Deliver a reproducible notebook with proper train/test split, cross-validation, model comparison, hyperparameter tuning, error analysis, and a 1-page summary documenting decisions, limitations, and risks.
- 🛠️ How: `Pipeline`; `ColumnTransformer` for mixed feature types; `GridSearchCV`/`RandomizedSearchCV`; fixed `random_state` throughout; clean documentation.
</details>

🔁 Flex — Validation basics consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 6 · Classical ML — Weeks 47–65 (Complete PRML, Interpretable ML)</b></summary>

Weeks 47–60 — PRML (Ch. 1–13 + review)
- 📖 [PRML (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- 🧪 Practice: Implement core algorithms from each chapter from scratch: probability distributions, linear models, neural networks, kernel methods, graphical models, mixture models, EM algorithm, approximate inference, and sampling methods.
- ✅ Pass (weekly): Implement the chapter's focal algorithm from scratch; verify correctness by comparing to sklearn/scipy baseline (within 2-5% accuracy); document mathematical derivations; use fixed seeds for reproducibility.
- 🛠️ How: Use NumPy for implementations; sklearn only as verification oracle; work on toy datasets; keep detailed notes linking code to book equations.

Weeks 61–65 — Interpretable ML (complete)
- 📖 [Interpretable ML](https://christophm.github.io/interpretable-ml-book/)
- 🧪 Practice: Apply model-agnostic interpretation methods: PDP, ICE, permutation importance, LIME, SHAP; understand intrinsically interpretable models; explore feature interaction methods.
- ✅ Pass (weekly): For a trained model, produce PDP/ICE plots for top features; compute permutation importance; generate SHAP values for individual predictions; write a 1-page analysis comparing methods' stability across 3 bootstrap resamples.
- 🛠️ How: `sklearn.inspection.PartialDependenceDisplay`; `permutation_importance`; `shap.Explainer`; compare explanations across train/test sets.
</details>

🔁 Flex — Validation & interpretation synthesis

---------------------------------------------------------------------

<details>
<summary><b>Phase 7 · Data Mining — Weeks 66–74 (Complete DM 3e)</b></summary>

Weeks 66–74 — Data Mining 3e (Ch. 1–12)
- 📖 [Data Mining 3e (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- 🧪 Practice: Per-chapter algorithmic work strictly matching the chapter (e.g., data preprocessing tasks; Apriori/FP-Growth; decision trees; k-means/DBSCAN; outlier detection)
- ✅ Pass (weekly): Implement a minimal working version for the chapter’s focal algorithm OR replicate results using a library; verify correctness on a deterministic toy and compare performance on a small real dataset.
- 🛠️ How: construct small synthetic datasets with known ground truth (fixed seeds); assert counts/clusters/rules match expectation.
</details>

🔁 Flex — Mining recap

---------------------------------------------------------------------

<details>
<summary><b>Phase 8 · Econometrics & Time Series — Weeks 75–96 (Complete Gujarati, Lütkepohl)</b></summary>

Weeks 75–86 — Basic Econometrics (complete)
- 📖 [Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf)
- 🧪 Practice: Reproduce a worked example per chapter using methods from that chapter only (OLS basics; classical assumption diagnostics; heteroskedasticity/autocorrelation remedies; functional form; limited dependent variables as presented)
- ✅ Pass (weekly): Match the textbook example’s coefficients and standard errors (within rounding) and include one robustness check discussed in that chapter (e.g., robust/HAC SEs when appropriate).
- 🛠️ How: `statsmodels` OLS/GLM, `cov_type="HC3"` or HAC if the chapter addresses it; include diagnostic plots taught there.

Weeks 87–96 — Lütkepohl (complete)
- 📖 [Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)
- 🧪 Practice: Implement multivariate time series analysis: VAR model specification, estimation, lag order selection, stability analysis, impulse response functions, forecast error variance decomposition, and cointegration/VECM.
- ✅ Pass (weekly): Fit VAR/VECM to macroeconomic data; select lag order using information criteria; verify stability (roots inside unit circle); compute and plot IRFs with confidence bands; perform Johansen cointegration test when applicable.
- 🛠️ How: `statsmodels.tsa.api.VAR`; `statsmodels.tsa.vector_ar.vecm.VECM`; `irf()` for impulse responses; rolling-window forecasts for evaluation.
</details>

🔁 Flex — Econometrics/time-series consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 9 · R for Data Science — Weeks 97–106 (Complete R4DS 2e)</b></summary>

Weeks 97–106 — R4DS (Complete)
- 📖 [R for Data Science (2e)](https://r4ds.hadley.nz)
- 🧪 Practice: Learn R and tidyverse progressively: data import, tidying (pivot_longer/wider), transformation (dplyr verbs), visualization (ggplot2), strings, factors, dates, functions, iteration, and communication (Quarto/RMarkdown).
- ✅ Pass (weekly): Complete a mini-analysis using only functions from chapters covered that week; produce a Quarto/RMarkdown report that renders end-to-end; include at least one visualization and one summary table.
- 🛠️ How: `library(tidyverse)`; `read_csv`; `dplyr` verbs (`filter`, `mutate`, `summarize`, `group_by`); `ggplot2`; `set.seed()` for reproducibility.
</details>

🔁 Flex — R consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 10 · Web Scraping & SQL — Weeks 107–112 (Complete BeautifulSoup, Selenium, SQL)</b></summary>

Week 107 — BeautifulSoup
- 📖 [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- 🧪 Practice: Scrape static HTML pages: fetch with requests, parse with BeautifulSoup, navigate the DOM, extract data using CSS selectors and tag methods.
- ✅ Pass: Scrape a static website and extract structured data; save as CSV/JSON with documented schema; check robots.txt before scraping; implement polite delays to avoid rate limiting (no HTTP 429 errors).
- 🛠️ How: `requests.get(url)`; `BeautifulSoup(html, "lxml")`; `.select()` for CSS selectors; `.find_all()` for tag-based search; `time.sleep()` between requests.

Weeks 108–109 — Selenium
- 📖 [Selenium (Python)](https://selenium-python.readthedocs.io/index.html)
- 🧪 Practice: Automate browser interactions for dynamic websites: handle JavaScript-rendered content, implement explicit waits, manage pagination and infinite scroll, fill forms.
- ✅ Pass (weekly): Scrape a JavaScript-rendered page (e.g., infinite scroll or content behind clicks); implement proper waits and error handling; save timestamped data with retry/timeout logs; handle at least one failure scenario gracefully.
- 🛠️ How: `webdriver.Chrome()`; `WebDriverWait` with `expected_conditions`; CSS/XPath selectors; `execute_script()` for scrolling; consistent viewport settings.

Weeks 110–112 — SQL Tutorial
- 📖 [SQL Tutorial](https://www.sqltutorial.org/)
- 🧪 Practice: Core SELECT/WHERE/JOIN; then subqueries/aggregations; then windows/CTEs (in tutorial order)
- ✅ Pass (weekly): Execute ≥20 queries aligned to the week’s tutorial sections; final week includes a small analytics schema and ≥10 window/CTE queries.
- 🛠️ How: SQLite/Postgres with seeded sample DB; save each query with expected rowcount.
</details>

🔁 Flex — ETL mini-project

---------------------------------------------------------------------

<details>
<summary><b>Phase 11 · Deep Learning — Weeks 113–132 (Complete D2L fundamentals, Goodfellow DL)</b></summary>

Weeks 113–120 — D2L (Fundamentals)
- 📖 [D2L](https://d2l.ai)
- 🧪 Practice: Topic-specific small models exactly as covered (MLP, CNN, RNN; optimization; regularization; data pipelines)
- ✅ Pass (weekly): Train the chapter’s model variant on a toy dataset with fixed seeds and one controlled ablation (optimizer OR regularization) taught in D2L; log curves/metrics.
- 🛠️ How: Follow D2L’s PyTorch/MXNet examples; fix seeds; keep experiments minimal and reproducible.

Week 121 — The Illustrated Transformer (Bridge)
- 📖 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- 🧪 Practice: Understand the Transformer architecture: self-attention mechanism, multi-head attention, positional encoding, encoder-decoder structure.
- ✅ Pass: Implement self-attention from scratch; verify tensor shapes at each step; implement attention masking; write unit tests for: (1) output shape correctness, (2) masked positions get zero attention, (3) attention weights sum to 1.
- 🛠️ How: Use NumPy or PyTorch; implement Q, K, V projections; scaled dot-product attention; verify with `assert` statements and test cases.

Weeks 122–132 — Deep Learning Book (Complete)
- 📖 [Deep Learning Book](https://www.deeplearningbook.org/)
- 🧪 Practice: For each chapter, run a small experiment that demonstrates the chapter’s key concept using building blocks learned in D2L
- ✅ Pass (weekly): Provide a controlled comparison or demonstration plot showing the expected qualitative effect (e.g., different inits, L2 vs dropout, step-size schedules).
- 🛠️ How: Small synthetic or standard toy datasets; fixed seeds; log and compare curves cleanly.
</details>

🔁 Flex — DL recap + tracked mini project

---------------------------------------------------------------------

<details>
<summary><b>Phase 12 · MLOps & Data Engineering — Weeks 133–156 (Complete Zoomcamps, ML Systems)</b></summary>

Weeks 133–140 — MLOps Zoomcamp
- 📖 [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- 🧪 Practice: Module-by-module implementation as taught (tracking, packaging, CI, serving, orchestration, monitoring)
- ✅ Pass (weekly): A runnable local pipeline from clean state to served endpoint with tests passing for that week’s scope.
- 🛠️ How: Docker/Compose; MLflow/W&B; `pytest`; minimal infra defined as per module.

Weeks 141–148 — Machine Learning Systems
- 📖 [ML Systems](https://mlsysbook.ai)
- 🧪 Practice: Write/extend a system design doc each week focusing only on that week’s concepts (SLA/SLOs; rollout/rollback; monitoring; data contracts; cost/reliability)
- ✅ Pass (weekly): The doc includes concrete metrics, failure scenarios, and operational procedures aligned to the chapter.
- 🛠️ How: ADR template; simple diagrams-as-code optional (e.g., Mermaid).

Weeks 149–156 — Data Engineering Zoomcamp
- 📖 [DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
- 🧪 Practice: Module-by-module pipeline work (ingestion, storage, batch/stream, orchestration, analytics eng, testing) as taught in the course
- ✅ Pass (weekly): Re-deployable pipeline from scratch with idempotent runs for that module’s scope.
- 🛠️ How: Terraform/Docker where required, dbt, Airflow/Prefect according to the module.
</details>

🔁 Flex — Ops/engineering consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 13 · LLMs & Agents — Weeks 157–160 (Complete HF Course + Agents)</b></summary>

Weeks 157–159 — Hugging Face Course
- 📖 [HF Course](https://huggingface.co/course/chapter1)
- 🧪 Practice: Learn the Hugging Face ecosystem: load and preprocess datasets, understand tokenizers, fine-tune pretrained models, run inference, evaluate with appropriate metrics.
- ✅ Pass (weekly): Complete the course exercises for that week's chapters; fine-tune a small transformer on a downstream task (e.g., text classification, NER); evaluate with task-appropriate metrics (accuracy, F1, etc.); log all configurations.
- 🛠️ How: `transformers` library for models; `datasets` for data loading; `Trainer` API for fine-tuning; `accelerate` for distributed training; Weights & Biases or TensorBoard for logging.

Week 160 — HF Agents
- 📖 [HF Agents](https://huggingface.co/learn/agents-course/unit0/introduction)
- 🧪 Practice: Build AI agents that use tools: understand agent architectures, implement tool calling, handle errors and timeouts, implement safety guardrails.
- ✅ Pass: Build an agent that completes a multi-step task using external tools; implement proper timeout handling; test with an injected failure scenario and verify graceful degradation; document safety checks and limitations.
- 🛠️ How: Use Hugging Face agents framework; implement `Tool` classes; set timeouts with `asyncio.timeout` or similar; log all tool calls and responses; implement input validation.
</details>

---------------------------------------------------------------------

<details>
<summary><b>Phase 14 · Consolidation, Capstone, Portfolio — Weeks 161–164</b></summary>

Week 161 — statsmodels deep dive
- 📖 [statsmodels](https://www.statsmodels.org/stable/index.html)
- 🧪 Practice: Master statsmodels by reproducing analyses from earlier phases: OLS with diagnostics, GLMs, time series models (ARIMA, VAR), hypothesis testing.
- ✅ Pass: Reproduce two econometric analyses matching original coefficients and standard errors; include full diagnostic suite (heteroskedasticity, autocorrelation tests); apply robust SEs where violations exist.
- 🛠️ How: `statsmodels.api.OLS/GLM`; `statsmodels.tsa` for time series; `het_breuschpagan`, `acorr_ljungbox` for diagnostics; `cov_type="HC3"` for robust SEs.

Week 162 — scikit-learn deep dive
- 📖 [scikit-learn](https://scikit-learn.org/stable/index.html)
- 🧪 Practice: Create a production-ready ML pipeline template: preprocessing (scaling, encoding), feature selection, model training with CV, hyperparameter tuning, probability calibration.
- ✅ Pass: Build a complete Pipeline with ColumnTransformer for mixed types; implement nested CV for unbiased evaluation; apply probability calibration (Platt scaling or isotonic); ensure deterministic results with fixed seeds.
- 🛠️ How: `Pipeline`; `ColumnTransformer`; `GridSearchCV`/`RandomizedSearchCV`; `CalibratedClassifierCV`; fixed `random_state` throughout.

Weeks 163–164 — Capstone & Portfolio
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
- Think Bayes — Weeks 25–32 — [Think Bayes](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple)
- Flexible Imputation of Missing Data — Weeks 33–36 — [FIMD](https://stefvanbuuren.name/fimd/)
- ISLP (Statistical Learning with Python) — Weeks 37–46 — [ISLP](https://www.statlearning.com/)
- PRML (Bishop) — Weeks 47–60 — [PRML (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- Interpretable Machine Learning — Weeks 61–65 — [Interpretable ML](https://christophm.github.io/interpretable-ml-book/)
- Data Mining: Concepts and Techniques (3e) — Weeks 66–74 — [Data Mining 3e (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- Basic Econometrics (Gujarati) — Weeks 75–86 — [Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf)
- New Introduction to Multiple Time Series (Lütkepohl) — Weeks 87–96 — [Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)
- R for Data Science (2e) — Weeks 97–106 — [R for Data Science (2e)](https://r4ds.hadley.nz)
- Beautiful Soup — Week 107 — [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- Selenium (Python) — Weeks 108–109 — [Selenium (Python)](https://selenium-python.readthedocs.io/index.html)
- SQL Tutorial — Weeks 110–112 — [SQL Tutorial](https://www.sqltutorial.org/)
- Dive into Deep Learning — Weeks 113–120 — [D2L](https://d2l.ai)
- The Illustrated Transformer — Week 121 — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Deep Learning — Weeks 122–132 — [Deep Learning Book](https://www.deeplearningbook.org/)
- MLOps Zoomcamp — Weeks 133–140 — [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- Machine Learning Systems — Weeks 141–148 — [ML Systems](https://mlsysbook.ai)
- Data Engineering Zoomcamp — Weeks 149–156 — [DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
- HF Course + HF Agents — Weeks 157–160 — [HF Course](https://huggingface.co/course/chapter1), [HF Agents](https://huggingface.co/learn/agents-course/unit0/introduction)

Notes
- Keep work in any format; seed randomness for reproducibility.
- Use Flex Weeks to finish pass items, review tricky parts, and add spaced-repetition cards (optional).
- Back to top ↑
