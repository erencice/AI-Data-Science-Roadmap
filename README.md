# AI & Data Science Weekly Plan — Activities, Practice, and Pass Criteria

![Duration](https://img.shields.io/badge/duration-~154_weeks-6f42c1)
![Pace](https://img.shields.io/badge/pace-8–10_h%2Fweek-0e8a16)
![Path](https://img.shields.io/badge/path-beginner%E2%86%92practitioner-0366d6)
![Style](https://img.shields.io/badge/style-cumulative%2C_concept%E2%86%92practice-555)

Zero prior knowledge is assumed. Learning order is strictly top-to-bottom. Each week includes a clear “Pass” requirement aligned to the primary resource.

— Quick jump —
- Phase 1 · Data Analysis Foundations
- Phase 2 · Mathematics for ML
- Phase 3 · Statistics Fundamentals
- Phase 4 · Bayesian Statistics & Missing Data
- Phase 5 · Classical ML
- Phase 6 · Data Mining
- Phase 7 · Econometrics & Time Series
- Phase 8 · R for Data Science
- Phase 9 · Web Scraping & SQL
- Phase 10 · Deep Learning
- Phase 11 · MLOps & Data Engineering
- Phase 12 · LLMs & Open-Source AI
- Phase 13 · Consolidation & Capstone

Legend
- 📖 Activities (primary source)
- 🧪 Practice (small tasks)
- ✅ Pass (weekly pass criterion)
- 🛠️ How (implementation hint)
- 🔁 Flex (catch-up, spaced review)

Duration and pacing
- Duration: ~154 weeks (≈3.0 years), 8–10 h/week
- Weekly output: small practical tasks only
- Frequent Flex Weeks between phases for consolidation

Main resources (cover-to-cover completion)
- Python for Data Analysis — Wes McKinney — [Python for Data Analysis](https://wesmckinney.com/book/)
- Mathematics for Machine Learning — Deisenroth, Faisal, Ong — [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- Think Stats — Allen B. Downey — [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- Think Bayes — Allen B. Downey — [Think Bayes](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple)
- Flexible Imputation of Missing Data — van Buuren — [FIMD](https://stefvanbuuren.name/fimd/)
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
- 🧪 Practice: Read CSV; Series/DataFrame basics; indexing; simple plots (exactly from Ch.1–2 topics)
- ✅ Pass: One notebook that loads a CSV, uses `.head()/.info()`, selects columns via `.loc/.iloc`, filters rows, and produces 4 labeled matplotlib/seaborn plots.
- 🛠️ How: `pd.read_csv`, `.loc/.iloc`, boolean masks, `plot.hist()`, `seaborn.countplot`.

Week 2 — P4DA Ch. 3–4
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Aggregation with `groupby`, merges/joins, reshaping with `stack/unstack/pivot`
- ✅ Pass: Build a summary table via `groupby().agg()`, merge it to a second table with `pd.merge`, and reshape it with `pivot_table`. Verify row/column counts at each step.
- 🛠️ How: `groupby().agg({"col":"mean"})`, `pd.merge(left, right, on="key")`, `pd.pivot_table(values, index, columns, aggfunc)`.

Week 3 — P4DA Ch. 5–6
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Text cleanup with `.str` methods; date parsing; dtype fixes
- ✅ Pass: Convert a messy date column to `datetime64[ns]`, standardize a string categorical column (trim/lower), and produce a 10-line data dictionary describing each column and dtype.
- 🛠️ How: `pd.to_datetime(..., errors="coerce")`, `df["col"].str.strip().str.lower()`, `df.astype`.

Week 4 — P4DA Ch. 7–8
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Time series indexing; resampling; rolling/window ops (as introduced in Ch.7–8)
- ✅ Pass: Set a DateTimeIndex, resample to weekly means, and compute a 7-step rolling mean; plot original vs resampled vs rolling mean in one figure.
- 🛠️ How: `df = df.set_index("date")`, `df.resample("W").mean()`, `.rolling(7).mean()`.

Week 5 — P4DA Ch. 9–10
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Categoricals; pivot tables; tidy summaries
- ✅ Pass: Convert a string column to ordered `Categorical` and produce a pivot table summarizing a numeric metric by that category. Justify the order.
- 🛠️ How: `pd.Categorical(df["cat"], categories=[...], ordered=True)`, `pd.pivot_table`.

Week 6 — P4DA Ch. 11–12 (+appendices)
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Functions and reuse; exporting artifacts; light performance care (vectorization where shown)
- ✅ Pass: Turn your EDA steps into small functions at top of a notebook and parameterize the input filepath; saving 1 CSV and 2 plots. Re-run on a second dataset by changing one variable.
- 🛠️ How: Define `load_data(path)`, `clean(df)`, `summarize(df)`; `df.to_csv`, `plt.savefig`.

Week 7 — P4DA Project A
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: End-to-end EDA using only chapters 1–12 capabilities
- ✅ Pass: Apply your parameterized EDA to a new dataset and write a 1-page memo with ≥3 insights, ≥2 hypotheses, and ≥1 data quality risk.
- 🛠️ How: Reuse Week 6 functions; keep code idempotent.

Week 8 — P4DA Project B
- 📖 Activities: [Python for Data Analysis](https://wesmckinney.com/book/)
- 🧪 Practice: Feature engineering strictly from transforms covered (dates, ratios, categories)
- ✅ Pass: Create 5 features (date parts, ratios, interactions limited to arithmetic) and document rationale and potential leakage.
- 🛠️ How: `df.assign(...)`, `pd.to_datetime(...).dt.month`, arithmetic features.
</details>

🔁 Flex — Consolidate EDA template and notes

---------------------------------------------------------------------

<details>
<summary><b>Phase 2 · Mathematics for ML — Weeks 9–18 (Complete MML)</b></summary>

Week 9 — Linear Algebra I
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: Vectors, norms, matrix ops; SVD intro
- ✅ Pass: Compute SVD on a toy matrix and reconstruct it from top-k components; report reconstruction error vs k.
- 🛠️ How: `U,S,Vt = np.linalg.svd(A, full_matrices=False)`; `A_k = U[:,:k] @ np.diag(S[:k]) @ Vt[:k]`.

Week 10 — Linear Algebra II
- 📖 [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- 🧪 Practice: Eigenvalues/vectors; conditioning
- ✅ Pass: Show eigenvector sensitivity by adding small Gaussian noise to a symmetric matrix and plotting angle change vs noise.
- 🛠️ How: `np.linalg.eig`; compute angle via normalized dot product.

Week 11 — Decompositions & Geometry
- 📖 [MML Book (PDF)](...)
- 🧪 Practice: QR vs normal equations for least squares
- ✅ Pass: Solve `min ||Ax-b||` via normal equations and via QR; compare residual norms.
- 🛠️ How: `np.linalg.qr(A)`; backsolve; `np.linalg.lstsq` for reference.

Week 12 — Vector Calculus I
- 📖 [MML Book (PDF)](...)
- 🧪 Practice: Gradients/Jacobians; gradient descent on convex quadratic
- ✅ Pass: Show monotone loss decrease for a suitable step size on `f(x)=1/2 x^T Q x + c^T x`.
- 🛠️ How: analytic gradient `Qx+c`; fixed small step.

Week 13 — Vector Calculus II
- 📖 [MML Book (PDF)](...)
- 🧪 Practice: Chain rule; finite-difference checks
- ✅ Pass: Compare analytic vs central-difference gradient on a 2D function; max abs diff < 1e-4.
- 🛠️ How: central differences with small `h`.

Week 14 — Probability I
- 📖 [MML Book (PDF)](...)
- 🧪 Practice: LLN/CLT simulations using distributions covered
- ✅ Pass: For Binomial and Poisson sample means, show variance ≈ theory and QQ-plots trending more linear as n increases.
- 🛠️ How: simulate many trials; compute sample mean variance; `scipy.stats.probplot` or manual quantiles.

Week 15 — Probability II
- 📖 [MML Book (PDF)](...)
- 🧪 Practice: Covariance; correlation; dependence vs zero-correlation
- ✅ Pass: Generate correlated Normals via Cholesky and recover covariance empirically with small Frobenius error (< 0.05).
- 🛠️ How: `L = cholesky(Sigma)`; `X = Z @ L.T`; `np.cov`.

Week 16 — Optimization I
- 📖 [MML Book (PDF)](...)
- 🧪 Practice: Convexity via Hessian; backtracking line search
- ✅ Pass: Verify convexity by PSD Hessian for two functions and implement backtracking line search on a convex quadratic.
- 🛠️ How: compute Hessian analytically or via finite differences; Armijo condition.

Week 17 — Optimization II
- 📖 [MML Book (PDF)](...)
- 🧪 Practice: Compare first- vs second-order methods introduced in MML
- ✅ Pass: Solve ridge-regularized least squares with Gradient Descent (with backtracking) vs Newton’s method; show iterations-to-tolerance.
- 🛠️ How: add λI to Q; implement Newton step using Hessian; compare convergence curves.

Week 18 — Review
- 📖 [MML Book (PDF)](...)
- 🧪 Practice: Concept map and short-link notes
- ✅ Pass: A one-page map with ≥10 links from math concepts to later ML choices (e.g., regularization ↔ condition number).
- 🛠️ How: diagram or bullet map; keep explicit link statements.
</details>

🔁 Flex — Retrieval practice and summaries

---------------------------------------------------------------------

<details>
<summary><b>Phase 3 · Statistics Fundamentals — Weeks 19–24 (Complete Think Stats)</b></summary>

Week 19 — Think Stats Ch. 1
- 📖 [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- 🧪 Practice: ECDF construction; histogram comparison
- ✅ Pass: Implement ECDF on real data; verify it is non-decreasing and ends at 1.0; compare to histogram shape.
- 🛠️ How: `np.sort`; `np.arange(1,n+1)/n`.

Week 20 — Think Stats Ch. 2
- 📖 [Think Stats (PDF)](...)
- 🧪 Practice: Robust vs classical descriptive stats
- ✅ Pass: Report mean/SD vs median/MAD/trimmed mean on 2 datasets and explain divergence due to skew/outliers.
- 🛠️ How: `scipy.stats.median_abs_deviation`; trimming via slice after sort.

Week 21 — Think Stats Ch. 3–4
- 📖 [Think Stats (PDF)](...)
- 🧪 Practice: Relationships; Pearson vs Spearman
- ✅ Pass: Show an example where Pearson and Spearman diverge and explain monotone non-linear dependence.
- 🛠️ How: `np.corrcoef`; `scipy.stats.spearmanr`.

Week 22 — Think Stats Ch. 5–6
- 📖 [Think Stats (PDF)](...)
- 🧪 Practice: Basic probability; simple Bayesian update
- ✅ Pass: Compute a Beta–Binomial posterior mean/var analytically and confirm via simulation.
- 🛠️ How: closed-form update; simulate posteriors.

Week 23 — Think Stats Ch. 7–8
- 📖 [Think Stats (PDF)](...)
- 🧪 Practice: Hypothesis testing
- ✅ Pass: Simulate empirical Type I ≈ α and produce a power curve for a specified effect size.
- 🛠️ How: repeated sampling; count rejections.

Week 24 — Think Stats Ch. 9–10 (+wrap)
- 📖 [Think Stats (PDF)](...)
- 🧪 Practice: Regression basics; diagnostics
- ✅ Pass: Fit OLS; show residual mean ≈ 0, residual vs fitted plot, and compute VIFs; flag VIF > 10 if any.
- 🛠️ How: `statsmodels.api.OLS`; `variance_inflation_factor`.
</details>

🔁 Flex — Stats recap

---------------------------------------------------------------------

<details>
<summary><b>Phase 4 · Bayesian & Missing Data — Weeks 25–36 (Complete Think Bayes, FIMD)</b></summary>

Weeks 25–32 — Think Bayes (Ch. 1–14, paced)
- 📖 [Think Bayes](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple)
- 🧪 Practice: Conjugates; posterior predictive checks; simple model comparison as presented in the book
- ✅ Pass (weekly): Implement a book-aligned Bayesian model (e.g., Beta–Binomial, Gamma–Poisson, Normal–Normal) with prior sensitivity and a posterior predictive check. For comparison, use the approach discussed in the chapter (e.g., predictive performance or simple Bayes factors where applicable).
- 🛠️ How: analytic posteriors when available; draw PPC replicates and compare a chosen statistic.

Weeks 33–36 — Flexible Imputation of Missing Data (complete)
- 📖 [FIMD](https://stefvanbuuren.name/fimd/)
- 🧪 Practice: Missingness mechanisms; MICE; sensitivity (as in book)
- ✅ Pass (weekly): Run MICE (m≥5) on a dataset; report pooled estimates per Rubin’s rules; compare to complete-case; perform delta-adjustment sensitivity where relevant.
- 🛠️ How: use a MICE implementation (e.g., statsmodels/impyute/sklearn-iterative as proxy) consistent with book procedures.
</details>

🔁 Flex — Consolidate Bayesian + MI

---------------------------------------------------------------------

<details>
<summary><b>Phase 5 · Classical ML — Weeks 37–55 (Complete PRML, Interpretable ML)</b></summary>

Weeks 37–50 — PRML (Ch. 1–13 + review)
- 📖 [PRML (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- 🧪 Practice: Implement chapter-aligned core algorithms using only concepts introduced so far (e.g., logistic regression, linear regression with basis functions, naive Bayes, kernels for regression, EM for GMM, simple graphical model inference)
- ✅ Pass (weekly): From-scratch implementation for that chapter demonstrates parity (within 2–5%) with a library baseline on a small toy dataset; include seeded reproducibility.
- 🛠️ How: use sklearn purely as an oracle for comparison; fix `random_state`; limit to toy-scale experiments.

Weeks 51–55 — Interpretable ML (complete)
- 📖 [Interpretable ML](https://christophm.github.io/interpretable-ml-book/)
- 🧪 Practice: Global (PDP/ICE, permutation) and local (e.g., SHAP) methods as presented
- ✅ Pass (weekly): Apply PDP/ICE and permutation importance; then SHAP to the same model; write a 1-page note on stability and limitations across 3 resamples.
- 🛠️ How: `sklearn.inspection.partial_dependence/plot_partial_dependence` (or newer API), `permutation_importance`, `shap` for local explanations.
</details>

🔁 Flex — Validation & interpretation synthesis

---------------------------------------------------------------------

<details>
<summary><b>Phase 6 · Data Mining — Weeks 56–64 (Complete DM 3e)</b></summary>

Weeks 56–64 — Data Mining 3e (Ch. 1–12)
- 📖 [Data Mining 3e (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- 🧪 Practice: Per-chapter algorithmic work strictly matching the chapter (e.g., data preprocessing tasks; Apriori/FP-Growth; decision trees; k-means/DBSCAN; outlier detection)
- ✅ Pass (weekly): Implement a minimal working version for the chapter’s focal algorithm OR replicate results using a library; verify correctness on a deterministic toy and compare performance on a small real dataset.
- 🛠️ How: construct small synthetic datasets with known ground truth (fixed seeds); assert counts/clusters/rules match expectation.
</details>

🔁 Flex — Mining recap

---------------------------------------------------------------------

<details>
<summary><b>Phase 7 · Econometrics & Time Series — Weeks 65–86 (Complete Gujarati, Lütkepohl)</b></summary>

Weeks 65–76 — Basic Econometrics (complete)
- 📖 [Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf)
- 🧪 Practice: Reproduce a worked example per chapter using methods from that chapter only (OLS basics; classical assumption diagnostics; heteroskedasticity/autocorrelation remedies; functional form; limited dependent variables as presented)
- ✅ Pass (weekly): Match the textbook example’s coefficients and standard errors (within rounding) and include one robustness check discussed in that chapter (e.g., robust/HAC SEs when appropriate).
- 🛠️ How: `statsmodels` OLS/GLM, `cov_type="HC3"` or HAC if the chapter addresses it; include diagnostic plots taught there.

Weeks 77–86 — Lütkepohl (complete)
- 📖 [Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)
- 🧪 Practice: VAR/VECM workflow exactly as the book presents (lag selection, stability checks, IRFs/FEVD, cointegration where applicable)
- ✅ Pass (weekly): Fit VAR/VECM on a macro dataset; pass residual diagnostics; include IRFs/FEVD; perform cointegration tests if required by the chapter.
- 🛠️ How: `statsmodels.tsa.api.VAR/VECM`; rolling-origin evaluation for forecast sections.
</details>

🔁 Flex — Econometrics/time-series consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 8 · R for Data Science — Weeks 87–96 (Complete R4DS 2e)</b></summary>

Weeks 87–96 — R4DS (Complete)
- 📖 [R for Data Science (2e)](https://r4ds.hadley.nz)
- 🧪 Practice: Weekly mini-analyses using only the chapters completed that week (wrangle → visualize → model or summary → render)
- ✅ Pass (weekly): Render a Quarto/Rmd report that re-runs end-to-end with one command, using a seed and only functions introduced in the completed chapters.
- 🛠️ How: `tidyverse` verbs for the covered chapters; `set.seed`; optional `targets/drake` for reproducible workflows.
</details>

🔁 Flex — R consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 9 · Web Scraping & SQL — Weeks 97–102 (Complete BeautifulSoup, Selenium, SQL)</b></summary>

Week 97 — BeautifulSoup
- 📖 [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- 🧪 Practice: Single static page scrape using only requests + bs4 (selectors, parsing, extraction)
- ✅ Pass: Save structured CSV/JSON with documented schema; respect robots.txt; no 429s.
- 🛠️ How: `requests.get`; `BeautifulSoup(html, "lxml")`; `.select` or `.find_all`; `time.sleep` backoff.

Weeks 98–99 — Selenium
- 📖 [Selenium (Python)](https://selenium-python.readthedocs.io/index.html)
- 🧪 Practice: Dynamic page automation as per docs (waits, selectors, pagination/scroll)
- ✅ Pass (weekly): Scrape a dynamic page (e.g., infinite scroll or simple login) and save a timestamped, deterministic snapshot with logs of retries/timeouts.
- 🛠️ How: `WebDriverWait`; CSS/XPath selectors; consistent viewport and user agent.

Weeks 100–102 — SQL Tutorial
- 📖 [SQL Tutorial](https://www.sqltutorial.org/)
- 🧪 Practice: Core SELECT/WHERE/JOIN; then subqueries/aggregations; then windows/CTEs (in tutorial order)
- ✅ Pass (weekly): Execute ≥20 queries aligned to the week’s tutorial sections; final week includes a small analytics schema and ≥10 window/CTE queries.
- 🛠️ How: SQLite/Postgres with seeded sample DB; save each query with expected rowcount.
</details>

🔁 Flex — ETL mini-project

---------------------------------------------------------------------

<details>
<summary><b>Phase 10 · Deep Learning — Weeks 103–122 (Complete D2L fundamentals, Goodfellow DL)</b></summary>

Weeks 103–110 — D2L (Fundamentals)
- 📖 [D2L](https://d2l.ai)
- 🧪 Practice: Topic-specific small models exactly as covered (MLP, CNN, RNN; optimization; regularization; data pipelines)
- ✅ Pass (weekly): Train the chapter’s model variant on a toy dataset with fixed seeds and one controlled ablation (optimizer OR regularization) taught in D2L; log curves/metrics.
- 🛠️ How: Follow D2L’s PyTorch/MXNet examples; fix seeds; keep experiments minimal and reproducible.

Week 111 — The Illustrated Transformer (Bridge)
- 📖 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- 🧪 Practice: Self-attention mechanics (shapes, masks, scaling) as explained in the article
- ✅ Pass: Implement toy self-attention and write unit tests for shape/mask/scaling behavior.
- 🛠️ How: NumPy/PyTorch; `assert` shape checks; verify mask zeros attention to padded tokens.

Weeks 112–122 — Deep Learning Book (Complete)
- 📖 [Deep Learning Book](https://www.deeplearningbook.org/)
- 🧪 Practice: For each chapter, run a small experiment that demonstrates the chapter’s key concept using building blocks learned in D2L
- ✅ Pass (weekly): Provide a controlled comparison or demonstration plot showing the expected qualitative effect (e.g., different inits, L2 vs dropout, step-size schedules).
- 🛠️ How: Small synthetic or standard toy datasets; fixed seeds; log and compare curves cleanly.
</details>

🔁 Flex — DL recap + tracked mini project

---------------------------------------------------------------------

<details>
<summary><b>Phase 11 · MLOps & Data Engineering — Weeks 123–146 (Complete Zoomcamps, ML Systems)</b></summary>

Weeks 123–130 — MLOps Zoomcamp
- 📖 [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- 🧪 Practice: Module-by-module implementation as taught (tracking, packaging, CI, serving, orchestration, monitoring)
- ✅ Pass (weekly): A runnable local pipeline from clean state to served endpoint with tests passing for that week’s scope.
- 🛠️ How: Docker/Compose; MLflow/W&B; `pytest`; minimal infra defined as per module.

Weeks 131–138 — Machine Learning Systems
- 📖 [ML Systems](https://mlsysbook.ai)
- 🧪 Practice: Write/extend a system design doc each week focusing only on that week’s concepts (SLA/SLOs; rollout/rollback; monitoring; data contracts; cost/reliability)
- ✅ Pass (weekly): The doc includes concrete metrics, failure scenarios, and operational procedures aligned to the chapter.
- 🛠️ How: ADR template; simple diagrams-as-code optional (e.g., Mermaid).

Weeks 139–146 — Data Engineering Zoomcamp
- 📖 [DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
- 🧪 Practice: Module-by-module pipeline work (ingestion, storage, batch/stream, orchestration, analytics eng, testing) as taught in the course
- ✅ Pass (weekly): Re-deployable pipeline from scratch with idempotent runs for that module’s scope.
- 🛠️ How: Terraform/Docker where required, dbt, Airflow/Prefect according to the module.
</details>

🔁 Flex — Ops/engineering consolidation

---------------------------------------------------------------------

<details>
<summary><b>Phase 12 · LLMs & Agents — Weeks 147–150 (Complete HF Course + Agents)</b></summary>

Weeks 147–149 — Hugging Face Course
- 📖 [HF Course](https://huggingface.co/course/chapter1)
- 🧪 Practice: Datasets, tokenizers, training, inference as covered by the course units
- ✅ Pass (weekly): Fine-tune or run inference with a small transformer; evaluate with a suitable metric; log configs exactly as the course demonstrates.
- 🛠️ How: `transformers`, `datasets`, `accelerate`; keep to course-recommended scripts.

Week 150 — HF Agents
- 📖 [HF Agents](https://huggingface.co/learn/agents-course/unit0/introduction)
- 🧪 Practice: Tool-using agent with timeouts and error handling as per course
- ✅ Pass: Agent completes a simple multi-step task within timeouts and handles one injected failure path gracefully; list safety checks.
- 🛠️ How: Use course framework; implement guards/timeouts as shown.
</details>

---------------------------------------------------------------------

<details>
<summary><b>Phase 13 · Consolidation, Capstone, Portfolio — Weeks 151–154</b></summary>

Week 151 — statsmodels deep dive
- 📖 [statsmodels](https://www.statsmodels.org/stable/index.html)
- 🧪 Practice: Reproduce two econometric analyses from earlier phases using only covered methods
- ✅ Pass: Match reference coefficients/SEs within tolerance; include robust SEs where applicable.

Week 152 — scikit-learn deep dive
- 📖 [scikit-learn](https://scikit-learn.org/stable/index.html)
- 🧪 Practice: Build a clean template ML pipeline using methods you have already learned (preprocess → CV → metric → calibration if relevant)
- ✅ Pass: Deterministically re-runs and produces calibrated probabilities (if classification).

Weeks 153–154 — Capstone & Portfolio
- 📖 Integrate end-to-end skills only from prior phases
- 🧪 Practice: Capstone with uncertainty quantification, interpretability, evaluation protocol, and non-technical brief
- ✅ Pass: Reproducible project script; README with assumptions/risks; clear results and decisions.
</details>

---------------------------------------------------------------------

Resource-to-Week Completion Map (cover-to-cover)
- Python for Data Analysis — Weeks 1–8 — [Python for Data Analysis](https://wesmckinney.com/book/)
- Mathematics for Machine Learning — Weeks 9–18 — [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- Think Stats — Weeks 19–24 — [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf)
- Think Bayes — Weeks 25–32 — [Think Bayes](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple)
- Flexible Imputation of Missing Data — Weeks 33–36 — [FIMD](https://stefvanbuuren.name/fimd/)
- PRML (Bishop) — Weeks 37–50 — [PRML (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- Interpretable Machine Learning — Weeks 51–55 — [Interpretable ML](https://christophm.github.io/interpretable-ml-book/)
- Data Mining: Concepts and Techniques (3e) — Weeks 56–64 — [Data Mining 3e (PDF)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- Basic Econometrics (Gujarati) — Weeks 65–76 — [Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf)
- New Introduction to Multiple Time Series (Lütkepohl) — Weeks 77–86 — [Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)
- R for Data Science (2e) — Weeks 87–96 — [R for Data Science (2e)](https://r4ds.hadley.nz)
- Beautiful Soup — Week 97 — [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- Selenium (Python) — Weeks 98–99 — [Selenium (Python)](https://selenium-python.readthedocs.io/index.html)
- SQL Tutorial — Weeks 100–102 — [SQL Tutorial](https://www.sqltutorial.org/)
- Dive into Deep Learning — Weeks 103–110 — [D2L](https://d2l.ai)
- The Illustrated Transformer — Week 111 — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Deep Learning — Weeks 112–122 — [Deep Learning Book](https://www.deeplearningbook.org/)
- MLOps Zoomcamp — Weeks 123–130 — [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- Machine Learning Systems — Weeks 131–138 — [ML Systems](https://mlsysbook.ai)
- Data Engineering Zoomcamp — Weeks 139–146 — [DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
- HF Course + HF Agents — Weeks 147–150 — [HF Course](https://huggingface.co/course/chapter1), [HF Agents](https://huggingface.co/learn/agents-course/unit0/introduction)

Notes
- Keep work in any format; seed randomness for reproducibility.
- Use Flex Weeks to finish pass items, review tricky parts, and add spaced-repetition cards (optional).
- Back to top ↑
