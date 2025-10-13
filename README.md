# 🚀 AI & Data Science Roadmap

A rigorously structured, end-to-end learning path for everyone starting from zero: absolute beginners, career switchers, students, and working professionals studying part‑time. The aim is to graduate as a Data Science Expert grounded in descriptive and inferential statistics, mathematics, econometrics and causal inference, classical ML, deep learning, and production practices.

## Contents
- [How to use this roadmap](#how-to-use-this-roadmap)
- [Pacing model (time assumptions)](#pacing-model-time-assumptions)
- [Competency outcomes (what “Data Science Expert” means)](#competency-outcomes-what-data-science-expert-means)
- [Core Curriculum (Stages 0–10)](#core-curriculum)
  - [Stage 0 — Orientation and Study Setup](#stage-0-orientation-and-study-setup)
  - [Stage 1 — Mathematics for ML and Optimization](#stage-1-mathematics-for-ml-and-optimization)
  - [Stage 2 — Probability and Statistics](#stage-2-probability-and-statistics)
  - [Stage 3 — Programming Fundamentals (Python-first)](#stage-3-programming-fundamentals-python-first)
  - [Stage 4 — EDA and Visualization](#stage-4-eda-and-visualization)
  - [Stage 5 — SQL and Data Modeling](#stage-5-sql-and-data-modeling)
  - [Stage 6 — Data Acquisition: Web Scraping and APIs](#stage-6-data-acquisition-web-scraping-and-apis)
  - [Stage 7 — Econometrics, Causal Inference, and Time Series](#stage-7-econometrics-causal-inference-and-time-series)
  - [Stage 8 — Classical Machine Learning](#stage-8-classical-machine-learning)
  - [Stage 9 — Deep Learning](#stage-9-deep-learning)
  - [Stage 10 — MLOps and Data Engineering](#stage-10-mlops-and-data-engineering)
- [Specialization Tracks (optional)](#specialization-tracks-optional)
  - [NLP and LLMs](#b-nlp-and-llms)
  - [Computer Vision](#c-computer-vision)
  - [Recommender Systems](#d-recommender-systems)
- [Capstone and Portfolio](#capstone-and-portfolio)
- [Appendix](#appendix)

---

## How to use this roadmap

- Follow stages in order; specialize after Stage 10 or alongside it if time allows.
- Each stage includes: Objectives, Place in our goal, Prerequisites, Estimated Effort, Weekly progression (sub‑stages), Essential/Supplementary resources with “Why now”, Practice aligned to what you just learned, Exit Criteria.
- Short, targeted on‑ramps are provided for true beginners. These are practical and scoped to avoid fatigue.
- Evidence-based progress: Treat each week as a study cycle. Produce the listed deliverable, check Exit Criteria, and keep a learning log (what you read, built, measured).

---

## Pacing model (time assumptions)

- For everyone: beginners, switchers, students, and working professionals.
- Estimates are net study hours per week:
  - Light pace: 3–5 h/week
  - Standard pace: 5–7 h/week (recommended)
  - Intensive pace: 8–12 h/week

Methodology for estimates:
- Textbooks/notes ~8–12 pages/hour + similar time for exercises/notes.
- Videos: runtime × 1.4–1.7 (pauses + note‑taking + small practice).
- Docs/tutorials: quickstarts 1–3 h; deeper guides 4–8 h.
- Projects are included where listed (typically 2–6 h).

Tip: If time is tight, complete the weekly Practice (Deliverable) to keep momentum; return to deeper reading next week.

---

## Competency outcomes (what “Data Science Expert” means)

By the end, you will be able to:
- Frame problems, design datasets/pipelines, and select appropriate statistical/ML methods.
- Apply descriptive and inferential statistics correctly, quantify uncertainty, and design credible experiments.
- Use econometrics and causal inference to estimate effects under assumptions; analyze panel and time‑series data with proper diagnostics and backtesting.
- Build and evaluate classical ML and deep learning models; interpret, communicate, and document decisions.
- Acquire data via APIs/scraping responsibly; transform with SQL; manage data quality; visualize and narrate insights.
- Ship reproducible projects with tests, containers, experiment tracking, CI/CD, and basic orchestration.

---

<a id="core-curriculum"></a>
## Core Curriculum (Stages 0–10)

<a id="stage-0-orientation-and-study-setup"></a>
### Stage 0 — Orientation and Study Setup
Estimated effort (total): 6–10 h · Calendar: ~1–2 weeks at 3–7 h/week

- Place in our goal: Establishes foundational habits (version control, environments) that make all subsequent learning reproducible and professional.
- Objectives: Adopt effective study habits; set up dev environment.
- Prerequisites: None

Zero‑to‑One On‑Ramp (optional)
- The Missing Semester (Shell/CLI essentials, 4–6 h skim) — [The Missing Semester](https://missing.csail.mit.edu/) — Why now: Command‑line fluency accelerates all later work.

Weekly progression
- Week 0.1 (3–5 h) — [Using venv](https://docs.python.org/3/library/venv.html) — Why now: Clean, reproducible environments from day one.  
  Practice (Deliverable): Create and activate a venv; freeze requirements; add setup.md.

  
- Week 0.2 (3–5 h) — [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git) — Why now: Version control and collaboration are foundational.  
  Practice (Deliverable): Initialize a repo; commit a template project; open a practice PR and merge it.

Supplementary
- [Poetry](https://python-poetry.org/docs/) or [Conda](https://docs.conda.io/en/latest/) — Optional environment managers.
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) — Project scaffolding.

Exit Criteria
- You can manage Python environments and maintain a clean repository.

---

<a id="stage-1-mathematics-for-ml-and-optimization"></a>
### Stage 1 — Mathematics for ML and Optimization
Estimated effort (total): 62–96 h · Calendar: ~9–16 weeks at 5–7 h/week

- Place in our goal: Mathematical language and optimization tools powering ML/DL training, diagnostics, and interpretation.
- Objectives: Linear algebra, vector calculus, optimization basics; probability primer.
- Prerequisites: None (on‑ramp below if needed)

Zero‑to‑One On‑Ramp (short; pick A or B)
- A) Algebra/Trig quick review (2–6 h) — Complete: Algebra Review Sections 1–5; Trig up to Unit Circle  
  [Paul’s Algebra/Trig Review](https://tutorial.math.lamar.edu/Extras/AlgebraTrigReview/AlgebraTrigIntro.aspx) — Why now: Concise refresh without a full course.
- B) Linear algebra intuition (3–6 h) — Complete: Episodes 1, 2, 3, 4, 6, and 8  
  [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) — Why now: Geometric intuition accelerates learning.

Weekly progression
- Week 1 (8–10 h) — MML Ch. 2 Linear Algebra (vectors, matrices, operations)  
  Resource: [Mathematics for Machine Learning](https://mml-book.github.io/) — Why now: LA underpins ML representations.  
  Practice (Deliverable): Compute dot products, norms, projections; solve 2×2/3×3 Ax=b (by hand and in NumPy); verify residuals < 1e‑8.

  
- Week 2 (6–8 h) — MML Ch. 3 Analytic Geometry (subspaces, orthogonality)  
  Practice (Deliverable): Decompose vectors into parallel/orthogonal components; least‑squares line fit via normal equations; residual plot.

  
- Week 3–4 (10–14 h) — MML Ch. 4 Matrix Decompositions (LU, QR, eigen)  
  Practice (Deliverable): Solve Ax=b with LU; least‑squares via QR; eigen‑decompose a small symmetric matrix; power iteration convergence plot.  
  Reference (no practice dependency): [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — derivatives section.

  
- Week 5 (6–8 h) — MML Ch. 5 Vector Calculus (gradients, Jacobians)  
  Practice (Deliverable): Derive gradients for quadratic forms; verify with numerical gradients; short comparison notes.

  
- Week 6–7 (12–18 h) — Convex Optimization (Boyd & Vandenberghe)  
  Resource: [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) — Complete: Ch. 2, 3, 4, 9 (skip advanced proofs first pass).  
  Practice (Deliverable): Implement GD vs. momentum on convex quadratics; plot convergence; report iterations to tolerance.

  
- Week 8 (6–8 h) — Probability Primer  
  Resource: [OpenIntro Statistics](https://www.openintro.org/book/os/) — Complete: Ch. 3 (3.1–3.5) and Ch. 4 (4.1–4.3).  
  Practice (Deliverable): Simulate Bernoulli/Binomial/Normal; visualize LLN/CLT with code and commentary.

Supplementary
- [Trig Cheat Sheet](https://tutorial.math.lamar.edu/pdf/Trig_Cheat_Sheet.pdf) — Handy reference.
- Think Complexity (2e) — Allen Downey — [Book page](https://greenteapress.com/wp/think-complexity/) · [PDF](https://greenteapress.com/thinkcomplexity2/thinkcomplexity2.pdf)

Exit Criteria
- Comfortable with vectors/matrices, derivatives/gradients, and basic convexity; can solve Ax=b and implement GD with empirical convergence.

---

<a id="stage-2-probability-and-statistics"></a>
### Stage 2 — Probability and Statistics
Estimated effort (total): 70–120 h · Calendar: ~12–21 weeks at 5–7 h/week

- Place in our goal: Descriptive and inferential statistics for valid analysis, experimentation, and uncertainty quantification.
- Objectives: Probability, estimation, hypothesis testing, regression/ANOVA, Bayesian basics, experimental design.
- Prerequisites: Stage 1

Zero‑to‑One On‑Ramp (short)
- StatQuest (6–12 h skim) — Complete: Descriptive, Probability Basics, P‑values/CI  
  [StatQuest](https://www.youtube.com/c/joshstarmer) — Why now: Clear intuition helps formal courses.

Weekly progression
- Week 1–2 (12–16 h) — Descriptive Statistics and Intro  
  Resource: [OpenIntro Statistics](https://www.openintro.org/book/os/) — Ch. 1–2.  
  Practice (Deliverable): Summary report with visuals (central tendency, spread, outliers) using a public dataset.

  
- Week 3–4 (15–25 h) — Probability Theory  
  Resource: [STAT 414](https://online.stat.psu.edu/stat414/) — Lessons 1–10 (core).  
  Practice (Deliverable): Monte Carlo sims (binomial, normal, exponential) with convergence plots and short write‑up.

  
- Week 5–7 (15–25 h) — Mathematical Statistics  
  Resource: [STAT 415](https://online.stat.psu.edu/stat415/) — Lessons 1–9 (estimation/inference).  
  Practice (Deliverable): Bootstrap CI vs. analytical CI on real data; compare coverage.

  
- Week 8–9 (12–16 h) — Regression and ANOVA  
  Resources: [STAT 501](https://online.stat.psu.edu/stat501/) Lessons 1–8; [STAT 502](https://online.stat.psu.edu/stat502/) Lessons 1–6.  
  Practice (Deliverable): Fit OLS; diagnostics (residuals, VIF); one/two‑way ANOVA with effect sizes.

  
- Week 10 (6–10 h) — Bayesian Primer  
  Resource: [Think Bayes (2e)](https://allendowney.github.io/ThinkBayes2/) — Ch. 1–4.  
  Practice (Deliverable): Beta‑Binomial A/B; posterior predictive checks; compare to frequentist test.

  
- Week 11 (6–8 h) — Experimentation  
  Resources: A/B Testing [guide](https://vkteam.medium.com/practitioners-guide-to-statistical-tests-ed2d580ef04f#1e3b), [planning](https://towardsdatascience.com/step-by-step-for-planning-an-a-b-test-ef3c93143c0b).  
  Practice (Deliverable): Power analysis; pre‑registration; mock A/B analysis with a decision memo.

Supplementary
- STAT 484/485 (R): [course pages](https://online.stat.psu.edu/stat484-485/) — Alternative R path.

Exit Criteria
- Can design experiments, analyze results, and interpret regression models with quantified uncertainty.

---

<a id="stage-3-programming-fundamentals-python-first"></a>
### Stage 3 — Programming Fundamentals (Python-first)
Estimated effort (total): 20–36 h · Calendar: ~3–7 weeks at 5–7 h/week

- Place in our goal: Engineering practices for reliable, testable data projects—skills hiring managers expect.
- Objectives: Python fluency, packaging, testing, typing, notebook hygiene, core DS libs.
- Prerequisites: Stage 0–2

Zero‑to‑One On‑Ramp
- Automate the Boring Stuff (10–16 h skim) — Parts I–II  
  [Automate the Boring Stuff](https://automatetheboringstuff.com/)

Weekly progression
- Week 1 (6–10 h) — Python Basics  
  Resource: [Official Tutorial](https://docs.python.org/3/tutorial/) or [Python Crash Course](https://www.youtube.com/watch?v=rfscVS0vtbw).  
  Practice (Deliverable): CSV summary CLI with argparse, logging, and `--help`.

  
- Week 2 (6–10 h) — pandas/numpy  
  Resource: [Python for Data Analysis](https://wesmckinney.com/book/) — Indexing, GroupBy, Reshaping, Time Series.  
  Practice (Deliverable): Reusable cleaning script; benchmark vectorized vs. loops; short results table.

  
- Week 3 (6–10 h) — Packaging/testing/typing  
  Resources: [PEP 8](https://peps.python.org/pep-0008/), [pytest](https://docs.pytest.org/).  
  Practice (Deliverable): Package the cleaning script; add unit tests and type hints; build a wheel.

  
- Week 4 (2–6 h) — Statistical modeling intro  
  Resource: [statsmodels](https://www.statsmodels.org/stable/index.html) — OLS tutorial + GLM overview.  
  Practice (Deliverable): Fit OLS; produce a regression report with assumptions and limitations.

Supplementary
- Kevin Sheppard notes — [Python Notes (PDF)](https://www.kevinsheppard.com/files/teaching/python/notes/python_introduction_2021.pdf)
- Real Python best practices — [Collection](https://realpython.com/tutorials/best-practices/)

Exit Criteria
- Comfortable with pandas, numpy, plotting; can ship a small, tested project.

---

<a id="stage-4-eda-and-visualization"></a>
### Stage 4 — EDA and Visualization
Estimated effort (total): 15–24 h · Calendar: ~2–5 weeks at 5–7 h/week

- Place in our goal: Turn raw data into insight; build communication skills and data quality discipline.
- Objectives: Data cleaning, profiling, visualization, data quality checks.
- Prerequisites: Stage 3

Zero‑to‑One On‑Ramp
- Kaggle Data Cleaning (3–5 h) — [Kaggle Data Cleaning](https://www.kaggle.com/learn/data-cleaning)

Weekly progression
- Week 1 (6–10 h) — Profiling and Cleaning  
  Resources: [ydata‑profiling](https://ydata-profiling.ydata.ai/docs/master/), [Python for Data Analysis](https://wesmckinney.com/book/).  
  Practice (Deliverable): Profiling report; reproducible cleaning notebook + script; checklist of issues.

  
- Week 2 (6–10 h) — Visualization  
  Resources: [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/stable/).  
  Practice (Deliverable): Small EDA dashboard (static/light interactive) with 3–5 key charts.

  
- Week 3 (3–4 h) — Data Quality Tests  
  Resources: [Great Expectations](https://docs.greatexpectations.io/), [missingno](https://github.com/ResidentMario/missingno).  
  Practice (Deliverable): Expectation suites for key tables; CI job to run them.

Supplementary
- [Altair](https://altair-viz.github.io/)
- [Plotly](https://plotly.com/python/)

Exit Criteria
- You can profile, clean, visualize data, and communicate insights clearly.

---

<a id="stage-5-sql-and-data-modeling"></a>
### Stage 5 — SQL and Data Modeling
Estimated effort (total): 15–28 h · Calendar: ~2–5 weeks at 5–7 h/week

- Place in our goal: Efficient data extraction/joins and schema design for reliable analytics.
- Objectives: Querying, joins, window functions, indexes, query plans; basic modeling.
- Prerequisites: Stage 3–4

Zero‑to‑One On‑Ramp
- Khan Academy SQL (4–8 h skim) — [Khan Academy SQL](https://www.khanacademy.org/computing/computer-programming/sql)

Weekly progression
- Week 1 (6–10 h) — SQL Fundamentals  
  Resource: [SQL Tutorial](https://www.sqltutorial.org/).  
  Practice (Deliverable): CRUD + analytical joins/subqueries; include result screenshots.

  
- Week 2 (6–10 h) — Advanced SQL  
  Resources: PostgreSQL [Window Functions](https://www.postgresql.org/docs/current/tutorial-window.html), [EXPLAIN](https://www.postgresql.org/docs/current/using-explain.html).  
  Practice (Deliverable): KPI queries with windows; analyze plans; add indexes and re‑measure.

  
- Week 3 (3–8 h) — Data Modeling  
  Resource: Kimball [overview](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/).  
  Practice (Deliverable): Design a star schema; ER diagram + rationale.

Supplementary
- [SQLBolt](https://sqlbolt.com/), [Mode SQL](https://mode.com/sql-tutorial/), [LeetCode SQL](https://leetcode.com/studyplan/top-sql-50/)
- Sample DBs: [Chinook](https://github.com/lerocha/chinook-database), [Northwind](https://github.com/microsoft/sql-server-samples/tree/master/samples/databases/northwind-pubs)

Exit Criteria
- Optimize queries, use windows/CTEs, design simple analytical schemas.

---

<a id="stage-6-data-acquisition-web-scraping-and-apis"></a>
### Stage 6 — Data Acquisition: Web Scraping and APIs
Estimated effort (total): 18–30 h · Calendar: ~3–6 weeks at 5–7 h/week

- Place in our goal: Reliable, ethical ingestion of external data at scale.
- Objectives: Robust scraping, API consumption, ethics/legal, tooling choice.
- Prerequisites: Stage 3

Zero‑to‑One On‑Ramp
- MDN: [HTTP overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview), [HTML basics](https://developer.mozilla.org/en-US/docs/Learn/Getting_started_with_the_web/HTML_basics)

Weekly progression
- Week 1 (3–6 h) — APIs and HTTP Clients  
  Resources: [Requests](https://requests.readthedocs.io/), [httpx](https://www.python-httpx.org/).  
  Practice (Deliverable): Small API client with pagination/auth/retries; readme with usage.

  
- Week 2 (6–10 h) — Static Scraping  
  Resource: [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/).  
  Practice (Deliverable): Extract structured data; persist to CSV/DB; log failures.

  
- Week 3 (6–10 h) — Dynamic Sites  
  Resources: [Playwright](https://playwright.dev/python/), [Selenium](https://selenium-python.readthedocs.io/).  
  Practice (Deliverable): Headless navigation; capture content behind interactions; robust waits.

  
- Week 4 (3–4 h) — Crawling and Ethics  
  Resource: [Scrapy](https://docs.scrapy.org/en/latest/); Ethics: [robots.txt](https://www.robotstxt.org/), [guide](https://scrapeops.io/python-scrapy-playbook/python-ethical-web-scraping/).  
  Practice (Deliverable): Spider with throttling/backoff; robots checks; output sample.

Supplementary
- Data Mining — [book info](https://www.sciencedirect.com/book/9780123814791/data-mining)

Exit Criteria
- Acquire data responsibly from static/dynamic sources.

---

<a id="stage-7-econometrics-causal-inference-and-time-series"></a>
### Stage 7 — Econometrics, Causal Inference, and Time Series
Estimated effort (total): 36–64 h · Calendar: ~6–12 weeks at 5–7 h/week

- Place in our goal: Move beyond correlation to credible estimation and sound forecasting. Establishes regression assumptions, identification, and temporal modeling with proper validation.
- Objectives: OLS and diagnostics, common violations and remedies, causal identification basics (DAGs, RCTs, confounding, DiD), time‑series fundamentals (stationarity, ARIMA), and backtesting.
- Prerequisites: Stage 2 (Statistics), Stage 3 (Programming)

Weekly progression
- Week 1 (8–12 h) — OLS Foundations and Gauss–Markov  
  Resource: Econometrics with R — [Econometrics with R](https://www.econometrics-with-r.org/) (OLS chapters) — Why now: Open, applied route to core regression concepts.  
  Practice (Deliverable): Fit OLS; residual diagnostics; interpret coefficients and uncertainty.

  
- Week 2 (6–10 h) — Diagnostics, Heteroskedasticity, Multicollinearity, Autocorrelation  
  Resources: Econometrics with R (diagnostics), statsmodels examples — [statsmodels](https://www.statsmodels.org/stable/index.html)  
  Practice (Deliverable): Breusch–Pagan test; White/HC robust SEs; VIF check; Durbin–Watson; apply appropriate remedy and document rationale.

  
- Week 3 (6–10 h) — Causal Inference Basics (Identification, DAGs, Omitted Variable Bias)  
  Resource: Cunningham — The Mixtape (free) — [Causal Inference: The Mixtape](https://mixtape.scunning.com/) — Why now: Modern, accessible causal toolkit.  
  Practice (Deliverable): Simulate confounding; show bias under naive OLS; specify DAG; discuss identification strategy.

  
- Week 4 (6–10 h) — Research Designs: Matching/PS, Difference‑in‑Differences, Fixed Effects  
  Resource: The Mixtape (DiD/FE chapters); optional: R4DS causal chapters or relevant tutorials.  
  Practice (Deliverable): Implement a 2×2 DiD and a panel FE model on a public dataset; assumption checks; effect interpretation.

  
- Week 5 (6–10 h) — Time Series Fundamentals (Decomposition, Stationarity, ACF/PACF)  
  Resources: FPP3 (free) — [FPP3](https://otexts.com/fpp3/); Python version — [Forecasting: The Pythonic Way](https://otexts.com/fpppy/) — Why now: Modern forecasting curriculum.  
  Practice (Deliverable): STL decomposition; unit‑root test (ADF); seasonal strength; write diagnostic notes.

  
- Week 6 (4–12 h) — ARIMA/SARIMA and Backtesting  
  Resources: statsmodels.tsa — [statsmodels.tsa](https://www.statsmodels.org/stable/tsa.html)  
  Practice (Deliverable): Fit ARIMA/SARIMA; rolling‑origin backtest; report MAE/MAPE; forecast with intervals; document failure modes.

Supplementary
- Gujarati & Porter — Basic Econometrics (reference) — [Publisher](https://www.mheducation.com/highered/product/basic-econometrics-gujarati-porter/M9780073375779.html)
- Wooldridge — Introductory Econometrics (reference) — [Cengage page](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge/)
- Lütkepohl — Multiple Time Series (advanced VAR/state space) — [Springer](https://link.springer.com/book/10.1007/978-3-540-27752-1)

Exit Criteria
- Diagnose and remedy OLS assumption violations; articulate identification assumptions; implement DiD/FE; build and evaluate ARIMA forecasts with rolling backtests.

---

<a id="stage-8-classical-machine-learning"></a>
### Stage 8 — Classical Machine Learning
Estimated effort (total): 30–50 h · Calendar: ~5–9 weeks at 5–7 h/week

- Place in our goal: Baseline modeling toolbox and evaluation mindset across domains.
- Objectives: Supervised/unsupervised basics, pipelines, validation, metrics, interpretation.
- Prerequisites: Stage 1–2–3–4

Zero‑to‑One On‑Ramp
- Kaggle Intro to ML (4–6 h) — [course](https://www.kaggle.com/learn/intro-to-machine-learning)  
- StatQuest ML (6–12 h skim) — [playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)

Weekly progression
- Week 1 (12–18 h) — Supervised Learning  
  Resource: [scikit‑learn](https://scikit-learn.org/stable/) — Pipelines, preprocessing, linear/logistic/tree/ensembles.  
  Practice (Deliverable): Full pipeline with ColumnTransformer + CV; baseline and tuned models; model card.

  
- Week 2 (6–10 h) — Model Selection and Validation  
  Resource: scikit‑learn CV/metrics.  
  Practice (Deliverable): Nested CV vs. holdout; report metric variance and uncertainty.

  
- Week 3 (5–8 h) — Interpretability  
  Resource: [Interpretable ML](https://christophm.github.io/interpretable-ml-book/).  
  Practice (Deliverable): Permutation importance, PDP/ICE, SHAP; interpretation notes and caveats.

  
- Week 4 (3–6 h) — Missing Data  
  Resource: [FIMD](https://stefvanbuuren.name/fimd/).  
  Practice (Deliverable): Compare simple imputations vs. MICE; downstream performance and bias discussion.

  
- Week 5 (4–8 h) — Unsupervised Basics (incl. PCA)  
  Resource: scikit‑learn clustering/dimensionality reduction.  
  Practice (Deliverable): PCA explained variance; customer clustering; silhouette score; UMAP visualization.  
  Optional tie‑in: MML Ch. 10 “Principal Component Analysis” for deeper LA derivations.

Essential text
- ISLR/ISLRv2 — Complete: Ch. 2–6 (core); skim Ch. 8 (trees) and Ch. 10 (unsupervised)  
  [Introduction to Statistical Learning](https://www.statlearning.com/)

Supplementary
- [mlcourse.ai](https://mlcourse.ai/book/index.html), [SHAP](https://shap.readthedocs.io/en/latest/)

Exit Criteria
- Ship a reproducible ML pipeline with meaningful evaluation and documented decisions.

---

<a id="stage-9-deep-learning"></a>
### Stage 9 — Deep Learning
Estimated effort (total): 40–70 h · Calendar: ~7–11 weeks at 5–7 h/week

- Place in our goal: Modern neural architectures and training practices for vision/NLP and beyond.
- Objectives: Neural nets, CNN/RNN basics, modern training, transfer learning, Transformers.
- Prerequisites: Stage 1–2–3–8

Zero‑to‑One On‑Ramp
- fast.ai (audit 6–10 h skim) — [course](https://course.fast.ai/) — first 3 lessons  
- Kaggle Intro to DL (3–5 h) — [course](https://www.kaggle.com/learn/intro-to-deep-learning)

Weekly progression
- Week 1 (12–18 h) — DL Fundamentals  
  Resources: [D2L](https://d2l.ai/) Ch. 2–6; [PyTorch Tutorials](https://pytorch.org/tutorials/) “Learn the Basics”.  
  Practice (Deliverable): Implement an MLP; add regularization/schedulers; track metrics in a table.

  
- Week 2 (8–12 h) — CNN Training  
  Practice (Deliverable): Train a CNN on CIFAR‑10; experiment with augmentation and mixup/cutmix; compare runs with logged metrics.

  
- Week 3 (6–8 h) — Sequence Models  
  Practice (Deliverable): LSTM baseline on a sequence dataset; compare to a classical baseline; error analysis.

  
- Week 4 (6–8 h) — Transformers Intro  
  Resource: [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).  
  Practice (Deliverable): Fine‑tune a small Transformer on text classification; evaluate; save artifacts.

  
- Week 5 (8–14 h) — Transfer Learning Project  
  Practice (Deliverable): End‑to‑end project with dataset prep, training, evaluation, and a demo notebook/app.

Supplementary
- [Deep Learning (Goodfellow et al.)](https://www.deeplearningbook.org/), [CS231n](http://cs231n.stanford.edu/), [fast.ai](https://course.fast.ai/)

Exit Criteria
- Train, debug, and deploy a DL model with solid metrics and documentation.

---

<a id="stage-10-mlops-and-data-engineering"></a>
### Stage 10 — MLOps and Data Engineering
Estimated effort (total): 40–60 h · Calendar: ~6–10 weeks at 5–7 h/week

- Place in our goal: Take projects to production with reproducibility, automation, and scalable data pipelines.
- Objectives: Experiment tracking, model/data versioning, containers, CI/CD, orchestration; batch/stream pipelines, warehouses, transformations, Spark.
- Prerequisites: Stage 8–9

Zero‑to‑One On‑Ramp
- Docker 101 (2–4 h) — [tutorial](https://www.docker.com/101-tutorial/), GH Actions Quickstart (1–2 h) — [guide](https://docs.github.com/en/actions/quickstart)

Weekly progression
- Week 1 (6–8 h) — Experiment Tracking  
  Resource: [MLflow](https://mlflow.org/) — Tracking + Models + Registry.  
  Practice (Deliverable): Track runs/artifacts; compare experiments; promote best model to registry.

  
- Week 2 (6–8 h) — Data/Model Versioning  
  Resource: [DVC](https://dvc.org/) — Get Started + Pipelines + Remote.  
  Practice (Deliverable): Version datasets; create pipelines; reproduce results end‑to‑end.

  
- Week 3 (6–8 h) — Containerization  
  Resource: [Docker – Get Started](https://docs.docker.com/get-started/) — best practices.  
  Practice (Deliverable): Containerize your ML project; validate locally and in CI.

  
- Week 4 (5–8 h) — CI/CD and Orchestration  
  Resources: GH Actions; [Airflow](https://airflow.apache.org/) / [Prefect](https://docs.prefect.io/); [Great Expectations](https://docs.greatexpectations.io/).  
  Practice (Deliverable): Weekly batch job with data checks and model refresh; passing CI.

  
- Week 5 (8–12 h) — Warehousing and Transformations  
  Resource: [dbt Fundamentals](https://docs.getdbt.com/docs/get-started-dbt).  
  Practice (Deliverable): Staging/model layer with tests/docs in dbt; exposures.

  
- Week 6 (8–12 h) — Spark and Streaming  
  Resource: Data Engineering Zoomcamp (Spark, Kafka) — [DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)  
  Practice (Deliverable): Spark ETL job; benchmark vs. pandas; simple Kafka ingestion; checkpointing.

Supplementary
- [Machine Learning Systems](https://mlsysbook.ai/)
- [Delta Lake](https://delta.io/), [Apache Iceberg](https://iceberg.apache.org/) — Lakehouse patterns.

Exit Criteria
- From notebook to reproducible, testable, containerized service with automated data/ML pipelines.

---

## Specialization Tracks (optional)

> Optional after Stage 10 or in parallel where relevant.

<a id="b-nlp-and-llms"></a>
### B) NLP and LLMs
Estimated effort (total): 24–40 h

Zero‑to‑One On‑Ramp
- spaCy Course (3–6 h) — [course](https://course.spacy.io/en/)

Weekly progression
- Week B1 (10–16 h) — Transformers Fundamentals  
  Resource: [Hugging Face Course](https://huggingface.co/learn/nlp-course/chapter1) — Ch. 1–4.  
  Practice (Deliverable): Fine‑tune a text classifier; track metrics; export artifacts.

  
- Week B2 (6–10 h) — RAG Systems  
  Resources: [LangChain](https://python.langchain.com/), [LlamaIndex](https://docs.llamaindex.ai/).  
  Practice (Deliverable): RAG app on your docs; evaluation with ragas; latency/quality trade‑offs.

  
- Week B3 (6–10 h) — Evaluation and Safety  
  Resources: [lm‑eval‑harness](https://github.com/EleutherAI/lm-eval-harness), [ragas](https://github.com/explodinggradients/ragas), [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework).  
  Practice (Deliverable): Build an evaluation suite; document safety mitigations.

  
- Week B4 (2–4 h, optional) — Agents  
  Resource: [HF Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction).  
  Practice (Deliverable): Prototype a simple agent with a constrained toolset.

Exit Criteria
- End‑to‑end RAG with evaluation and basic safety.

---

<a id="c-computer-vision"></a>
### C) Computer Vision
Estimated effort (total): 24–40 h

Zero‑to‑One On‑Ramp
- PyTorch 60‑min Blitz (2–4 h) — [tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

Weekly progression
- Week C1 (10–16 h) — Vision Foundations  
  Resource: [torchvision tutorials](https://pytorch.org/vision/stable/index.html#tutorials).  
  Practice (Deliverable): Train a ResNet with augmentations; evaluate with confusion matrix/AUC.

  
- Week C2 (6–10 h) — Practical DL  
  Resources: [fastai vision](https://docs.fast.ai/vision.learner.html), [course](https://course.fast.ai/).  
  Practice (Deliverable): Prototype multiple architectures; compare results.

  
- Week C3 (4–8 h, optional) — Theory  
  Resource: [CS231n](http://cs231n.stanford.edu/).  
  Practice (Deliverable): Custom augmentation/evaluation protocol and brief report.

Exit Criteria
- Fine‑tuned vision model with clear evaluation.

---

<a id="d-recommender-systems"></a>
### D) Recommender Systems
Estimated effort (total): 18–30 h

Zero‑to‑One On‑Ramp
- Recsys basics (2–4 h) — [Google Developers](https://developers.google.com/machine-learning/recommendation/collaborative/basics)

Weekly progression
- Week D1 (6–10 h) — MF and Implicit Feedback  
  Resource: [implicit](https://github.com/benfred/implicit).  
  Practice (Deliverable): Train ALS/BPR on interactions; tune hyperparams; offline metrics.

  
- Week D2 (6–10 h) — Pipelines and Evaluation  
  Resource: [Microsoft Recommenders](https://github.com/microsoft/recommenders).  
  Practice (Deliverable): Offline eval pipeline; MAP/NDCG/Recall@k; ablations.

  
- Week D3 (6–10 h) — Ranking Metrics  
  Resource: [Metrics overview (PDF)](https://cmci.colorado.edu/classes/INFO-4604/files/rec_sys_metrics.pdf).  
  Practice (Deliverable): Compare candidate generators/rankers with proper ranking metrics.

Exit Criteria
- Top‑N recommender with offline eval and simple online serving.

---

## Capstone and Portfolio

Deliverables
- 1 Capstone (end‑to‑end): problem framing → data → modeling → deployment → docs
- 2–3 polished mid‑size projects from earlier stages

Checklist
- Clear README, architecture diagram, environment file, tests, Makefile/CLI
- Reproducible runs, tracked experiments, meaningful metrics, demo (app/notebook)

Presentation
- One‑page case study blog per project, emphasizing decisions, uncertainty, and impact.

---

## Appendix

- Datasets
  - [UCI Machine Learning Repository](https://archive.ics.uci.edu/) — Curated datasets for benchmarking.
  - [Kaggle Datasets](https://www.kaggle.com/datasets) — Variety + public notebooks.
  - [Google Dataset Search](https://datasetsearch.research.google.com/) — Meta‑search to find domain data.
- Templates
  - [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) — Standardized project structure.
- Study tips
  - Timeboxing, spaced repetition, “project‑first” learning — Improves retention and portfolio output.
