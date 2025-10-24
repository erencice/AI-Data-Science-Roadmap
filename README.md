# AI & Data Science Weekly Plan — Activities and Practice (Beginner-friendly, Cumulative)

Duration and pacing
- Duration: 120 weeks (≈2.3 years), 8–12 h/week (target 8–10 h)
- This plan assumes you start from zero and builds cumulatively.
- Weekly output = short practice tasks only (no files/reports/tests required).
- Each week lists up to 3 practice tasks that use only what you’ve learned so far.
- Flex Weeks are included to reduce pressure and allow catch-up.

Table of Contents
- Workload Model and How to Use This Plan
- Global Conventions (lightweight)
- Pacing & Flex Weeks
- Phase 0 · Orientation and Setup
- Phase 1 · Mathematics for ML
- Phase 2 · Statistics Fundamentals
- Phase 3 · Bayesian Statistics & Missing Data
- Phase 4 · Machine Learning (Classical, Theory-Rich)
- Phase 5 · Econometrics & Time Series
- Phase 6 · R for Data Science (Advanced)
- Phase 7 · Web Scraping, SQL, Data Mining
- Phase 8 · Deep Learning
- Phase 9 · MLOps & Data Engineering
- Phase 10 · LLMs & Open-Source AI
- Phase 11 · Consolidation, Capstones, Portfolio
- Resource-to-Week Completion Map

Workload Model and How to Use This Plan
- Weekly mix: ~40% reading, ~45% coding/exercises, ~15% review.
- Practice-first: Every week ends with 1–3 practical exercises.
- No pass/fail rules for topics you haven’t learned yet; the plan is cumulative.
- If time is tight: push extras to the next Flex Week.

Global Conventions (lightweight)
- Keep your work in any format you like (notebooks, scripts, notes). No required filenames.
- Use reproducible seeds when you simulate or train.
- Add 3–5 spaced-repetition cards weekly if that helps you (optional).

Pacing & Flex Weeks (to lighten load)
Use Flex Weeks to:
- Finish/sketch practice you skipped
- Revisit tricky chapters
- Add a brief personal summary (optional)
- Make 10–20 flashcards (optional)

Flex Weeks (16 total): after Weeks 4, 8, 13, 16, 20, 24, 28, 31, 34, 39, 45, 52, 56, 62, 68, 76

---------------------------------------------------------------------

PHASE 0 · Orientation and Setup

Week 1 — Orientation, trig refresher (H~8–10)
Activities
- Install basics (Python, R, Jupyter/VS Code, Git) with your preferred approach
- Read: [Trigonometric Cheat Sheet (PDF)](https://tutorial.math.lamar.edu/pdf/Trig_Cheat_Sheet.pdf)
Practice
- Recreate 3 trigonometric identities numerically (e.g., angle addition) and plot sin/cos over one period.
- Explain in a short note how trig links to seasonality and rotations.

Week 2 — Python foundations (H~8–10)
Activities
- Watch: [Python Crash Course (YouTube)](https://www.youtube.com/watch?v=rfscVS0vtbw)
- Read: [Kevin Sheppard Python Notes (PDF)](https://www.kevinsheppard.com/files/teaching/python/notes/python_introduction_2021.pdf)
Practice
- Write small scripts that: (1) parse a CSV, (2) compute z-scores, (3) plot a histogram.
- Implement a simple Python class with dunder methods and demonstrate usage.

---------------------------------------------------------------------

PHASE 1 · Mathematics for ML

Week 3 — Python for Data Analysis I (H~8–10)
Activities
- Read: [Python for Data Analysis — Wes McKinney](https://wesmckinney.com/book/) (Ch. 1–4)
Practice
- Load any dataset and perform indexing, filtering, groupby, joins, and reshaping; make 4–6 plots.
- Summarize 3 insights and 2 hypotheses you’d test later.

Week 4 — Data cleaning patterns (H~8–10)
Activities
- Skim: [Python for Data Analysis (PDF, UOregon)](https://ix.cs.uoregon.edu/~norris/cis407/books/python_for_data_analysis.pdf)
- Read: [Data Science & Analytics with Python (PDF)](https://mathstat.dal.ca/~brown/sound/python/P1-Data_Science_and_Analytics_with_Python_2b29.pdf) (Part I)
Practice
- Build a small cleaning pipeline: handle missingness (simple impute), outliers (winsorize/clip), and data types.
- Document 5 common EDA pitfalls you observed and how you mitigated them.

Flex-01 — Catch-up and reinforce (H~6–8)
- Finish Week 3–4 practice
- Optional: 10 LeetCode warmups (arrays/strings) via [LeetCode Study Plan](https://leetcode.com/studyplan/) or [LeetCode Explore](https://leetcode.com/explore/learn/)

Week 5 — Linear Algebra I: PCA intuition (H~8–10)
Activities
- Read: [Mathematics for Machine Learning (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) (Linear Algebra chapters)
- Reference: [Everything You Always Wanted to Know About Mathematics (PDF)](https://www.math.cmu.edu/~jmackey/151_128/bws_book.pdf)
Practice
- Implement PCA from SVD on a toy dataset and plot variance explained and component loadings.

Week 6 — Linear Algebra II: conditioning and stability (H~8–10)
Activities
- Read: [The Theory of Matrices — Gantmacher (PDF)](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/gantmacher1.pdf) (selected sections)
Practice
- Create ill-conditioned matrices and study how small noise perturbs eigenvalues/vectors; relate to PCA stability.

Week 7 — Calculus and Optimization I (H~8–10)
Activities
- Read: [Mathematics for Machine Learning (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) (Calculus/Optimization chapters)
Practice
- Implement gradient descent for least squares and visualize convergence for different step sizes.

Week 8 — Algorithmic thinking (H~8–10)
Activities
- Learn DS&A basics: [BCS2 DS&A](https://opendsa.cs.vt.edu/ODSA/Books/bghs-stem-code-bcs/bcs2/spring-2020/1/html/index.html)
Practice
- Solve 8–10 short DS&A problems; write time/space complexity rationale for 3 of them.

Flex-02 — Catch-up and reinforce (H~6–8)
- Finish Week 5–8 practice
- Optional: More DS&A practice on [LeetCode](https://leetcode.com/)

---------------------------------------------------------------------

PHASE 2 · Statistics Fundamentals

Week 9 — Descriptive statistics (H~8–10)
Activities
- Read: [STAT 100](https://online.stat.psu.edu/stat100/), [STAT 200](https://online.stat.psu.edu/stat200/)
Practice
- Compute robust summaries (median, MAD, trimmed mean) vs mean/SD on 2 datasets; plot hist/KDE/box/ECDF.

Week 10 — Probability I (H~8–10)
Activities
- Read: [STAT 414](https://online.stat.psu.edu/stat414/)
Practice
- Simulate LLN and CLT for Binomial/Poisson/Normal; compare empirical moments to theory; include convergence plots.

Week 11 — Probability II (H~8–10)
Activities
- Continue: [STAT 414](https://online.stat.psu.edu/stat414/)
Practice
- Simulate correlated Normals via Cholesky and verify covariance recovery; explain independence vs zero-correlation.

Week 12 — Mathematical statistics I (H~8–10)
Activities
- Read: [STAT 415](https://online.stat.psu.edu/stat415/), [Casella & Berger — Statistical Inference (PDF)](https://pages.stat.wisc.edu/~shao/stat610/Casella_Berger_Statistical_Inference.pdf) (Ch. 1–7)
Practice
- Derive MLEs for Normal(μ,σ²), Exponential(λ), Bernoulli(p) and verify numerically on simulated data.

Week 13 — Mathematical statistics II (H~8–10)
Activities
- Continue: [STAT 415](https://online.stat.psu.edu/stat415/)
Practice
- Compare analytic CIs vs bootstrap CIs; run a small coverage study across n and skewness.

Flex-03 — Catch-up and reinforce (H~6–8)
- Finish Weeks 9–13 practice

Week 14 — Hypothesis testing (H~8–10)
Activities
- Read: [STAT 500](https://online.stat.psu.edu/stat500/), [Practitioner’s Guide to Statistical Tests](https://vkteam.medium.com/practitioners-guide-to-statistical-tests-ed2d580ef04f#1e3b), [Planning A/B Tests](https://towardsdatascience.com/step-by-step-for-planning-an-a-b-test-ef3c93143c0b)
Practice
- Design a basic A/B test (effect size, α, power) and simulate Type I/II error rates under your assumptions.

Week 15 — Regression I (H~8–10)
Activities
- Read: [STAT 501](https://online.stat.psu.edu/stat501/)
Practice
- Fit a simple OLS model; perform residual diagnostics and discuss assumptions; compute VIFs for multicollinearity.

Week 16 — ANOVA & DoE (H~8–10)
Activities
- Read: [STAT 502](https://online.stat.psu.edu/stat502/)
Practice
- Run one-way and two-way ANOVA on a dataset; report effects and discuss design considerations.

Flex-04 — Catch-up and reinforce (H~6–8)
- Finish Weeks 14–16 practice

Week 17 — Discrete data & GLMs (H~8–10)
Activities
- Read: [STAT 504](https://online.stat.psu.edu/stat504/), [Statsmodels GLM docs](https://www.statsmodels.org/stable/glm.html)
Practice
- Fit logistic regression; produce ROC/PR and calibration curve; interpret odds ratios.

Week 18 — Multivariate analysis (H~8–10)
Activities
- Read: [STAT 505](https://online.stat.psu.edu/stat505/)
Practice
- Compare PCA→LDA vs logistic regression on a classification dataset; discuss when LDA helps.

Week 19 — Sampling theory (H~8–10)
Activities
- Read: [STAT 506](https://online.stat.psu.edu/stat506/)
Practice
- Simulate SRS vs stratified vs cluster sampling and compare estimator variance and cost trade-offs.

Week 20 — Epidemiology (H~8–10)
Activities
- Read: [STAT 507](https://online.stat.psu.edu/stat507/)
Practice
- Compute OR and RR with CIs from contingency tables; sketch a simple causal DAG and discuss confounding.

Flex-05 — Catch-up and reinforce (H~6–8)
- Finish Weeks 17–20 practice

Week 21 — Time series (H~8–10)
Activities
- Read: [STAT 510](https://online.stat.psu.edu/stat510/)
Practice
- Identify and fit an ARIMA/SARIMA; do rolling-origin evaluation; check residuals (ACF/PACF, Ljung–Box).

Week 22 — Applied research methods (H~6–8)
Activities
- Read: [STAT 800](https://online.stat.psu.edu/stat800/)
Practice
- Draft a short preregistration plan (estimands, primary outcome, α, power, stopping rules) for a simple study.

Week 23 — Think Stats (H~8–10)
Activities
- Read: [Think Stats (PDF)](https://greenteapress.com/thinkstats/thinkstats.pdf) (Ch. 1–8)
Practice
- Implement PMF/CDF/ECDF utilities on real data; compare empirical vs parametric summaries.

Week 24 — EDA automation & patterns (H~8–10)
Activities
- Review: [Python for Data Analysis](https://wesmckinney.com/book/); Try: [Dataprep.ai](https://dataprep.ai)
Practice
- Build a reusable EDA workflow (config-friendly) and run it on two datasets with a short sanity-check list.

Flex-06 — Catch-up and reinforce (H~6–8)
- Finish Weeks 21–24 practice

---------------------------------------------------------------------

PHASE 3 · Bayesian Statistics & Missing Data

Week 25 — Bayesian I (H~8–10)
Activities
- Read: [Think Bayes](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple) (Ch. 1–6)
Practice
- Implement Beta–Binomial A/B with priors of your choice; compute posterior, decisions under simple loss, and PPC.

Week 26 — Bayesian II (H~8–10)
Activities
- Continue: [Think Bayes](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple)
- Reference: [ArviZ](https://www.arviz.org/)
Practice
- Compare two simple models with WAIC/LOO and discuss prior/posterior predictive checks.

Week 27 — MCMC & hierarchical models (H~8–10)
Activities
- Choose one: [PyMC](https://www.pymc.io/welcome.html) or [NumPyro](https://num.pyro.ai/en/stable/); use [ArviZ](https://www.arviz.org/) for diagnostics
Practice
- Fit a hierarchical normal or logistic model; report R-hat/ESS and demonstrate partial pooling benefit.

Week 28 — Missing data & MI (H~8–10)
Activities
- Read: [Flexible Imputation of Missing Data (FIMD)](https://stefvanbuuren.name/fimd/)
Practice
- Run a simple MICE workflow on a dataset; compare MI vs complete-case estimates and discuss sensitivity.

Flex-07 — Catch-up and reinforce (H~6–8)
- Finish Weeks 25–28 practice

---------------------------------------------------------------------

PHASE 4 · Machine Learning (Classical, Theory-Rich)

Week 29 — Validation & metrics (H~8–10)
Activities
- Read: [mlcourse.ai Book](https://mlcourse.ai/book/index.html), [sklearn model_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)
Practice
- Build a leakage-safe pipeline with nested CV; justify metric choice for your problem.

Week 30 — Linear models & regularization (H~8–10)
Activities
- Read: [An Introduction to Statistical Learning](https://www.statlearning.com/) (linear models, Ridge/Lasso)
Practice
- Plot regularization paths and discuss bias–variance trade-offs for Ridge vs Lasso.

Week 31 — Trees & ensembles (H~8–10)
Activities
- Read: [mlcourse.ai Book](https://mlcourse.ai/book/index.html) (trees/ensembles), [Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html), [SHAP](https://shap.readthedocs.io/en/latest/)
Practice
- Train RF and GBM; compare performance and feature importance stability across resamples.

Flex-08 — Catch-up and reinforce (H~6–8)
- Finish Weeks 29–31 practice

Week 32 — SVMs, kNN, imbalance (H~8–10)
Activities
- Read: [ISL](https://www.statlearning.com/) (SVM/kNN), [Calibration](https://scikit-learn.org/stable/modules/calibration.html)
Practice
- Fit SVM and kNN; calibrate probabilities; evaluate with cost-sensitive metrics on imbalanced data.

Week 33 — Clustering (H~8–10)
Activities
- Read: [mlcourse.ai Book](https://mlcourse.ai/book/index.html) (clustering), [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), [GMM](https://scikit-learn.org/stable/modules/mixture.html)
Practice
- Compare k-means, hierarchical, DBSCAN, and GMM on synthetic and real data; use silhouette/BIC.

Week 34 — DR & anomalies (H~8–10)
Activities
- Read: PCA review; [t‑SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [UMAP](https://umap-learn.readthedocs.io/en/latest/); [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html), [LOF](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
Practice
- Contrast PCA vs t‑SNE/UMAP visuals; detect planted anomalies with IsolationForest/LOF and report precision/recall.

Flex-09 — Catch-up and reinforce (H~6–8)
- Finish Weeks 32–34 practice

Week 35 — PRML I (H~8–10)
Activities
- Read: [Pattern Recognition and Machine Learning — Bishop (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
Practice
- Implement logistic regression from scratch (GD or Newton) and show parity with sklearn on a small dataset.

Week 36 — PRML II + sklearn pipelines (H~8–10)
Activities
- Read: [Kernel ridge (sklearn)](https://scikit-learn.org/stable/modules/kernel_ridge.html) and PRML kernels
Practice
- Run an RBF kernel ridge experiment and discuss hyperparameter effects.

Week 37 — Interpretability (H~8–10)
Activities
- Read: [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
Practice
- Produce PDP/ICE and permutation/SHAP explanations; test stability across resamples.

Week 38 — Think Complexity (H~6–8)
Activities
- Read: [Think Complexity (2e)](https://greenteapress.com/wp/think-complexity/) (selected chapters)
Practice
- Write 3 actionable implications for your ML system design from complexity insights.

Week 39 — ML Refined (H~8–10)
Activities
- Read: [Machine Learning Refined](https://github.com/neonwatty/machine-learning-refined) (selected)
Practice
- Visualize decision boundaries as model complexity increases; relate to bias–variance.

Flex-10 — Catch-up and reinforce (H~6–8)
- Finish Weeks 35–39 practice

---------------------------------------------------------------------

PHASE 5 · Econometrics & Time Series

Week 40 — Econometric Theorems (H~8–10)
Activities
- Read: [Econometric Theorems (Book)](https://bookdown.org/ts_robinson1994/10EconometricTheorems/)
Practice
- Select 10 theorems and jot a one-line “when it matters” note for each in applied modeling.

Week 41 — Gujarati (H~8–10)
Activities
- Read: [Basic Econometrics — Gujarati (PDF)](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf)
Practice
- Replicate 1–2 textbook regressions in Python (statsmodels) and interpret coefficients and SEs.

Week 42 — Brooks (Finance) (H~8–10)
Activities
- Read: [Intro Econometrics for Finance — Brooks (PDF)](https://new.mmf.lnu.edu.ua/wp-content/uploads/2018/03/brooks_econometr_finance_2nd.pdf)
- Reference: [Newey–West in statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html)
Practice
- Fit a finance regression; compute HAC SEs; discuss why raw R² can be misleading.

Week 43 — Forecasting I (H~8–10)
Activities
- Read: [FPP3](https://otexts.com/fpp3/), follow an [ARIMA in Python tutorial](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)
Practice
- Build a walk-forward ARIMA/SARIMA forecast with a naive baseline and compare errors.

Week 44 — Forecasting II (H~8–10)
Activities
- Read: [FPPpy](https://otexts.com/fpppy/)
Practice
- Recreate your Week 43 workflow in R or refine Python; discuss calendar effects and evaluation.

Week 45 — Kaggle Time Series (H~8–10)
Activities
- Take: [Kaggle: Time Series Basics](https://www.kaggle.com/learn/time-series)
Practice
- Publish a minimal time-series notebook; note 3 choices that improved your score.

Flex-11 — Catch-up and reinforce (H~6–8)
- Finish Weeks 40–45 practice

Week 46 — Advanced Econometrics (H~8–10)
Activities
- Read: [Econometrics II](https://vladislav-morozov.github.io/econometrics-2/)
Practice
- Run a small Monte Carlo study comparing estimator bias/variance across sample sizes.

Week 47 — Panel heterogeneity (H~8–10)
Activities
- Read: [Econometrics with Unobserved Heterogeneity](https://vladislav-morozov.github.io/econometrics-heterogeneity/)
Practice
- Fit FE and RE models; run a Hausman test; discuss when RE may be invalid.

Week 48 — R implementations & notes (H~8–10)
Activities
- Read: [Using R for Intro Econometrics](https://pyoflife.com/using-r-for-introductory-econometrics/), [Diebold Notes (PDF)](https://www.sas.upenn.edu/~fdiebold/Teaching104/Econometrics.pdf)
Practice
- Re-implement one econometric example in R with tidy models; compare conclusions to Python.

Week 49 — Greene (H~8–10)
Activities
- Read: [Greene — Econometric Analysis (PDF)](https://www.ctanujit.org/uploads/2/5/3/9/25393293/_econometric_analysis_by_greence.pdf)
Practice
- Write a 1–2 page synthesis linking Greene’s theory to methods you used so far.

Weeks 50–52 — Lütkepohl (H~10–12 each)
Activities
- Read: [New Introduction to Multiple Time Series — Lütkepohl (PDF)](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)
- Reference: [statsmodels VECM](https://www.statsmodels.org/stable/vector_ar.html#vecm)
Practice
- Fit VAR/VECM to a macro dataset; produce IRFs/FEVD; discuss cointegration results.

Flex-12 — Catch-up and reinforce (H~6–8)
- Finish Weeks 46–52 practice

---------------------------------------------------------------------

PHASE 6 · R for Data Science (Advanced)

Week 53 — R4DS I (H~8–10)
Activities
- Read: [R for Data Science (2e)](https://r4ds.hadley.nz)
Practice
- Build an R EDA workflow on a fresh dataset (wrangle + visualize).

Week 54 — R4DS II + Efficient R (H~8–10)
Activities
- Read: [Efficient R Programming](https://csgillespie.github.io/efficientR/)
Practice
- Profile a slow R pipeline and make it at least ~2× faster via vectorization or other improvements.

Week 55 — R Graphics + Text (H~8–10)
Activities
- Read: [R Graphics Cookbook](https://r-graphics.org), [Tidy Text Mining](https://www.tidytextmining.com)
Practice
- Create a short RMarkdown/Quarto report with 3 polished plots and a basic text analysis.

Week 56 — GLMs/Multilevel in R (H~8–10)
Activities
- Read: [Applied GLMs and Multilevel Models in R](https://bookdown.org/roback/bookdown-BeyondMLR/)
Practice
- Fit a multilevel logistic with random intercepts; interpret variance components and ICC.

Flex-13 — Catch-up and reinforce (H~6–8)
- Finish Weeks 53–56 practice

---------------------------------------------------------------------

PHASE 7 · Web Scraping, SQL, Data Mining

Week 57 — Web scraping I (H~8–10)
Activities
- Read: [Beautiful Soup Docs](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) (or [mirror](https://tedboy.github.io/bs4_doc/index.html)), [Selenium Docs](https://selenium-python.readthedocs.io/index.html)
Practice
- Scrape one static page and one dynamic page politely (robots.txt, backoff) and store results locally.

Week 58 — Web scraping II (H~8–10)
Activities
- Practice: [Practice Web Scraping](https://www.scrapingcourse.com/ecommerce/)
Practice
- Build a mini ETL: scrape → clean → store (simple DB or files) → query; note 3 ethical risks and mitigations.

Weeks 59–60 — Data Mining (H~8–10 each)
Activities
- Read: [Data Mining: Concepts and Techniques (3e)](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
Practice
- Implement one classic algorithm (e.g., Apriori) and compare results to a library implementation.

Week 61 — SQL I (H~8–10)
Activities
- Learn: [SQL Tutorial](https://www.sqltutorial.org/)
Practice
- Write 25–30 queries (joins, subqueries, windows) on any relational dataset.

Week 62 — SQL II (H~8–10)
Activities
- Read: [SQL Roadmap](https://roadmap.sh/sql)
Practice
- Design a small analytics schema and write 8–10 complex CTE/window queries with clear intent.

Flex-14 — Catch-up and reinforce (H~6–8)
- Finish Weeks 57–62 practice

---------------------------------------------------------------------

PHASE 8 · Deep Learning

Weeks 63–64 — D2L (H~8–10 each)
Activities
- Read/Do: [Dive into Deep Learning](https://d2l.ai) (MLP/CNN/RNN basics; training loops; regularization)
Practice
- Train two small models and run an ablation on optimizer/regularization with seeded runs.

Week 65 — Transformers (H~8–10)
Activities
- Read: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
Practice
- Implement a toy self-attention block and verify shape/mask behavior and scaling.

Weeks 66–67 — Deep Learning Book (H~8–10 each)
Activities
- Read: [Deep Learning — Goodfellow et al.](https://www.deeplearningbook.org/)
Practice
- Connect two theory topics (optimization/regularization) to specific training choices; demonstrate impact.

Week 68 — Applied ML Practices (H~8–10)
Activities
- Read: [Applied ML Practices](https://github.com/eugeneyan/applied-ml)
Practice
- Adopt 3 production-friendly patterns (e.g., data validation, sampling strategy, config-driven pipelines) in a toy project.

Flex-15 — Catch-up and reinforce (H~6–8)
- Finish Weeks 63–68 practice

---------------------------------------------------------------------

PHASE 9 · MLOps & Data Engineering

Weeks 69–72 — MLOps Zoomcamp (H~8–10 each)
Activities
- Follow: [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
Practice
- Set up experiment tracking, a simple orchestrated training job, containerize a small model, and serve it locally.

Weeks 73–74 — ML Systems (H~8–10 each)
Activities
- Read: [Machine Learning Systems (Book)](https://mlsysbook.ai)
Practice
- Draft an architecture for a realistic ML use-case (dataflow, SLAs, monitoring, rollback).

Weeks 75–76 — Data Engineering Zoomcamp (H~8–10 each)
Activities
- Follow: [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
Practice
- Build a mini batch + stream pipeline; note cost/performance trade-offs and idempotency behavior.

Flex-16 — Catch-up and reinforce (H~6–8)
- Finish Weeks 69–76 practice

---------------------------------------------------------------------

PHASE 10 · LLMs & Open-Source AI

Weeks 77–78 — Hugging Face Course (H~8–10 each)
Activities
- Follow: [Hugging Face Course](https://huggingface.co/course/chapter1)
Practice
- Fine-tune a small model and evaluate with an appropriate metric; record key training settings.

Week 79 — AI Agents (H~8–10)
Activities
- Follow: [HF Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)
Practice
- Build a simple tool-using agent with timeouts/fail-safes; list your safety checks.

---------------------------------------------------------------------

PHASE 11 · Consolidation, Capstones, Portfolio

Week 80 — Statsmodels deep dive (H~8–10)
Activities
- Read: [Statsmodels Docs](https://www.statsmodels.org/stable/index.html)
Practice
- Recreate two econometric analyses in statsmodels (use robust SEs if needed) and compare to earlier work.

Week 81 — scikit-learn deep dive (H~8–10)
Activities
- Read: [scikit-learn Docs](https://scikit-learn.org/stable/index.html)
Practice
- Build a clean template pipeline (preprocess, CV, metrics, calibration) and run it on a new dataset.

Week 82 — AI Engineering reading (H~6–8)
Activities
- Browse: [AI Engineering Reading List (Latent.Space)](https://www.latent.space/p/2025-papers)
Practice
- Write 3 one-page takeaways connecting papers to your projects or interests.

Week 83 — Think Like a Data Scientist (H~6–8)
Activities
- Read: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3475303), [Full PDF](https://lmsspada.kemdiktisaintek.go.id/pluginfile.php/752025/mod_resource/content/2/Think%20Like%20a%20Data%20Scientist.pdf)
Practice
- Produce a stakeholder decision memo template and fill it for a past analysis.

Week 84 — Algorithmic Trading Cookbook (H~8–10)
Activities
- Browse: [Python for Algorithmic Trading Cookbook (GitHub)](https://github.com/PacktPublishing/Python-for-Algorithmic-Trading-Cookbook)
Practice
- Backtest two simple strategies and report risk/return metrics with clear caveats.

Weeks 85–92 — Capstone 1 (H~8–10 each)
Scope
- Acquire and prepare data; handle missingness; do inference; add a Bayesian component; train/validate models; ensure interpretability.
Practice
- Run an end-to-end project with documented assumptions and uncertainty; produce a concise technical summary and a non-technical brief.

Weeks 93–100 — Capstone 2: Econometrics & Forecasting (H~8–10 each)
Scope
- Build VAR/VECM or a causal panel; compare to ML forecasting; check stability/diagnostics; outline a deployment plan.
Practice
- Deliver forecasts with error analysis and an operationalization note.

Weeks 101–104 — Portfolio & Oral defense (H~8–10 each)
Activities
- Clean up code/notebooks; add walkthroughs; rehearse an oral defense.
Practice
- Prepare short explanations/derivations: Neyman–Pearson, OLS and Var(β̂), Beta–Binomial posterior, ARIMA identification, bias–variance trade-off.

---------------------------------------------------------------------

Resource-to-Week Completion Map (all click-through)
- Mathematics for ML — Deisenroth/Faisal/Ong: Weeks 5–7 — [MML Book (PDF)](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf)
- Everything You Always Wanted to Know About Mathematics: Weeks 5–7 (reference) — [Book (PDF)](https://www.math.cmu.edu/~jmackey/151_128/bws_book.pdf)
- Trigonometric Cheat Sheet: Week 1 — [PDF](https://tutorial.math.lamar.edu/pdf/Trig_Cheat_Sheet.pdf)
- The Theory of Matrices — Gantmacher: Week 6 — [PDF](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/gantmacher1.pdf)

- STAT 100, 200: Week 9 — [STAT 100](https://online.stat.psu.edu/stat100/), [STAT 200](https://online.stat.psu.edu/stat200/)
- STAT 414: Weeks 10–11 — [Course](https://online.stat.psu.edu/stat414/)
- STAT 415: Weeks 12–13 — [Course](https://online.stat.psu.edu/stat415/), [Casella & Berger PDF](https://pages.stat.wisc.edu/~shao/stat610/Casella_Berger_Statistical_Inference.pdf)
- STAT 500: Week 14 — [Course](https://online.stat.psu.edu/stat500/)
- A/B Testing guides: Week 14 — [Practitioner’s Guide](https://vkteam.medium.com/practitioners-guide-to-statistical-tests-ed2d580ef04f#1e3b), [Planning A/B Tests](https://towardsdatascience.com/step-by-step-for-planning-an-a-b-test-ef3c93143c0b)
- Think Stats: Week 23 — [PDF](https://greenteapress.com/thinkstats/thinkstats.pdf)
- Think Bayes: Weeks 25–26 — [Text](https://open.umn.edu/opentextbooks/textbooks/think-bayes-bayesian-statistics-made-simple)
- STAT 501–507, 510, 800: Weeks 15–22 — [501](https://online.stat.psu.edu/stat501/), [502](https://online.stat.psu.edu/stat502/), [504](https://online.stat.psu.edu/stat504/), [505](https://online.stat.psu.edu/stat505/), [506](https://online.stat.psu.edu/stat506/), [507](https://online.stat.psu.edu/stat507/), [510](https://online.stat.psu.edu/stat510/), [800](https://online.stat.psu.edu/stat800/)

- Econometrics: Week 40 — [Econometric Theorems](https://bookdown.org/ts_robinson1994/10EconometricTheorems/); Week 41 — [Gujarati PDF](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf); Week 42 — [Brooks PDF](https://new.mmf.lnu.edu.ua/wp-content/uploads/2018/03/brooks_econometr_finance_2nd.pdf); Week 48 — [Using R for Intro Econometrics](https://pyoflife.com/using-r-for-introductory-econometrics/), [Diebold Notes PDF](https://www.sas.upenn.edu/~fdiebold/Teaching104/Econometrics.pdf); Week 49 — [Greene PDF](https://www.ctanujit.org/uploads/2/5/3/9/25393293/_econometric_analysis_by_greence.pdf); Weeks 50–52 — [Lütkepohl PDF](https://www.cur.ac.rw/mis/main/library/documents/book_file/2005_Book_NewIntroductionToMultipleTimeS.pdf)

- Python foundations: Week 2 — [Python Crash Course Video](https://www.youtube.com/watch?v=rfscVS0vtbw), [Kevin Sheppard Notes PDF](https://www.kevinsheppard.com/files/teaching/python/notes/python_introduction_2021.pdf); Week 3–4 — [Python for Data Analysis](https://wesmckinney.com/book/), [P4DA PDF (UOregon)](https://ix.cs.uoregon.edu/~norris/cis407/books/python_for_data_analysis.pdf), [Data Science & Analytics with Python PDF](https://mathstat.dal.ca/~brown/sound/python/P1-Data_Science_and_Analytics_with_Python_2b29.pdf)

- Validation/ML: Weeks 29–31, 33 — [mlcourse.ai Book](https://mlcourse.ai/book/index.html); Week 30, 32 — [ISL](https://www.statlearning.com/); Week 37 — [Interpretable ML](https://christophm.github.io/interpretable-ml-book/); Weeks 35–36 — [PRML PDF](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf); Week 38 — [Think Complexity](https://greenteapress.com/wp/think-complexity/); Week 39 — [ML Refined](https://github.com/neonwatty/machine-learning-refined); sklearn docs throughout — [scikit-learn](https://scikit-learn.org/stable/index.html)

- SQL & Scraping & Mining: Week 57 — [Beautiful Soup Docs](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) / [Mirror](https://tedboy.github.io/bs4_doc/index.html), [Selenium Docs](https://selenium-python.readthedocs.io/index.html); Week 58 — [Practice Web Scraping](https://www.scrapingcourse.com/ecommerce/); Week 61 — [SQL Tutorial](https://www.sqltutorial.org/); Week 62 — [SQL Roadmap](https://roadmap.sh/sql); Weeks 59–60 — [Data Mining 3e PDF](https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)

- Deep Learning: Weeks 63–64 — [Dive into Deep Learning](https://d2l.ai); Week 65 — [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/); Weeks 66–67 — [Deep Learning Book](https://www.deeplearningbook.org/); Week 68 — [Applied ML Practices](https://github.com/eugeneyan/applied-ml)

- MLOps & DE: Weeks 69–72 — [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp); Weeks 73–74 — [ML Systems Book](https://mlsysbook.ai); Weeks 75–76 — [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)

- LLMs & Agents: Weeks 77–78 — [Hugging Face Course](https://huggingface.co/course/chapter1); Week 79 — [HF Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)

- Final readings & portfolio: Week 82 — [Latent.Space list](https://www.latent.space/p/2025-papers); Week 83 — [Think Like a Data Scientist SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3475303) / [Full PDF](https://lmsspada.kemdiktisaintek.go.id/pluginfile.php/752025/mod_resource/content/2/Think%20Like%20a%20Data%20Scientist.pdf); Week 84 — [Algorithmic Trading Cookbook](https://github.com/PacktPublishing/Python-for-Algorithmic-Trading-Cookbook)

All links are clickable. Practice tasks are limited to what you’ve learned so far, and weeks are cumulative with Flex Weeks for catch-up.
