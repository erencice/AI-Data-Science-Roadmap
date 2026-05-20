# Data Science & AI Mastery Roadmap (Optimized 2026)

> A professional, execution-oriented syllabus for serious practitioners.
>
> **Pace:** 10–12 hrs/week · **Duration:** ~52 weeks core + 6 buffer + 8 elective · **Format:** Self-paced, project-driven

---

## Roadmap Overview

| Phase | Name | Duration |
|-------|------|----------|
| 1 | Data Foundations | 5 weeks |
| 2 | Statistics & Probability | 6 weeks |
| 3 | ML Foundations | 6 weeks |
| 4 | Applied ML | 5 weeks |
| 5 | Deep Learning | 8 weeks |
| 5.5 | LLM/GenAI Engineering | 6 weeks |
| 6 | Time Series & Causal | 4 weeks |
| 7 | MLOps & Production | 6 weeks |
| 8 | Capstone & Portfolio | 8 weeks |
| | **Core Total** | **~52 weeks** |
| | Buffer Weeks (6x) | **~6 weeks** |
| E1 | Bayesian & Advanced (Elective) | 4 weeks |
| E2 | Generative AI & Diffusion (Elective) | 4 weeks |

---

## How to Use This Roadmap

### Learning Principles

1. **30/70 Rule:** Maximum 30% theory, 70% hands-on coding every week. If you spent 4 hours reading, spend 9+ hours coding.
2. **Max 3 New Concepts/Week:** Cognitive load management. If a week introduces probability distributions, hypothesis testing, and Python simulation — that's 3. No more.
3. **Spaced Repetition:** Every 4–6 weeks, revisit earlier concepts by applying them in new contexts. The roadmap builds this in automatically.
4. **Code Every Day:** Passive reading without implementation is not learning. Even 30 minutes of coding beats 3 hours of reading.
5. **Read the Checkpoint First:** Before starting each week, know exactly what "done" looks like. Work backward from the deliverable.

### Anti-Pattern Warnings

| Anti-Pattern | What It Looks Like | How to Avoid |
|---|---|---|
| **Tutorial Hell** | Watching videos without writing original code | 1:2 rule — 1 hour tutorial = 2 hours hands-on coding |
| **Certification Collecting** | Collecting certificates but no GitHub activity | GitHub > Certificate. Every phase ends with a project, not a quiz |
| **Perfectionism** | Spending 3 weeks on one notebook | "Done > Perfect." Time-box each checkpoint to 1 week |
| **Theory Overload** | Reading 200 pages before writing code | Code-first approach — try, fail, then read to understand why |
| **Skipping Phases** | Jumping to Deep Learning without stats | Each phase is a prerequisite. Never skip. |

### Quick Wins

| Week | Deliverable |
|------|-------------|
| **W2** | First EDA — real-world dataset with 5+ insights, 3+ visuals |
| **W4** | First Quarto report — professional HTML/PDF analysis |
| **W5** | M1: Data Ready — pandas + SQL practical exam |
| **W11** | M2: Stats Ready — A/B test analysis with correct interpretation |
| **W17** | M3: ML Ready — first ML model comparison study on Kaggle |

### Buffer Weeks

Every 8 weeks there is a built-in buffer week for catch-up, review, project extension, or rest:

| Range | Buffer Week |
|-------|-------------|
| W1–8 | **W9** |
| W10–17 | **W18** |
| W19–26 | **W27** |
| W28–35 | **W36** |
| W37–44 | **W45** |
| W46–52 | **Capstone extension** |

Buffer weeks are **planned**, not optional. Use them intentionally.

---

<details>
<summary><h2>Phase 1: Data Foundations</h2></summary>

| | |
|---|---|
| **Duration** | 5 Weeks |
| **Resources** | [Python for Data Analysis, 3rd Ed. — Wes McKinney](https://wesmckinney.com/book/), [SQLBolt](https://sqlbolt.com/), [DataLemur](https://datalemur.com/), [pgexercises](https://pgexercises.com/) |
| **Depth** | McKinney ~550pp (Ch. 1–12), SQLBolt ~20 exercises, DataLemur (interview-focused). At 10–12 hrs/week: ~80pp + 50 SQL exercises total. |

### Week 1: Python Environment, NumPy & pandas Basics

**Study:**
- McKinney: Chapters 1–4 (Preliminaries, Python Basics, NumPy Basics)
- Set up environment: Python 3.11+, Jupyter Lab or VS Code, conda/venv, git

**Practice:**
- Write 30+ Python one-liners: list comprehensions, dict operations, generators
- Implement matrix multiplication from scratch using only NumPy indexing (no `np.matmul`)
- Benchmark pure Python loop vs. vectorized NumPy for 3 operations

**Checkpoint:**
> `week01_foundations.ipynb`: NumPy array operations, broadcasting visual explanation, performance benchmark table, and 10 self-written `assert` tests. Must run end-to-end with a single "Run All."

---

### Week 2: pandas — Loading, Cleaning & First EDA *(Quick Win #1)*

**Study:**
- McKinney: Chapters 5–7 (pandas intro, Data Loading, Data Cleaning)
- Focus: Series, DataFrame, `.loc`/`.iloc`, handling missing values, dtypes

**Practice:**
- Download a real-world messy dataset (e.g., NYC 311, Airbnb listings) with 50,000+ rows
- Build a complete data cleaning pipeline: handle nulls, fix dtypes, parse dates, rename columns
- Perform exploratory analysis: 5+ insights, 3+ visualizations

**Checkpoint:**
> `week02_eda.ipynb`: Load messy dataset → generate data quality report (% nulls, dtype summary, unique counts) → clean DataFrame → 5 insights with visualizations → save to Parquet. This is your first portfolio piece — push to GitHub.

---

### Week 3: pandas — GroupBy, Merging & SQL Foundations

**Study:**
- McKinney: Chapters 8–10 (Wrangling, GroupBy, Aggregation)
- SQLBolt: Lessons 1–10 (SELECT, WHERE, JOINs, aggregates)

**Practice:**
- Using your Week 2 dataset: 3+ non-trivial GroupBy analyses with custom `.agg()` calls
- Practice `merge`, `join`, `concat` by combining two related datasets
- Complete SQLBolt exercises 1–10

**Checkpoint:**
> `week03_groupby_sql.ipynb`: Multi-key merge (all 4 join types), pivot table, GroupBy with custom aggregations. Plus: SQLBolt completion screenshots.

---

### Week 4: Advanced SQL & First Quarto Report *(Quick Win #2)*

**Study:**
- SQLBolt: Lessons 11–15 (Window functions, subqueries)
- DataLemur: 10 Easy/Medium SQL problems
- Install Quarto

**Practice:**
- Solve 10 DataLemur SQL problems (focus on window functions: `ROW_NUMBER`, `RANK`, `LAG`)
- Use `pgexercises.com` for 10 PostgreSQL practice problems
- Convert your Week 2 EDA into a professional Quarto report

**Checkpoint:**
> `phase1_report.qmd`: Quarto document with code, figures, tables, and narrative. Must render to both HTML and PDF via `quarto render`. Include: data quality analysis, key findings, limitations. Publish to GitHub Pages.

---

### Week 5: Phase 1 Capstone — End-to-End Analysis

**Study:**
- McKinney: Chapters 11–12 (Time Series, Advanced pandas) — skim for awareness
- Review any weak areas from Weeks 1–4

**Practice:**
- End-to-end analysis on a new dataset (choose from: [Kaggle Datasets](https://www.kaggle.com/datasets), [UCI ML Repository](https://archive.ics.uci.edu/))
- Pipeline: ingestion → cleaning → feature engineering → GroupBy insights → visualization → narrative conclusions

**Checkpoint:**
> `phase1_capstone/`: Complete analysis with Quarto report, 8+ visualizations, summary table of findings, and a "Limitations & Next Steps" section. Must be reproducible with a single `quarto render` command.

**M1: Data Ready** — You can load, clean, analyze, and visualize any tabular dataset. You can write SQL with window functions.

</details>

---

<details>
<summary><h2>Phase 2: Statistics & Probability</h2></summary>

| | |
|---|---|
| **Duration** | 6 Weeks |
| **Resources** | [StatQuest (YouTube)](https://www.youtube.com/c/joshstarmer), [OpenIntro Statistics](https://www.openintro.org/book/os/), [3Blue1Brown Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), [Seeing Theory](https://seeing-theory.brown.edu/) |
| **Depth** | StatQuest ~15 hrs, OpenIntro ~400pp (Ch. 1–8), 3B1B ~3 hrs, Seeing Theory interactive. 100% free, practitioner-focused. |

### Week 6: Probability Distributions & Descriptive Statistics

**Study:**
- StatQuest: Probability & Distribution series (~2 hrs)
- OpenIntro Statistics: Chapters 1–3 (Data types, distributions, probability)
- Seeing Theory: Probability & Distributions interactive modules

**Practice:**
- Simulate the Birthday Problem in Python (group sizes 1–80); plot probability curve
- Monte Carlo estimator for Birthday Problem — verify convergence to analytic formula
- Plot PDFs/CDFs for Normal, Exponential, Poisson, Binomial with varied parameters

**Checkpoint:**
> `week06_probability.ipynb`: Birthday Problem simulation + convergence plot, distribution PDF/CDF grid with parameter sensitivity, and written explanation of when to use each distribution.

---

### Week 7: Hypothesis Testing & Confidence Intervals

**Study:**
- StatQuest: Hypothesis Testing series (~2 hrs)
- OpenIntro Statistics: Chapters 4–5 (Statistical inference, confidence intervals)

**Practice:**
- Implement permutation test from scratch for two-sample comparison
- Implement bootstrap confidence intervals (95% CI) for mean, median, variance
- Compare results to `scipy.stats.ttest_ind` — discuss when they differ

**Checkpoint:**
> `week07_hypothesis.ipynb`: Permutation test vs. t-test on real data, bootstrap CI with 1,000 resamples visualized, and written discussion of assumptions and when to use each test.

---

### Week 8: Linear Algebra for ML

**Study:**
- 3Blue1Brown: Episodes 1–10 (Vectors, matrices, determinants, eigenvectors, SVD)
- Seeing Theory: Linear Algebra interactive modules

**Practice:**
- Implement Power Iteration from scratch to find dominant eigenvector
- Apply eigendecomposition to PCA on a 2D dataset — compare to `sklearn.decomposition.PCA`
- Image compression using SVD: show reconstruction at k=5, 20, 50, 100 singular values

**Checkpoint:**
> `week08_linalg.ipynb`: Power Iteration with convergence tracking, manual PCA vs. sklearn comparison, SVD image compression with PSNR metric table.

---

### Week 9: Regression Foundations & A/B Test Analysis

**Study:**
- StatQuest: Linear Regression series (~1.5 hrs)
- OpenIntro Statistics: Chapter 8 (Multiple & logistic regression)

**Practice:**
- Implement OLS regression from scratch using matrix algebra (no sklearn)
- Analyze residuals: heteroskedasticity, autocorrelation, Q-Q plots
- Download a real A/B test dataset; run full analysis with hypothesis testing

**Checkpoint:**
> `week09_regression.ipynb`: From-scratch OLS with coefficient comparison to `statsmodels.OLS`, residual diagnostic plots (4-panel), and A/B test analysis with business recommendation.

---

### Week 10: Statistical Modeling Essentials

**Study:**
- StatQuest: Logistic Regression, Poisson Regression, ANOVA (~2 hrs)
- OpenIntro Statistics: Chapter 9 (ANO & model diagnostics)

**Practice:**
- Implement logistic regression from scratch using gradient descent
- Apply to a binary classification dataset; plot decision boundary
- Run one-way ANOVA on a real dataset; follow up with Tukey's HSD

**Checkpoint:**
> `week10_stat_modeling.ipynb`: From-scratch logistic regression with learning curve, ANOVA with post-hoc test, and written interpretation of all results in business context.

---

### Week 11: Phase 2 Capstone — Data-Driven Decision

**Study:**
- Review all StatQuest videos from this phase
- Re-read OpenIntro Statistics chapters where you felt weakest

**Practice:**
- Full statistical analysis on a business problem: "Did this marketing campaign actually work?"
- Must include: research question → data exploration → appropriate test selection → execution → interpretation → business recommendation

**Checkpoint:**
> `phase2_capstone.qmd`: Blog-post format statistical analysis with Quarto. Must include: research question, data description, test selection rationale, results with visualizations, and actionable business recommendation.

**M2: Stats Ready** — You can select and execute appropriate statistical tests, interpret p-values and confidence intervals, and translate results into business decisions.

</details>

---

<details>
<summary><h2>Phase 3: ML Foundations</h2></summary>

| | |
|---|---|
| **Duration** | 6 Weeks |
| **Resources** | [An Introduction to Statistical Learning with Python (ISLP)](https://www.statlearning.com/) |
| **Depth** | ISLP ~600pp with Python labs in every chapter. Gold-standard ML textbook. Focus Ch. 2–10. At 10–12 hrs/week: ~100pp + labs per week. |

### Week 12: Statistical Learning Overview & Linear Regression

**Study:**
- ISLP: Chapters 1–3 (Statistical Learning intro, Linear Regression) — full chapters + Python labs
- StatQuest: Cross-Validation series (~1 hr)

**Practice:**
- Complete the Chapter 3 Python lab in full
- Implement Ridge and Lasso regression from scratch using coordinate descent
- Plot coefficient paths vs. λ for both

**Checkpoint:**
> `week12_linreg.ipynb`: Full ISLP Chapter 3 lab + from-scratch Ridge/Lasso with coefficient path plots. Include model comparison table (OLS vs. Ridge vs. Lasso) with MSE, R², and training time.

---

### Week 13: Classification

**Study:**
- ISLP: Chapter 4 (Classification) — full chapter + Python lab
- Topics: Logistic Regression, LDA, QDA, KNN

**Practice:**
- Apply all 4 classifiers to a dataset; produce comparative confusion matrix table
- Implement k-fold cross-validation from scratch
- Tune KNN's k parameter using your from-scratch CV

**Checkpoint:**
> `week13_classification.ipynb`: All 4 classifiers with Accuracy, Precision, Recall, F1, AUC comparison table. From-scratch k-fold CV with hyperparameter tuning demonstration.

---

### Week 14: Resampling & Model Selection

**Study:**
- ISLP: Chapters 5–6 (Resampling Methods, Model Selection & Regularization) — full chapters + labs
- Focus: k-fold CV, bootstrap, best subset selection, Ridge, Lasso, PCR

**Practice:**
- Implement bootstrap for estimating standard error of any statistic (passed as a function)
- Best subset selection via exhaustive search for p ≤ 15
- Tune Ridge and Lasso via cross-validation; plot validation curve

**Checkpoint:**
> `week14_resampling.ipynb`: From-scratch bootstrap SE for 3 statistics, subset selection with AIC/BIC/adjusted-R² comparison, regularization path for Ridge and Lasso with optimal λ via 10-fold CV.

---

### Week 15: Nonlinear Models & Splines

**Study:**
- ISLP: Chapter 7 (Moving Beyond Linearity) — polynomial regression, step functions, splines, GAMs

**Practice:**
- Implement a cubic spline from scratch with knots at specified quantiles
- Fit a GAM to a dataset; plot partial dependence plots for each predictor
- Compare polynomial degree 1–5 on a nonlinear dataset; plot bias-variance trade-off

**Checkpoint:**
> `week15_nonlinear.ipynb`: From-scratch cubic spline vs. `scipy.interpolate.CubicSpline`, GAM with partial dependence plots, and written explanation of bias-variance trade-off for polynomial degree.

---

### Week 16: Decision Trees, Ensembles & SVMs

**Study:**
- ISLP: Chapters 8–9 (Tree-Based Methods, SVMs) — full chapters + labs
- Topics: CART, Random Forests, Gradient Boosting, Bagging, SVM (linear, RBF)

**Practice:**
- Implement CART (classification tree) from scratch using recursive binary splitting and Gini impurity
- Train Random Forest and Gradient Boosting; plot feature importances
- Compare linear, polynomial, and RBF kernels on a non-linearly separable dataset

**Checkpoint:**
> `week16_trees_svm.ipynb`: From-scratch CART with tree visualization, RF and GBM comparison table, feature importance bar chart, SVM decision boundary plots for all 3 kernels.

---

### Week 17: Phase 3 Capstone — ML Pipeline v1

**Study:**
- ISLP: Chapter 12 (Unsupervised Learning) — PCA, K-Means, Hierarchical Clustering
- Review all previous chapters

**Practice:**
- Full ML pipeline: data → preprocessing → model comparison (5+ model classes) → evaluation → model card
- PCA → K-Means → Hierarchical clustering on a high-dimensional dataset
- Multiple testing correction (Bonferroni, BH-FDR) on a gene expression example

**Checkpoint:**
> `phase3_capstone/`: Complete ML pipeline with proper CV, at least 5 model classes tested, final model card with performance metrics and limitations. GitHub repo with README and reproducible setup.

**M3: ML Ready** — You can build, evaluate, and compare ML models. You understand cross-validation, regularization, and model selection.

</details>

---

<details>
<summary><h2>Phase 4: Applied ML</h2></summary>

| | |
|---|---|
| **Duration** | 5 Weeks |
| **Resources** | [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html), [Kaggle ML Micro-Courses](https://www.kaggle.com/learn) (Intro to ML, Intermediate ML), [StatQuest ML Series](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF), [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) |
| **Depth** | Scikit-Learn User Guide (comprehensive), Kaggle (~5 hrs), StatQuest (~4 hrs), Google MLCC (~15 hrs). Code-first, production-oriented. |

### Week 18: End-to-End ML Projects & Data Pipelines

**Study:**
- Scikit-Learn User Guide:
  - [Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html) → "Linear Models" (Ridge, Lasso, ElasticNet)
  - [Model Selection](https://scikit-learn.org/stable/model_selection.html) → "Cross-validation", "Grid Search"
  - [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline) → `Pipeline`, `ColumnTransformer`, `make_pipeline`
- Kaggle ML Micro-Courses: [Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning), [Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning)
- StatQuest: ML Overview series (~1 hr)

**Practice:**
- Complete the Kaggle "Intro to Machine Learning" course project (Housing dataset)
- Build a scikit-learn Pipeline that handles: imputation → scaling → feature selection → model
- Add unit tests for your pipeline using `pytest`

**Checkpoint:**
> `week18_pipeline/`: Reproducible scikit-learn Pipeline with `pytest` tests. Must include: data loading, preprocessing, model training, evaluation, and model saving/loading.

---

### Week 19: Feature Engineering Masterclass

**Study:**
- Scikit-Learn User Guide:
  - [Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html) → "Classification" (LogisticRegression, SGDClassifier)
  - [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html) → "Stochastic Gradient Descent"
  - [Tree Models](https://scikit-learn.org/stable/modules/tree.html) → DecisionTreeClassifier, DecisionTreeRegressor, `max_depth`, `min_samples_split`
- Google ML Crash Course: [Classification](https://developers.google.com/machine-learning/crash-course/classification) module
- [Google Decision Forests Guide](https://developers.google.com/machine-learning/decision-forests) — Decision trees, random forests, gradient boosted trees
- StatQuest: Decision Trees, Random Forests series (~1.5 hrs)

**Practice:**
- Create 20+ features from a raw dataset (polynomial, interaction, binning, target encoding)
- Feature importance analysis: permutation importance, SHAP values
- Compare model performance with raw features vs. engineered features

**Checkpoint:**
> `week19_feature_eng.ipynb`: 20+ engineered features, feature importance analysis with SHAP summary plot, performance comparison table (raw vs. engineered), and written analysis of which features matter most.

---

### Week 20: Hyperparameter Tuning & Model Interpretability

**Study:**
- Scikit-Learn User Guide:
  - [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html) → `SVC`, `kernel='rbf'`, `C`, `gamma` parameters
  - [Model Selection](https://scikit-learn.org/stable/model_selection.html) → `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`
- SHAP documentation: [shap.readthedocs.io](https://shap.readthedocs.io/) (examples + API)
- StatQuest: SVM, Hyperparameter Tuning series (~1.5 hrs)

**Practice:**
- Hyperparameter tuning shootout: GridSearchCV vs. RandomizedSearchCV vs. Optuna on 3 models
- Train a complex model (GBM); apply SHAP values, PDP, and LIME
- Produce: SHAP summary, SHAP waterfall for 3 individual predictions, PDP for top 3 features

**Checkpoint:**
> `week20_tuning_interpret.ipynb`: Tuning comparison table (3 strategies × 3 models), full interpretability analysis with SHAP/LIME/PDP, and written section on when each method should be preferred.

---

### Week 21: Imbalanced Learning & Ensemble Techniques

**Study:**
- Scikit-Learn User Guide:
  - [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) → `RandomForestClassifier`, `GradientBoostingClassifier`, `VotingClassifier`, `StackingClassifier`
  - [Decomposition](https://scikit-learn.org/stable/modules/decomposition.html) → `PCA`, `TruncatedSVD`
- StatQuest: Boosting, PCA, t-SNE series (~2 hrs)

**Practice:**
- Handle an imbalanced dataset: SMOTE, class weights, threshold tuning — compare AUC-PR
- Build a stacking ensemble with 3 base learners and a meta-learner
- Apply PCA + UMAP to a high-dimensional dataset; visualize clusters

**Checkpoint:**
> `week21_imbalanced_ensemble.ipynb`: Imbalanced learning comparison (4 strategies), stacking ensemble with out-of-fold predictions, PCA + UMAP visualization with cluster analysis.

---

### Week 22: Phase 4 Capstone — Production-Ready ML

**Study:**
- Scikit-Learn User Guide:
  - [Clustering](https://scikit-learn.org/stable/modules/clustering.html) → `KMeans`, `DBSCAN`, `AgglomerativeClustering`
  - [Manifold Learning](https://scikit-learn.org/stable/modules/manifold.html) → `UMAP` (via `umap-learn`), `t-SNE`
  - Review: [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline) for capstone
- SHAP docs: Counterfactual explanations, global surrogate models

**Practice:**
- End-to-end production-ready ML project:
  - Data pipeline with scikit-learn Pipeline
  - Feature engineering (20+ features)
  - Hyperparameter tuning (Optuna or GridSearchCV)
  - Model interpretability (SHAP + PDP)
  - Model card with performance metrics, limitations, and ethical considerations

**Checkpoint:**
> `phase4_capstone/`: GitHub repo with modular Python code, `requirements.txt`, `pytest` tests, model card, and README. Must reproduce with a single `python run.py` command.

**M4: Applied ML Ready** — You can build production-ready ML pipelines with feature engineering, tuning, and interpretability.

</details>

---

<details>
<summary><h2>Phase 5: Deep Learning</h2></summary>

| | |
|---|---|
| **Duration** | 8 Weeks |
| **Resources** | [Dive into Deep Learning (D2L.ai)](https://d2l.ai/), [Understanding Deep Learning — Simon Prince](https://udlbook.github.io/udlbook/) |
| **Depth** | D2L.ai (interactive notebooks, continuously updated), Prince ~600pp (modern, visually rich, free online). At 10–12 hrs/week: D2L notebooks + Prince chapters. |

### Week 23: Neural Networks Foundations

**Study:**
- Prince: Chapters 1–5 (Introduction, Supervised learning, Shallow neural networks, Deep neural networks, Loss functions)
- D2L: Chapters 3–5 (MLPs, backpropagation, numerical stability)

**Practice:**
- Implement a 2-layer neural network with backpropagation from scratch using only NumPy
- Gradient checking: verify your backprop implementation
- Train on a classification dataset; compare to `sklearn.neural_network.MLPClassifier`

**Checkpoint:**
> `week23_nn_scratch.ipynb`: From-scratch neural network with gradient checking (must pass), training/validation loss curves, and accuracy comparison to sklearn MLP.

---

### Week 24: Optimization, Initialization & Regularization

**Study:**
- Prince: Chapters 6–9 (Training, Optimizers, Initialization, Regularization)
- D2L: Chapter 6 (Builders' Guide — optimization deep-dive)

**Practice:**
- Implement and compare SGD, Momentum, RMSProp, and Adam optimizers from scratch
- Demonstrate initialization effects: Xavier vs. He vs. random on training dynamics
- Ablation study: Dropout, L2 regularization, early stopping — record performance

**Checkpoint:**
> `week24_optimization.ipynb`: 4 optimizer implementations from scratch, side-by-side convergence plots, initialization experiment with training loss curves, ablation table (8 combinations).

---

### Week 25: Convolutional Neural Networks

**Study:**
- Prince: Chapters 10–11 (Convolutional networks, Residual networks)
- D2L: Chapters 7–8 (CNNs, Modern CNNs — AlexNet, VGG, ResNet)

**Practice:**
- Implement a 2D convolution operation from scratch using only NumPy
- Build and train a ResNet-18 (from `torchvision.models`) on CIFAR-10; fine-tune to >90% accuracy
- Grad-CAM visualization: show what the network "sees" for 5 sample predictions

**Checkpoint:**
> `week25_cnn.ipynb`: From-scratch 2D convolution, ResNet-18 training with final test accuracy and learning curves, Grad-CAM visualizations for 5 samples, and feature map visualizations for 3 layers.

---

### Week 26: RNNs, LSTMs & Sequence Models

**Study:**
- Prince: Chapter 12 (Recurrent neural networks)
- D2L: Chapter 9 (RNNs) — all sections

**Practice:**
- Implement an LSTM cell from scratch using only `torch.Tensor` operations (no `nn.LSTM`)
- Build a character-level language model; generate text samples after training
- Compare LSTM vs. GRU on a sequence prediction task

**Checkpoint:**
> `week26_rnn_lstm.ipynb`: From-scratch LSTM cell verified against `nn.LSTM`, character LM with generated text samples at epoch 1, 10, 50, and final perplexity metric.

---

### Week 27: Attention Mechanism & Transformer Architecture

**Study:**
- Prince: Chapter 13 (Transformers)
- D2L: Chapter 11 (Attention Mechanisms)
- Read: [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

**Practice:**
- Implement scaled dot-product attention from scratch
- Implement multi-head attention from scratch
- Implement a full Transformer encoder block (no `nn.TransformerEncoder`)

**Checkpoint:**
> `week27_attention.ipynb`: From-scratch attention mechanisms, Transformer encoder block tested on a simple sequence task, annotated diagram matching Illustrated Transformer's visualization. **This is the most important from-scratch implementation in Phase 5.**

---

### Week 28: Transfer Learning & Modern Architectures

**Study:**
- Prince: Chapters 14–16 (Transfer learning, Modern architectures, Self-supervised learning)
- D2L: Chapter 14 (Computer Vision — Fine-Tuning)

**Practice:**
- Fine-tune a pre-trained ResNet-50 on a custom 5-class image dataset
- Compare: full fine-tuning vs. head-only vs. last-2-layers; plot accuracy vs. epochs
- Apply to a text classification task using HuggingFace Transformers (pre-trained BERT)

**Checkpoint:**
> `week28_transfer.ipynb`: 3-strategy fine-tuning comparison table, Grad-CAM visualizations, HuggingFace BERT fine-tuning on text classification with evaluation metrics.

---

### Week 29: Generative Models — VAEs & GANs

**Study:**
- Prince: Chapters 17–18 (Autoencoders, Generative adversarial networks)
- D2L: Chapter 20 (GANs)

**Practice:**
- Implement a Variational Autoencoder (VAE) from scratch; train on MNIST
- Visualize latent space with t-SNE; generate new samples by sampling from latent space
- Implement a DC-GAN; train on CIFAR-10; generate sample grid

**Checkpoint:**
> `week29_generative.ipynb`: VAE with latent space t-SNE visualization, image reconstructions, DC-GAN with generated samples grid, and written comparison of VAE vs. GAN strengths/weaknesses.

---

### Week 30: Phase 5 Capstone — DL + LLM Project

**Study:**
- Review all Prince chapters and D2L notebooks from this phase
- HuggingFace Course: Chapters 1–3 (Transformers overview)

**Practice:**
- End-to-end deep learning project combining CNN/Transformer with LLM:
  - Option A: Image captioning system (CNN encoder + LLM decoder)
  - Option B: Document understanding pipeline (OCR + LLM summarization)
  - Option C: Multimodal search (image + text embeddings)

**Checkpoint:**
> `phase5_capstone/`: Complete DL project with trained model, evaluation report, and demo (Gradio or Streamlit). Upload model to HuggingFace Hub. Include model card with architecture, training details, and limitations.

**M5: DL Ready** — You understand neural network architectures, can train CNNs/RNNs/Transformers, and can apply transfer learning.

</details>

---

<details>
<summary><h2>Phase 5.5: LLM/GenAI Engineering</h2></summary>

| | |
|---|---|
| **Duration** | 6 Weeks |
| **Resources** | [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course), [LangChain](https://python.langchain.com/), [RAGAS](https://docs.ragas.io/), [DeepLearning.AI Short Courses](https://www.deeplearning.ai/short-courses/) |
| **Depth** | Most critical new module for 2026 job readiness (65%+ of DS postings require LLM/GenAI). All resources free. At 10–12 hrs/week: ~80 hours total. |

### Week 31: LLM Fundamentals — Transformers, Tokenization & Scaling

**Study:**
- HuggingFace NLP Course: Chapters 1–3 (Transformers, using pipelines, processing data)
- Read: "Attention Is All You Need" (Vaswani et al., 2017) — architecture sections
- Karpathy: ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) (2 hrs)

**Practice:**
- Implement scaled dot-product attention from scratch (review from Week 27, now in LLM context)
- Compare tokenizers: BPE (tiktoken), WordPiece, SentencePiece — tokenize the same text, compare outputs
- Experiment with OpenAI/Anthropic API: zero-shot, few-shot, chain-of-thought prompts

**Checkpoint:**
> `week31_llm_fundamentals.ipynb`: Attention mechanism from-scratch, tokenizer comparison notebook (same text, 3 tokenizers, token count + output comparison), and a prompt experimentation log with 10+ prompts across 3 prompt patterns.

---

### Week 32: Prompt Engineering & LLM APIs

**Study:**
- [OpenAI Cookbook](https://cookbook.openai.com/) — prompt patterns, function calling, structured output
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)
- DeepLearning.AI: "Prompt Engineering for Developers" (short course)

**Practice:**
- Build a production-quality prompt library: 20+ prompt patterns (zero-shot, few-shot, CoT, ReAct, system prompts)
- Implement structured output parsing (JSON mode, function calling) with OpenAI/Anthropic API
- Cost optimization: token counting, caching, rate limiting

**Checkpoint:**
> `week32_prompt_engineering/`: Prompt library with 20+ patterns, each with: prompt template, example input/output, cost estimate, and best-use-case documentation. API integration project with structured output parsing and error handling.

---

### Week 33: RAG Architecture — Embeddings, Vector DBs & Retrieval

**Study:**
- LangChain: [RAG tutorials](https://python.langchain.com/docs/use_cases/question_answering/)
- "RAG From Scratch" (LangChain YouTube series)
- Embedding models overview: text-embedding-3, bge, e5

**Practice:**
- Build a full RAG pipeline:
  - Document ingestion and chunking (fixed, semantic, recursive — compare)
  - Embedding generation and storage in ChromaDB (local)
  - Retrieval with similarity search
  - LLM-powered Q&A with retrieved context
- Compare retrieval strategies: dense search, hybrid search (BM25 + dense), reranking

**Checkpoint:**
> `week33_rag_pipeline/`: Full RAG application with ChromaDB, 3 chunking strategies compared (retrieval accuracy, latency), hybrid search implementation, and Streamlit demo interface. This is a portfolio-worthy project.

---

### Week 34: Fine-tuning LLMs — LoRA, QLoRA & PEFT

**Study:**
- HuggingFace: ["Fine-tuning LLMs" guide](https://huggingface.co/docs/transformers/training)
- Unsloth tutorials (free tier: Colab/Kaggle)
- Read: "QLoRA" paper (Dettmers et al., 2023) — key sections

**Practice:**
- Prepare a custom instruction-tuning dataset (Alpaca format, 500+ examples)
- Fine-tune a base model (Llama-3-8B or Mistral-7B) using LoRA/QLoRA via PEFT
- Evaluate: perplexity, human evaluation, benchmark comparison (before vs. after fine-tuning)
- Export to GGUF format; test with Ollama

**Checkpoint:**
> `week34_finetuning/`: Fine-tuned model on custom dataset with benchmark report (pre vs. post fine-tuning), training logs, and model uploaded to HuggingFace Hub. Include: dataset description, training configuration, evaluation methodology, and results.

---

### Week 35: LLM Evaluation & Production

**Study:**
- RAGAS documentation: evaluation metrics (faithfulness, answer relevance, context precision)
- LangSmith tutorials: LLM observability, tracing, evaluation
- "Building LLM Apps for Production" — monitoring, guardrails, cost tracking

**Practice:**
- Build an LLM evaluation pipeline using RAGAS:
  - Generate test dataset (questions, ground truth answers, contexts)
  - Evaluate RAG pipeline on faithfulness, answer relevance, context recall
  - Compare 2 retrieval strategies using RAGAS metrics
- Set up monitoring: latency tracking, token usage, error rates
- Implement guardrails: output validation, content filtering

**Checkpoint:**
> `week35_llm_eval/`: Evaluation pipeline with RAGAS metrics dashboard, comparison report of 2 RAG configurations, monitoring setup with latency/token usage tracking, and guardrails implementation.

---

### Week 36: Agentic AI & Phase 5.5 Capstone

**Study:**
- LangGraph tutorials: building agents with LLMs
- CrewAI documentation: multi-agent systems
- Tool use & function calling patterns: ReAct, Plan-and-Execute

**Practice:**
- Build a production LLM application combining:
  - RAG pipeline (from Week 33)
  - Agent with tool use (web search, calculator, database query)
  - Evaluation pipeline (from Week 35)
- Deploy as a web application (Streamlit/Gradio)
- Write a blog post explaining your architecture and decisions

**Checkpoint:**
> `phase5.5_capstone/`: End-to-end LLM application with RAG + Agent + Evaluation, deployed demo URL, blog post, and GitHub repo with full documentation. This is your strongest portfolio piece for 2026 job applications.

**M6: LLM Ready** — You can build, evaluate, and deploy LLM-powered applications. You understand RAG, fine-tuning, prompt engineering, and agentic patterns.

</details>

---

<details>
<summary><h2>Phase 6: Time Series & Causal Inference</h2></summary>

| | |
|---|---|
| **Duration** | 4 Weeks |
| **Resources** | [Causal Inference: The Mixtape](https://mixtape.scunning.com/), [StatsModels Time Series](https://www.statsmodels.org/stable/tsa.html), [Forecasting: Principles & Practice (FPP3)](https://otexts.com/fpp3/), [StatQuest Time Series](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFq--41Vc-MEzP) |
| **Depth** | Mixtape ~600pp (Ch. 1–7, 9), FPP3 ~400pp (Ch. 1–10), StatsModels docs (Python examples), StatQuest (~2 hrs). FPP3 for theory → StatsModels for implementation. |

**FPP3 Chapters:**

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [Ch. 2](https://otexts.com/fpp3/graphics.html) | Time Series Graphics | Time plots, seasonal plots, lag plots |
| [Ch. 3](https://otexts.com/fpp3/decomposition.html) | Decomposition | Trend, seasonality, remainder, STL |
| [Ch. 4](https://otexts.com/fpp3/features.html) | Features | ACF features, STL features, exploration |
| [Ch. 5](https://otexts.com/fpp3/toolbox.html) | Forecaster's Toolbox | Simple methods, residuals, accuracy, CV |
| [Ch. 7](https://otexts.com/fpp3/regression.html) | Time Series Regression | Linear models, predictor selection |
| [Ch. 8](https://otexts.com/fpp3/expsmooth.html) | Exponential Smoothing | SES, Holt's, Holt-Winters, ETS |
| [Ch. 9](https://otexts.com/fpp3/arima.html) | ARIMA Models | Stationarity, differencing, ACF/PACF |
| [Ch. 10](https://otexts.com/fpp3/dynamic.html) | Dynamic Regression | ARIMA + exogenous variables, Fourier |
| [Ch. 12](https://otexts.com/fpp3/advanced.html) | Advanced Forecasting | VAR, Prophet, neural networks |

### Week 37: Time Series — Stationarity, ARIMA & Forecasting

**Study:**
- FPP3: [Ch. 2](https://otexts.com/fpp3/graphics.html) (Time Series Graphics), [Ch. 3](https://otexts.com/fpp3/decomposition.html) (Decomposition), [Ch. 9](https://otexts.com/fpp3/arima.html) (ARIMA Models)
- StatsModels Time Series docs: [ARIMA](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html), [SARIMAX](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_internet.html), [VAR](https://www.statsmodels.org/stable/vector_ar.html) examples
- StatQuest: Time Series Forecasting series (~1 hr)

**Practice:**
- Implement the Augmented Dickey-Fuller (ADF) test from scratch
- Fit ARIMA(p,d,q) models to an economic time series; select orders using AIC/BIC
- Generate 12-step-ahead forecasts with confidence intervals
- Compare ARIMA vs. Prophet on the same dataset

**Checkpoint:**
> `week37_timeseries.ipynb`: From-scratch ADF test vs. `statsmodels`, ARIMA model selection with ACF/PACF plots, 12-step forecast with confidence intervals, ARIMA vs. Prophet comparison table.

---

### Week 38: Causal Inference — DAGs & Potential Outcomes

**Study:**
- Mixtape: Chapters 1–3 (Introduction, Probability review, DAGs)
- Practice drawing DAGs with `graphviz` or `dagitty`

**Practice:**
- Draw DAGs for 3 different research questions
- Identify confounders, mediators, and colliders in each DAG
- Determine the correct adjustment set for each
- Simulate collider bias to demonstrate why controlling for colliders is dangerous

**Checkpoint:**
> `week38_causal_dags.ipynb`: 3 annotated DAGs with confounder/mediator/collider identification, d-separation analysis, and collider bias simulation with before/after effect estimates.

---

### Week 39: Matching, DiD & IV Estimation

**Study:**
- Mixtape: Chapters 4–5 (Potential Outcomes, Matching)
- Mixtape: Chapter 9 (Difference-in-Differences)

**Practice:**
- Implement Propensity Score Matching from scratch; apply to LaLonde (1986) job training dataset
- Implement canonical 2×2 DiD estimator; verify parallel trends visually
- Apply to a real policy change dataset; produce event study plot

**Checkpoint:**
> `week39_matching_did.ipynb`: PSM with love plot (covariate balance before/after), ATE estimate with CI, 2×2 DiD with parallel trends test, event study plot (leads and lags), and written causal interpretation.

---

### Week 40: Phase 6 Capstone — Causal Analysis

**Study:**
- Mixtape: Chapter 7 (Instrumental Variables) — essentials only
- Review all previous weeks

**Practice:**
- Full causal analysis on an observational dataset:
  - Research question → DAG construction → identification strategy
  - Method selection (matching, DiD, or IV) → execution → robustness checks
  - Written report with causal interpretation

**Checkpoint:**
> `phase6_capstone.qmd`: Causal analysis report in Quarto format. Must include: research question, DAG, identification assumption, method, results with confidence intervals, robustness checks, and discussion of assumption plausibility.

**M6.5: Causal Ready** — You can identify causal relationships from observational data and communicate results with appropriate caveats.

</details>

---

<details>
<summary><h2>Phase 7: MLOps & Production</h2></summary>

| | |
|---|---|
| **Duration** | 6 Weeks |
| **Resources** | [MLOps Zoomcamp — DataTalksClub](https://github.com/DataTalksClub/mlops-zoomcamp), [Machine Learning Systems — mlsysbook.ai](https://mlsysbook.ai/) (Ch. 1–7) |
| **Depth** | MLOps Zoomcamp (9 modules, hands-on), mlsysbook (Ch. 1–7). Infrastructure-heavy, very practical. |

### Week 41: Docker & Containerized ML

**Study:**
- MLOps Zoomcamp: Module 1 (Introduction, MLOps maturity)
- mlsysbook: Chapters 1–2 (ML lifecycle overview)

**Practice:**
- Containerize your Phase 4 Applied ML project in Docker
- Write `Dockerfile`, `docker-compose.yml`, and `Makefile` with targets for `train`, `predict`, `test`
- Build a FastAPI model serving endpoint

**Checkpoint:**
> `week41_docker/`: Fully containerized ML project. `docker-compose up` must reproduce training, evaluation, and inference. FastAPI `/predict` endpoint with request validation (Pydantic) and health check.

---

### Week 42: Experiment Tracking & Model Registry

**Study:**
- MLOps Zoomcamp: Module 2 (Experiment Tracking — MLflow)
- mlsysbook: Chapter 4 (Model Training)

**Practice:**
- Integrate MLflow into your Phase 5 Deep Learning project
- Log all hyperparameters, metrics, and artifacts for 20 runs
- Register the best model in MLflow Model Registry

**Checkpoint:**
> `week42_mlflow/`: MLflow-instrumented training script with 20 hyperparameter configurations, registered model in Model Registry, and MLflow UI screenshot showing run comparison.

---

### Week 43: ML Pipelines & Orchestration

**Study:**
- MLOps Zoomcamp: Module 3 (Orchestration — Prefect or Mage)
- mlsysbook: Chapter 5 (Model Deployment)

**Practice:**
- Build an end-to-end ML pipeline using Prefect: data ingestion → feature engineering → training → evaluation → model registration
- Pipeline must be schedulable and handle failures with retries
- Add logging and alerting

**Checkpoint:**
> `week43_pipeline/`: Working Prefect pipeline with 5+ tasks, retry logic, and logging. Must complete a full run end-to-end without manual intervention. Include pipeline visualization from Prefect UI.

---

### Week 44: Model Monitoring & Drift Detection

**Study:**
- MLOps Zoomcamp: Module 5 (Model Monitoring)
- mlsysbook: Chapter 7 (Model Monitoring)

**Practice:**
- Set up Evidently to monitor prediction drift and data drift
- Simulate distribution shift over 5 time periods
- Trigger automated alert when drift exceeds threshold
- Build a monitoring dashboard (Streamlit or Grafana)

**Checkpoint:**
> `week44_monitoring/`: Monitoring pipeline with Evidently reports for 5 time periods showing drift progression, automated alert system, and dashboard with key metrics (accuracy, drift score, latency).

---

### Week 45: CI/CD for ML & Cloud Deployment

**Study:**
- mlsysbook: Chapters 9–10 (MLOps Tooling, Production Infrastructure)
- AWS SageMaker / GCP Vertex AI overview (choose one cloud provider)

**Practice:**
- Set up GitHub Actions CI/CD pipeline: `pytest` → Docker build → MLflow training → deploy to staging
- Deploy your model to a cloud service (AWS SageMaker, GCP Vertex AI, or Render/Railway)
- Add data quality checks (Great Expectations basics)

**Checkpoint:**
> `week45_cicd/`: GitHub repository with CI/CD workflow file, Dockerfile, passing CI/CD run log, cloud-deployed model endpoint URL, and data quality check configuration. Pipeline must complete in under 15 minutes.

---

### Week 46: Phase 7 Capstone — Production ML System

**Study:**
- Review all MLOps Zoomcamp modules
- mlsysbook: final chapters (review)

**Practice:**
- Build a production-ready ML system end-to-end:
  - Data ingestion (scheduled via Prefect)
  - Automated training pipeline with MLflow tracking
  - FastAPI serving with monitoring
  - CI/CD via GitHub Actions
  - Data quality checks

**Checkpoint:**
> `phase7_capstone/`: Public GitHub repository with complete documentation, system architecture diagram, and 2-minute demo video. Must be deployable from scratch with a single `make deploy` command.

**M7: Production Ready** — You can build, deploy, monitor, and maintain production ML systems.

</details>

---

<details>
<summary><h2>Phase 8: Capstone & Portfolio</h2></summary>

| | |
|---|---|
| **Duration** | 8 Weeks |
| **Resources** | Real-world datasets, [Kaggle](https://www.kaggle.com/), [HuggingFace Datasets](https://huggingface.co/datasets), industry benchmarks |
| **Depth** | Synthesis phase — no new textbooks. Apply everything to build an original, end-to-end project demonstrating job readiness. |

### Week 47: Capstone Proposal & Data Collection

**Study:**
- Review industry job postings for your target role
- Study 3–5 successful DS portfolios on GitHub

**Practice:**
- Define your capstone project: original problem, dataset, methodology, expected deliverables
- Collect and validate your dataset
- Write a project proposal (1–2 pages) with: problem statement, data description, methodology, success criteria, timeline

**Checkpoint:**
> `capstone/PROPOSAL.md`: Project proposal with problem statement, dataset description, methodology plan, success criteria, and timeline. Dataset validated and loaded.

---

### Week 48–49: Build — Core Implementation

**Practice:**
- Data pipeline: ingestion, cleaning, feature engineering
- Model development: train, evaluate, iterate
- If applicable: LLM integration (RAG, fine-tuning, or agentic patterns)
- Version control everything: data, code, models, experiments

**Checkpoint:**
> `capstone/` with working pipeline: data → features → model → evaluation. MLflow tracking with 10+ runs. Mid-point review: are you on track for Week 52 delivery?

---

### Week 50: Build — Production & Deployment

**Practice:**
- Containerize your solution (Docker)
- Deploy as a web service (FastAPI + Streamlit/Gradio)
- Add monitoring and logging
- Write comprehensive documentation

**Checkpoint:**
> Deployed capstone application with live URL, Docker configuration, and API documentation.

---

### Week 51: Portfolio Polish

**Practice:**
- Optimize GitHub profile: README, pinned repositories, activity graph
- Write READMEs for all 10 portfolio projects
- Create demo videos (2 minutes each) for top 3 projects
- Optimize LinkedIn profile with project links and skills

**Checkpoint:**
> GitHub profile with 10+ projects, each with professional README. 3 demo videos. LinkedIn profile optimized with DS/AI keywords and project links.

---

### Week 52: Interview Preparation & Final Review

**Practice:**
- SQL interview practice (DataLemur, StrataScratch) — 20+ problems
- ML system design mock interviews — 3 scenarios
- Statistics & probability puzzles — 15+ problems
- Behavioral interview prep: STAR method, project discussion

**Checkpoint:**
> `interview_prep/`: SQL solutions (20+), ML system design notes (3 scenarios), statistics cheat sheet, behavioral interview stories (5 STAR-format stories). Mock interview completed with ≥80% score.

**M8: Job Ready** — You have a portfolio of 10+ projects, deployed applications, and interview preparation. You are ready to apply for Data Scientist, ML Engineer, or AI Engineer roles.

</details>

---

## Buffer Weeks

Buffer weeks are **planned recovery periods** — not optional extras. Use them intentionally:

| Buffer Week | Timing | Recommended Use |
|---|---|---|
| **W9** | After Phase 1 | Catch up on SQL/pandas weak spots, extend Phase 1 capstone |
| **W18** | After Phase 3 | Review ISLP labs, extend ML pipeline project |
| **W27** | After Phase 5 | Review DL concepts, extend CNN/Transformer project |
| **W36** | After Phase 5.5 | Extend RAG application, practice prompt engineering |
| **W45** | After Phase 7 | Fix deployment issues, extend monitoring dashboard |
| **W52+** | After Phase 8 | Capstone extension, additional interview prep, rest |

**Buffer Week Decision Guide:**
- Behind schedule? → Catch up on missed checkpoints
- On schedule? → Review weak areas, extend a project, or take a break
- Ahead of schedule? → Start an elective, contribute to open source, or begin job applications

---

<details>
<summary><h2>Elective E1: Bayesian & Advanced Statistics</h2></summary>

| | |
|---|---|
| **Duration** | 4 Weeks (Optional) |
| **Prerequisites** | Phase 2 (Statistics & Probability), Phase 3 (ML Foundations) |
| **Resources** | [Think Bayes, 2nd Ed. — Allen B. Downey](https://allendowney.github.io/ThinkBayes2/), [Flexible Imputation of Missing Data — Stef van Buuren](https://stefvanbuuren.name/fimd/) (Ch. 1–3) |

---

### Week E1.1: Bayesian Inference Basics

**Study:**
- Think Bayes: Chapters 1–4 (Bayes' Theorem, distributions, estimation)

**Practice:**
- Implement the "Cookie Problem," "Monty Hall," and "M&M Problem" using grid approximation
- Build a Bayesian A/B test from scratch for conversion rate comparison

**Checkpoint:**
> `e1_week1_bayesian.ipynb`: Classic Bayesian problems solved with grid approximation, A/B test with posterior distributions, and written explanation of Bayesian vs. frequentist interpretation.

---

### Week E1.2: Bayesian Estimation & PyMC

**Study:**
- Think Bayes: Chapters 5–8 (Odds, mixtures, simulation)
- Install PyMC; read official "Getting Started"

**Practice:**
- Implement hierarchical model for school test scores (8-schools problem) using PyMC
- Use MCMC sampling; diagnose convergence with R-hat and trace plots

**Checkpoint:**
> `e1_week2_pymc.ipynb`: PyMC hierarchical model, trace plots, posterior predictive checks, and written interpretation of partial pooling vs. no-pooling.

---

### Week E1.3: Missing Data Mechanisms

**Study:**
- FIMD: Chapters 1–3 (Introduction, missing data mechanisms, single imputation)
- Understand MCAR, MAR, MNAR

**Practice:**
- Simulate all three missing data mechanisms on a real dataset
- Compare: complete case analysis, mean imputation, regression imputation — assess bias

**Checkpoint:**
> `e1_week3_missing_data.ipynb`: MCAR/MAR/MNAR simulation with empirical bias tables, comparison of 3 imputation strategies, and written explanation of when each mechanism produces biased estimates.

---

### Week E1.4: Multiple Imputation & Capstone

**Study:**
- FIMD: Chapters 4–6 (MICE, analysis of imputed data)
- Think Bayes: Chapters 9–13 (review)

**Practice:**
- Apply multiple imputation (m=20) to a dataset with 30%+ missingness
- Pool results using Rubin's rules
- End-to-end Bayesian + missing data pipeline

**Checkpoint:**
> `e1_capstone.ipynb`: MICE imputation with m=20, Rubin's rules pooling, PyMC model fitted on imputed data, pooled posterior inference. Written 1-page discussion of uncertainty sources.

</details>

---

<details>
<summary><h2>Elective E2: Generative AI & Diffusion Models</h2></summary>

| | |
|---|---|
| **Duration** | 4 Weeks (Optional) |
| **Prerequisites** | Phase 5 (Deep Learning) |
| **Resources** | [MIT Diffusion Course 2026](https://diffusion.csail.mit.edu/2026/index.html) |

---

### Week E2.1: Diffusion Fundamentals

**Study:**
- MIT Diffusion: Lectures 1–3 (Introduction, Denoising Score Matching, DDPM)
- Read: Ho et al. (2020) DDPM paper

**Practice:**
- Implement DDPM from scratch in PyTorch
- Train on MNIST; visualize forward noising and reverse denoising

**Checkpoint:**
> `e2_week1_ddpm.ipynb`: From-scratch DDPM, forward process visualization (T=0 to T=1000), reverse-process sample grid showing progressive denoising.

---

### Week E2.2: Score Matching & SDEs

**Study:**
- MIT Diffusion: Lectures 4–6 (Score Matching, SDEs, Continuous-Time Diffusion)
- Read: Song et al. (2021) SDE paper — Sections 1–4

**Practice:**
- Implement denoising score matching loss from scratch
- Train score network on 2D toy data; visualize learned score field as vector field

**Checkpoint:**
> `e2_week2_score.ipynb`: Score matching implementation, 2D score field visualization (quiver plot), learned vs. true score field comparison.

---

### Week E2.3: Flow Matching

**Study:**
- MIT Diffusion: Lectures 7–9 (Continuous Normalizing Flows, Flow Matching)
- Read: Lipman et al. (2022) Flow Matching paper

**Practice:**
- Implement Conditional Flow Matching using `torchdiffeq` ODE solver
- Train on 2D toy data; visualize flow trajectories

**Checkpoint:**
> `e2_week3_flow.ipynb`: CFM implementation, animated flow trajectory visualization, DDPM vs. CFM comparison on same toy dataset.

---

### Week E2.4: Latent Diffusion & Capstone

**Study:**
- MIT Diffusion: Lectures 10–12 (Latent Diffusion, Classifier-Free Guidance, Stable Diffusion)
- Read: Rombach et al. (2022) LDM paper

**Practice:**
- Implement classifier-free guidance for conditional DDPM on MNIST
- Vary guidance scale (w = 0, 1, 3, 7, 10); observe quality/diversity trade-off
- Train diffusion or flow-matching model on domain of your choice

**Checkpoint:**
> `e2_capstone/`: Complete generative modeling project with technical report (~1500 words): mathematical derivation, architecture decisions, training details, quantitative evaluation (FID or domain-specific metric), limitations section.

</details>

---

## Final Synthesis & Portfolio

Upon completing this roadmap, you will have:

- **10 portfolio projects** across the full data science and AI stack (2 LLM/GenAI)
- **From-scratch implementations** of critical algorithms: OLS, logistic regression, CART, basic neural network, attention mechanism
- **Practical statistics foundation**: hypothesis testing, confidence intervals, Bayesian inference, causal reasoning
- **Production engineering skills**: Docker, MLflow, FastAPI, CI/CD, monitoring
- **LLM/GenAI competency**: RAG, fine-tuning, prompt engineering, evaluation, agentic patterns

### Portfolio Structure (10 Projects)

| # | Project | Phase |
|---|---------|-------|
| 01 | EDA + data quality report | Phase 1 |
| 02 | A/B test + business decision | Phase 2 |
| 03 | ISLP capstone — model comparison | Phase 3 |
| 04 | Feature engineering + SHAP | Phase 4 |
| 05 | CNN/Transformer project | Phase 5 |
| 06 | Production RAG application | Phase 5.5 |
| 07 | Fine-tuned model + benchmark | Phase 5.5 |
| 08 | Observational data causal inference | Phase 6 |
| 09 | Deployed ML system + monitoring | Phase 7 |
| 10 | End-to-end original project | Phase 8 |

### Project Quality Checklist

Each portfolio project should have:
- [ ] Professional README with problem statement, methodology, results
- [ ] Reproducible code (single command to run)
- [ ] Visualizations (minimum 3 per project)
- [ ] Written analysis/interpretation
- [ ] GitHub link (public repository)
- [ ] For deployed projects: live demo URL

### Continuing Education

After completing this roadmap:
- Read recent NeurIPS/ICML/ICLR proceedings in your specialization
- Contribute to open-source (scikit-learn, HuggingFace, LangChain)
- Publish a capstone project as a blog post or technical article
- Join DS/AI communities (Kaggle, HuggingFace Discord, local meetups)
- Stay current: LLM/GenAI field moves fast — review the resource list quarterly

### Job Readiness Timeline

| Milestone | When | What You Can Apply For |
|---|---|---|
| **M1: Data Ready** | Week 5 | Data Analyst, Junior Data Analyst |
| **M3: ML Ready** | Week 17 | Junior Data Scientist, ML Analyst |
| **M4: Applied ML Ready** | Week 22 | Data Scientist (entry-level) |
| **M6: LLM Ready** | Week 35 | AI Engineer, LLM Engineer |
| **M7: Production Ready** | Week 45 | ML Engineer, MLOps Engineer |
| **M8: Job Ready** | Week 52 | Senior Data Scientist, ML Engineer, AI Engineer |

---

> *"The impediment to action advances action. What stands in the way becomes the way."*
> — Marcus Aurelius
>
> Start with Week 1. The rest will follow.

---

*Roadmap Version 2.0 (Optimized 2026) | Designed for 10–12 hrs/week | Total: ~52 weeks core + 6 buffer + 8 elective*
