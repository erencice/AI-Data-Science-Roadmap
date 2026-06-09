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

### 🤖 Practice with AI Assistance

As you work through the topics in this roadmap, **use AI tools (ChatGPT, Claude, Gemini, etc.) as an active study partner** to significantly accelerate your learning. AI-assisted practice goes beyond traditional studying by providing **deeper understanding, instant feedback, and personalized learning**.

**How to Use AI for Practice:**

| Method | Description | Example Prompt |
|--------|-------------|----------------|
| **Concept Explanation** | Get difficult topics explained from multiple angles | *"Explain the Central Limit Theorem as if I'm 5 years old, then give me the technical details"* |
| **Quiz & Self-Testing** | Generate quizzes on the week's topics to test yourself | *"Create a 10-question quiz on hypothesis testing, p-values, and confidence intervals"* |
| **Code Review** | Have AI review your code and suggest improvements | *"Review my Python code, explain any bugs, and suggest improvements"* |
| **Problem Solving** | Work through real-world scenario-based problems | *"Give me an A/B testing scenario and walk me through solving it step by step"* |
| **Compare & Contrast** | Deepen understanding by comparing similar concepts | *"Compare t-test vs. z-test in a table — when should I use each?"* |
| **Project Ideas** | Get project suggestions to reinforce what you've learned | *"Suggest 3 projects using real datasets to practice my statistics skills"* |

> **💡 Why This Matters:** Research shows that actively questioning and applying learned knowledge in different contexts can improve long-term retention by up to 40%. AI tools support this process with a 24/7 accessible, patient, and personalized mentor. Instead of asking "explain this to me," prefer active learning prompts like **"quiz me on this topic"** or **"how can I improve this code?"**

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

---

### Week 2: pandas — Loading, Cleaning & First EDA *(Quick Win #1)*

**Study:**
- McKinney: Chapters 5–7 (pandas intro, Data Loading, Data Cleaning)
- Focus: Series, DataFrame, `.loc`/`.iloc`, handling missing values, dtypes

---

### Week 3: pandas — GroupBy, Merging & SQL Foundations

**Study:**
- McKinney: Chapters 8–10 (Wrangling, GroupBy, Aggregation)
- SQLBolt: Lessons 1–10 (SELECT, WHERE, JOINs, aggregates)

---

### Week 4: Advanced SQL & First Quarto Report *(Quick Win #2)*

**Study:**
- SQLBolt: Lessons 11–15 (Window functions, subqueries)
- DataLemur: 10 Easy/Medium SQL problems
- Install Quarto

---

### Week 5: Phase 1 Capstone — End-to-End Analysis

**Study:**
- McKinney: Chapters 11–12 (Time Series, Advanced pandas) — skim for awareness
- Review any weak areas from Weeks 1–4

**M1: Data Ready** — You can load, clean, analyze, and visualize any tabular dataset. You can write SQL with window functions.

</details>

---

<details>
<summary><h2>Phase 2: Statistics & Probability</h2></summary>

| | |
|---|---|
| **Duration** | 6 Weeks |
| **Resources** | [Statistics and Probability Full Course (YouTube)](https://www.youtube.com/watch?v=sbbYntt5CJk) |
| **Depth** | ~11 hours of video lectures + hands-on Python/Jupyter notebook implementations for statistical analysis. |

### Week 6: Introduction to Statistics & Descriptive Data Analysis

**Study:**
- Watch Section 1 & 2 of the video course:
  - Getting started with statistics, data classification, and the statistical study process.
  - Frequency distributions, graphical displays (bar charts, histograms, stem-and-leaf plots), and data visualization analysis.

---

### Week 7: Measures of Central Tendency & Dispersion

**Study:**
- Watch Section 3 of the video course:
  - Measures of center (mean, median, mode).
  - Measures of dispersion (range, variance, standard deviation).
  - Measures of relative position (percentiles, quartiles, z-scores).

---

### Week 8: Probability Foundations & Counting Rules

**Study:**
- Watch Section 4 of the video course:
  - Introduction to probability, sample spaces, and events.
  - Addition and multiplication rules of probability.
  - Permutations, combinations, and counting principles.

---

### Week 9: Probability Distributions (Discrete & Continuous)

**Study:**
- Watch Section 5 of the video course:
  - Discrete probability distributions (Binomial, Poisson, Hypergeometric).
  - Continuous probability distributions (Uniform, Exponential, Normal).
  - The Standard Normal Distribution and z-tables.

---

### Week 10: Sampling Distributions & Inferential Statistics

**Study:**
- Watch Section 6 of the video course:
  - Sampling distributions and the Central Limit Theorem (CLT).
  - Confidence intervals for population means and proportions.
  - Hypothesis testing: z-tests, t-tests, null vs. alternative hypotheses, p-values, Type I and Type II errors.

---

### Week 11: Phase 2 Capstone — Data-Driven Decision (A/B Test Analysis)

**Study:**
- Review the entire course content from the video, focusing on inferential statistics.
- Read about A/B testing design, sample size determination, and statistical power.

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

---

### Week 13: Classification

**Study:**
- ISLP: Chapter 4 (Classification) — full chapter + Python lab
- Topics: Logistic Regression, LDA, QDA, KNN

---

### Week 14: Resampling & Model Selection

**Study:**
- ISLP: Chapters 5–6 (Resampling Methods, Model Selection & Regularization) — full chapters + labs
- Focus: k-fold CV, bootstrap, best subset selection, Ridge, Lasso, PCR

---

### Week 15: Nonlinear Models & Splines

**Study:**
- ISLP: Chapter 7 (Moving Beyond Linearity) — polynomial regression, step functions, splines, GAMs

---

### Week 16: Decision Trees, Ensembles & SVMs

**Study:**
- ISLP: Chapters 8–9 (Tree-Based Methods, SVMs) — full chapters + labs
- Topics: CART, Random Forests, Gradient Boosting, Bagging, SVM (linear, RBF)

---

### Week 17: Phase 3 Capstone — ML Pipeline v1

**Study:**
- ISLP: Chapter 12 (Unsupervised Learning) — PCA, K-Means, Hierarchical Clustering
- Review all previous chapters

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

---

### Week 20: Hyperparameter Tuning & Model Interpretability

**Study:**
- Scikit-Learn User Guide:
  - [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html) → `SVC`, `kernel='rbf'`, `C`, `gamma` parameters
  - [Model Selection](https://scikit-learn.org/stable/model_selection.html) → `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`
- SHAP documentation: [shap.readthedocs.io](https://shap.readthedocs.io/) (examples + API)
- StatQuest: SVM, Hyperparameter Tuning series (~1.5 hrs)

---

### Week 21: Imbalanced Learning & Ensemble Techniques

**Study:**
- Scikit-Learn User Guide:
  - [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) → `RandomForestClassifier`, `GradientBoostingClassifier`, `VotingClassifier`, `StackingClassifier`
  - [Decomposition](https://scikit-learn.org/stable/modules/decomposition.html) → `PCA`, `TruncatedSVD`
- StatQuest: Boosting, PCA, t-SNE series (~2 hrs)

---

### Week 22: Phase 4 Capstone — Production-Ready ML

**Study:**
- Scikit-Learn User Guide:
  - [Clustering](https://scikit-learn.org/stable/modules/clustering.html) → `KMeans`, `DBSCAN`, `AgglomerativeClustering`
  - [Manifold Learning](https://scikit-learn.org/stable/modules/manifold.html) → `UMAP` (via `umap-learn`), `t-SNE`
  - Review: [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline) for capstone
- SHAP docs: Counterfactual explanations, global surrogate models

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

---

### Week 24: Optimization, Initialization & Regularization

**Study:**
- Prince: Chapters 6–9 (Training, Optimizers, Initialization, Regularization)
- D2L: Chapter 6 (Builders' Guide — optimization deep-dive)

---

### Week 25: Convolutional Neural Networks

**Study:**
- Prince: Chapters 10–11 (Convolutional networks, Residual networks)
- D2L: Chapters 7–8 (CNNs, Modern CNNs — AlexNet, VGG, ResNet)

---

### Week 26: RNNs, LSTMs & Sequence Models

**Study:**
- Prince: Chapter 12 (Recurrent neural networks)
- D2L: Chapter 9 (RNNs) — all sections

---

### Week 27: Attention Mechanism & Transformer Architecture

**Study:**
- Prince: Chapter 13 (Transformers)
- D2L: Chapter 11 (Attention Mechanisms)
- Read: [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

---

### Week 28: Transfer Learning & Modern Architectures

**Study:**
- Prince: Chapters 14–16 (Transfer learning, Modern architectures, Self-supervised learning)
- D2L: Chapter 14 (Computer Vision — Fine-Tuning)

---

### Week 29: Generative Models — VAEs & GANs

**Study:**
- Prince: Chapters 17–18 (Autoencoders, Generative adversarial networks)
- D2L: Chapter 20 (GANs)

---

### Week 30: Phase 5 Capstone — DL + LLM Project

**Study:**
- Review all Prince chapters and D2L notebooks from this phase
- HuggingFace Course: [Ch. 1 (Transformers, what can they do?)](https://huggingface.co/learn/nlp-course/chapter1/1), [Ch. 2 (Using 🤗 Transformers)](https://huggingface.co/learn/nlp-course/chapter2/1), [Ch. 3 (Fine-tuning a pretrained model)](https://huggingface.co/learn/nlp-course/chapter3/1)

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

---

### Week 32: Prompt Engineering & LLM APIs

**Study:**
- OpenAI Cookbook:
  - [How to format inputs](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models) — system/user/assistant roles
  - [How to stream completions](https://cookbook.openai.com/examples/how_to_stream_completions) — streaming, token-by-token output
  - [How to count tokens](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) — token estimation, cost tracking
  - [How to fine-tune chat models](https://cookbook.openai.com/examples/how_to_finetune_chat_models) — fine-tuning workflow, dataset format
- OpenAI Docs: [Function Calling](https://platform.openai.com/docs/guides/function-calling) — tool use, structured outputs
- Anthropic Prompt Engineering Guide:
  - [Use XML tags](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags) — XML tags, structure
  - [Long context window tips](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips) — context management
- DeepLearning.AI: [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) (Andrew Ng + Isa Fulford, ~1.5 hrs)

---

### Week 33: RAG Architecture — Embeddings, Vector DBs & Retrieval

**Study:**
- LangChain: [Build a Question Answering Application](https://python.langchain.com/docs/use_cases/question_answering/) — RAG pipeline walkthrough
- LangChain YouTube: [RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFJuae29fQZ7OoqVrPHnDqMqVqMqM) — Episodes 1–6 (loading, splitting, embedding, retrieval, chat)
- Embedding models: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — compare text-embedding-3, bge-large, e5-large-v2

---

### Week 34: Fine-tuning LLMs — LoRA, QLoRA & PEFT

**Study:**
- HuggingFace: [PEFT documentation](https://huggingface.co/docs/peft/en/index) — LoRA, QLoRA configuration, `get_peft_model`
- HuggingFace: [Fine-tuning LLMs guide](https://huggingface.co/docs/transformers/training) — `Trainer` API, `TrainingArguments`, dataset preparation
- Unsloth: [Fine-tune Llama-3 8B (Colab notebook)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb) — step-by-step QLoRA tutorial
- Read: "QLoRA" paper (Dettmers et al., 2023) — Sections 1–3 (introduction, method, results)

---

### Week 35: LLM Evaluation & Production

**Study:**
- RAGAS documentation: [Getting Started](https://docs.ragas.io/en/stable/getstarted/) — faithfulness, answer_relevance, context_precision metrics
- LangSmith: [Tracing tutorial](https://docs.smith.langchain.com/old/cookbook/tracing_faq) — trace LLM calls, latency, token usage
- LangSmith: [Evaluation guide](https://docs.smith.langchain.com/old/cookbook/evaluation_faq) — datasets, evaluators, comparison runs

---

### Week 36: Agentic AI & Phase 5.5 Capstone

**Study:**
- LangGraph: [Build a Tool-Calling Agent](https://langchain-ai.github.io/langgraph/tutorials/introduction/) — ReAct pattern, tool execution, human-in-the-loop
- LangGraph: [Multi-Agent Workflows](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/) — supervisor pattern, agent handoff
- Tool use patterns: [OpenAI function calling](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models) — tool use, structured outputs, ReAct pattern

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

---

### Week 38: Causal Inference — DAGs & Potential Outcomes

**Study:**
- Mixtape: Chapters 1–3 (Introduction, Probability review, DAGs)
- Practice drawing DAGs with `graphviz` or `dagitty`

---

### Week 39: Matching, DiD & IV Estimation

**Study:**
- Mixtape: Chapters 4–5 (Potential Outcomes, Matching)
- Mixtape: Chapter 9 (Difference-in-Differences)

---

### Week 40: Phase 6 Capstone — Causal Analysis

**Study:**
- Mixtape: Chapter 7 (Instrumental Variables) — essentials only
- Review all previous weeks

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

---

### Week 42: Experiment Tracking & Model Registry

**Study:**
- MLOps Zoomcamp: Module 2 (Experiment Tracking — MLflow)
- mlsysbook: Chapter 4 (Model Training)

---

### Week 43: ML Pipelines & Orchestration

**Study:**
- MLOps Zoomcamp: Module 3 (Orchestration — Prefect or Mage)
- mlsysbook: Chapter 5 (Model Deployment)

---

### Week 44: Model Monitoring & Drift Detection

**Study:**
- MLOps Zoomcamp: Module 5 (Model Monitoring)
- mlsysbook: Chapter 7 (Model Monitoring)

---

### Week 45: CI/CD for ML & Cloud Deployment

**Study:**
- mlsysbook: [Ch. 13: ML Operations](https://mlsysbook.ai/book/contents/core/ops/ops.html), [Model Serving](https://mlsysbook.ai/vol1/contents/vol1/model_serving/model_serving.html)
- Cloud deployment (choose one):
  - **AWS:** [SageMaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models.html) — deploy pre-trained models
  - **GCP:** [Vertex AI Model Registry](https://cloud.google.com/vertex-ai/docs/model-registry/introduction) — model versioning, online prediction
  - **Render/Railway:** [Deploy FastAPI on Render](https://render.com/docs/deploy-fastapi) — simpler alternative for portfolio projects
- Great Expectations: [Getting Started tutorial](https://docs.greatexpectations.io/docs/tutorials/getting_started/) — data validation, expectations, checkpoints

---

### Week 46: Phase 7 Capstone — Production ML System

**Study:**
- Review all MLOps Zoomcamp modules (1–5)
- mlsysbook: [Production System Case Studies](https://mlsysbook.ai/book/contents/core/ml_systems/ml_systems.html) — real-world production systems

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

---

### Week 48–49: Build — Core Implementation

---

### Week 50: Build — Production & Deployment

---

### Week 51: Portfolio Polish

---

### Week 52: Interview Preparation & Final Review

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

---

### Week E1.2: Bayesian Estimation & PyMC

**Study:**
- Think Bayes: Chapters 5–8 (Odds, mixtures, simulation)
- Install PyMC; read official "Getting Started"

---

### Week E1.3: Missing Data Mechanisms

**Study:**
- FIMD: Chapters 1–3 (Introduction, missing data mechanisms, single imputation)
- Understand MCAR, MAR, MNAR

---

### Week E1.4: Multiple Imputation & Capstone

**Study:**
- FIMD: Chapters 4–6 (MICE, analysis of imputed data)
- Think Bayes: Chapters 9–13 (review)

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

---

### Week E2.2: Score Matching & SDEs

**Study:**
- MIT Diffusion: Lectures 4–6 (Score Matching, SDEs, Continuous-Time Diffusion)
- Read: Song et al. (2021) SDE paper — Sections 1–4

---

### Week E2.3: Flow Matching

**Study:**
- MIT Diffusion: Lectures 7–9 (Continuous Normalizing Flows, Flow Matching)
- Read: Lipman et al. (2022) Flow Matching paper

---

### Week E2.4: Latent Diffusion & Capstone

**Study:**
- MIT Diffusion: Lectures 10–12 (Latent Diffusion, Classifier-Free Guidance, Stable Diffusion)
- Read: Rombach et al. (2022) LDM paper

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
