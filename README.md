# Data Science & AI Mastery Roadmap (2026 Q3 Edition)

> A professional, execution-oriented syllabus for serious practitioners.
>
> **Pace:** 10–12 hrs/week · **Duration:** ~64 weeks core + 6 buffer + 8 elective · **Format:** Self-paced, project-driven

---

## Roadmap Overview

| Phase | Name | Duration |
|-------|------|----------|
| 1 | Data Foundations | 5 weeks |
| 2 | Statistics & Probability | 6 weeks |
| 3 | ML Foundations | 7 weeks |
| 4 | Applied ML | 5 weeks |
| 5 | Deep Learning | 8 weeks |
| 6 | LLM/GenAI Engineering | 7 weeks |
| 7 | Agentic AI & AI Safety | 3 weeks |
| 8 | Econometrics — Regression & Diagnostics | 6 weeks |
| 9 | Advanced Econometrics & Time Series | 5 weeks |
| 10 | MLOps & Production | 6 weeks |
| 11 | Capstone & Portfolio | 6 weeks |
| | **Core Total** | **~64 weeks** |
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
| **W5** | M1: Data Ready — pandas + SQL practical exam (including DuckDB/Polars for large datasets) |
| **W11** | M2: Stats Ready — A/B test analysis with correct interpretation |
| **W17** | M3: ML Ready — first ML model comparison study on Kaggle |

### Buffer Weeks

Every 8 weeks there is a built-in buffer week for catch-up, review, project extension, or rest:

| Range | Buffer Week |
|-------|-------------|
| W1–8 | **W9** |
| W10–17 | **W18** |
| W18–25 | **W26** |
| W26–33 | **W34** |
| W34–41 | **W42** |
| W42–52 | **W53** |
| W53–64 | **Capstone extension** |

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
- **Bonus:** Try [DuckDB](https://duckdb.org/docs/) — run SQL directly on pandas DataFrames with zero setup. For datasets >1GB, DuckDB and [Polars](https://pola.rs/) outperform pandas significantly.

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

**M1: Data Ready** — You can load, clean, analyze, and visualize any tabular dataset. You can write SQL with window functions. You know when to use pandas vs. DuckDB/Polars.

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
| **Duration** | 7 Weeks |
| **Resources** | [An Introduction to Statistical Learning with Python (ISLP)](https://www.statlearning.com/) |
| **Depth** | ISLP ~600pp with Python labs in every chapter. Gold-standard ML textbook. Focus Ch. 2–10, 12. At 10–12 hrs/week: ~85pp + labs per week. Expanded to 7 weeks for sustainable pace. |

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

### Week 14: Resampling Methods

**Study:**
- ISLP: Chapter 5 (Resampling Methods) — full chapter + lab
- Focus: k-fold CV, bootstrap, validation set approach

---

### Week 15: Model Selection & Regularization

**Study:**
- ISLP: Chapter 6 (Model Selection & Regularization) — full chapter + lab
- Focus: Best subset selection, forward/backward stepwise, Ridge, Lasso, PCR, PLS

---

### Week 16: Moving Beyond Linearity

**Study:**
- ISLP: Chapter 7 (Moving Beyond Linearity) — polynomial regression, step functions, splines, GAMs
- ISLP: Chapter 8 (Tree-Based Methods) — CART, Random Forests, Bagging

---

### Week 17: Support Vector Machines

**Study:**
- ISLP: Chapter 9 (Support Vector Machines) — full chapter + lab
- Topics: Maximal margin classifier, support vector classifier, SVM (linear, RBF), SVMs with more than 2 classes

---

### Week 18: Phase 3 Capstone — ML Pipeline v1

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

### Week 19: End-to-End ML Projects & Data Pipelines

**Study:**
- Scikit-Learn User Guide:
  - [Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html) → "Linear Models" (Ridge, Lasso, ElasticNet)
  - [Model Selection](https://scikit-learn.org/stable/model_selection.html) → "Cross-validation", "Grid Search"
  - [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline) → `Pipeline`, `ColumnTransformer`, `make_pipeline`
- Kaggle ML Micro-Courses: [Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning), [Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning)
- StatQuest: ML Overview series (~1 hr)

---

### Week 20: Feature Engineering Masterclass

**Study:**
- Scikit-Learn User Guide:
  - [Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html) → "Classification" (LogisticRegression, SGDClassifier)
  - [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html) → "Stochastic Gradient Descent"
  - [Tree Models](https://scikit-learn.org/stable/modules/tree.html) → DecisionTreeClassifier, DecisionTreeRegressor, `max_depth`, `min_samples_split`
- Google ML Crash Course: [Classification](https://developers.google.com/machine-learning/crash-course/classification) module
- [Google Decision Forests Guide](https://developers.google.com/machine-learning/decision-forests) — Decision trees, random forests, gradient boosted trees
- StatQuest: Decision Trees, Random Forests series (~1.5 hrs)

---

### Week 21: Hyperparameter Tuning & Model Interpretability

**Study:**
- Scikit-Learn User Guide:
  - [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html) → `SVC`, `kernel='rbf'`, `C`, `gamma` parameters
  - [Model Selection](https://scikit-learn.org/stable/model_selection.html) → `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`
- SHAP documentation: [shap.readthedocs.io](https://shap.readthedocs.io/) (examples + API)
- StatQuest: SVM, Hyperparameter Tuning series (~1.5 hrs)

---

### Week 22: Imbalanced Learning & Ensemble Techniques

**Study:**
- Scikit-Learn User Guide:
  - [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) → `RandomForestClassifier`, `GradientBoostingClassifier`, `VotingClassifier`, `StackingClassifier`
  - [Decomposition](https://scikit-learn.org/stable/modules/decomposition.html) → `PCA`, `TruncatedSVD`
- StatQuest: Boosting, PCA, t-SNE series (~2 hrs)

---

### Week 23: Phase 4 Capstone — Production-Ready ML

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

### Week 24: Neural Networks Foundations

**Study:**
- Prince: Chapters 1–5 (Introduction, Supervised learning, Shallow neural networks, Deep neural networks, Loss functions)
- D2L: Chapters 3–5 (MLPs, backpropagation, numerical stability)

---

### Week 25: Optimization, Initialization & Regularization

**Study:**
- Prince: Chapters 6–9 (Training, Optimizers, Initialization, Regularization)
- D2L: Chapter 6 (Builders' Guide — optimization deep-dive)

---

### Week 26: Convolutional Neural Networks

**Study:**
- Prince: Chapters 10–11 (Convolutional networks, Residual networks)
- D2L: Chapters 7–8 (CNNs, Modern CNNs — AlexNet, VGG, ResNet)

---

### Week 27: RNNs, LSTMs & Sequence Models

**Study:**
- Prince: Chapter 12 (Recurrent neural networks)
- D2L: Chapter 9 (RNNs) — all sections

---

### Week 28: Attention Mechanism & Transformer Architecture

**Study:**
- Prince: Chapter 13 (Transformers)
- D2L: Chapter 11 (Attention Mechanisms)
- Read: [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

---

### Week 29: Transfer Learning & Modern Architectures

**Study:**
- Prince: Chapters 14–16 (Transfer learning, Modern architectures, Self-supervised learning)
- D2L: Chapter 14 (Computer Vision — Fine-Tuning)

---

### Week 30: Generative Models — VAEs & GANs

**Study:**
- Prince: Chapters 17–18 (Autoencoders, Generative adversarial networks)
- D2L: Chapter 20 (GANs)

---

### Week 31: Phase 5 Capstone — DL + LLM Project

**Study:**
- Review all Prince chapters and D2L notebooks from this phase
- HuggingFace Course: [Ch. 1 (Transformers, what can they do?)](https://huggingface.co/learn/nlp-course/chapter1/1), [Ch. 2 (Using 🤗 Transformers)](https://huggingface.co/learn/nlp-course/chapter2/1), [Ch. 3 (Fine-tuning a pretrained model)](https://huggingface.co/learn/nlp-course/chapter3/1)

**M5: DL Ready** — You understand neural network architectures, can train CNNs/RNNs/Transformers, and can apply transfer learning.

</details>

---

<details>
<summary><h2>Phase 6: LLM/GenAI Engineering</h2></summary>

| | |
|---|---|
| **Duration** | 7 Weeks |
| **Resources** | [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course), [LangChain](https://python.langchain.com/), [RAGAS](https://docs.ragas.io/), [DeepLearning.AI Short Courses](https://www.deeplearning.ai/short-courses/), [Ollama](https://ollama.com/) |
| **Depth** | Most critical module for 2026 job readiness (70%+ of DS postings). All resources free. Trimmed to 7 weeks (from 8) to balance curriculum — slack moved to Agentic AI and Econometrics. |

### Week 32: LLM Fundamentals — Transformers, Tokenization & Scaling

**Study:**
- HuggingFace NLP Course: Chapters 1–3 (Transformers, using pipelines, processing data)
- Read: "Attention Is All You Need" (Vaswani et al., 2017) — architecture sections
- Karpathy: ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) (2 hrs)

---

### Week 33: Prompt Engineering & LLM APIs

**Study:**
- OpenAI Cookbook — updated recipes:
  - [Prompt generation for instruction-tuned LLMs](https://cookbook.openai.com/examples/prompt_generation_for_instruction_tuned_llms)
  - [How to count tokens with tiktoken](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
  - [Structured outputs](https://cookbook.openai.com/examples/structured_outputs_intro) — JSON mode, function calling
- OpenAI Docs: [Function Calling](https://platform.openai.com/docs/guides/function-calling) — tool use, structured outputs
- Anthropic Prompt Engineering Guide:
  - [Prompt engineering overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) — XML tags, long context tips, structured prompts
- DeepLearning.AI: [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) (Andrew Ng + Isa Fulford, ~1.5 hrs)

---

### Week 34: RAG Architecture — Embeddings, Vector DBs & Retrieval

**Study:**
- LangChain: [Build a Question Answering Application](https://python.langchain.com/docs/use_cases/question_answering/) — RAG pipeline walkthrough
- LangChain YouTube: [RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFJuae29fQZ7OoqVrPHnDqMqVqMqM) — Episodes 1–6
- Vector Databases: Install and experiment with [ChromaDB](https://www.trychroma.com/) — local, open-source vector store
- Embedding models: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — compare text-embedding-3, bge-large, e5-large-v2

---

### Week 35: Advanced RAG — Hybrid Search, Reranking & GraphRAG

**Study:**
- Hybrid Search: Combine sparse (BM25) + dense (embeddings) retrieval. Try LangChain's [Ensemble Retriever](https://python.langchain.com/docs/integrations/retrievers/bm25)
- Reranking: [Cohere Rerank](https://docs.cohere.com/docs/rerank-2) — improve retrieval quality post-hoc
- GraphRAG: [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) — knowledge-graph-based retrieval. Read through the quickstart
- Multimodal RAG: Embed images + text with models like [Jina CLIP](https://jina.ai/news/jina-clip-v2-your-multilingual-multimodal-embedding-model/)

---

### Week 36: Small Language Models & Local Inference

**Study:**
- [Ollama](https://ollama.com/): Install and run local models (Llama 3, Phi-4, Gemma 3, Mistral)
- Quantization: Understand GGUF, GPTQ, AWQ formats. Read: [LLM Compression Overview](https://huggingface.co/docs/transformers/quantization/overview)
- Practical: Run Llama-3.1-8B locally via Ollama, build a lightweight RAG pipeline without API costs
- Use case comparison: When to use API (GPT-4o, Claude) vs. local (Ollama) vs. fine-tuned small model

---

### Week 37: Fine-tuning LLMs — LoRA, QLoRA & PEFT

**Study:**
- HuggingFace: [PEFT documentation](https://huggingface.co/docs/peft/en/index) — LoRA, QLoRA configuration, `get_peft_model`
- HuggingFace: [Fine-tuning LLMs guide](https://huggingface.co/docs/transformers/training) — `Trainer` API, `TrainingArguments`, dataset preparation
- Unsloth: [Fine-tune Llama-3.1 8B (Colab notebook)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb) — step-by-step QLoRA tutorial
- Read: "QLoRA" paper (Dettmers et al., 2023) — Sections 1–3

---

### Week 38: LLM Evaluation, Multimodal AI & Phase 6 Capstone

**Study:**
- RAGAS documentation: [Getting Started](https://docs.ragas.io/en/latest/getstarted/) — faithfulness, answer_relevance, context_precision
- LangSmith Observability: [LangSmith platform](https://docs.langchain.com/langsmith) — trace LLM calls, latency, token usage
- Vision-Language Models: [GPT-4o Vision](https://platform.openai.com/docs/guides/vision), [Claude Vision](https://docs.anthropic.com/en/docs/build-with-claude/vision)
- [HuggingFace Multimodal Models](https://huggingface.co/models?pipeline_tag=image-text-to-text): LLaVA, Qwen-VL

**Hands-on:** Build a multimodal RAG pipeline (PDF with charts → extraction → embedding → Q&A). Set up LangSmith tracing. Evaluate your RAG system with RAGAS metrics.

**M6: LLM Ready** — You can build, evaluate, and deploy LLM-powered applications. You understand advanced RAG, fine-tuning, prompt engineering, small models, and multimodal AI.

</details>

---

<details>
<summary><h2>Phase 7: Agentic AI & AI Safety</h2></summary>

| | |
|---|---|
| **Duration** | 3 Weeks |
| **Resources** | [LangGraph](https://langchain-ai.github.io/langgraph/), [CrewAI](https://docs.crewai.com/), [Anthropic Model Context Protocol (MCP)](https://modelcontextprotocol.io/), [Guardrails AI](https://www.guardrailsai.com/) |
| **Depth** | Agentic workflows are the defining paradigm of 2025–2026. AI Safety is a baseline employer expectation. Expanded to 3 weeks — this is the job-market differentiator. |

### Week 39: Agentic AI — Multi-Agent Systems & Tool Use

**Study:**
- LangGraph: [Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/) — build a tool-calling agent (ReAct pattern)
- LangGraph: [Multi-Agent Collaboration](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/) — supervisor pattern, agent handoff
- CrewAI: [Quickstart](https://docs.crewai.com/quickstart) — role-based multi-agent orchestration
- OpenAI Function Calling: [Tool use with chat models](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models) — ReAct pattern, structured outputs

---

### Week 40: MCP, Protocols & Advanced Agent Patterns

**Study:**
- Anthropic MCP: [Model Context Protocol](https://modelcontextprotocol.io/) — standardized tool/source integration. Build a simple MCP server
- Agent patterns: Router, orchestrator-worker, critic-refine, hierarchical
- Human-in-the-loop: Approval gates, interrupt patterns in LangGraph
- Memory: Conversation memory, vector memory, entity memory in agent systems

---

### Week 41: AI Safety, Guardrails & Responsible AI

**Study:**
- Guardrails: [Guardrails AI](https://www.guardrailsai.com/) — output validation, structure enforcement
- Red-Teaming: Practice adversarial testing on your own RAG/agent systems. Read: [Anthropic Red Teaming Guide](https://docs.anthropic.com/en/docs/test-and-evaluate/red-teaming)
- Bias & Fairness: [HuggingFace Evaluate](https://huggingface.co/docs/evaluate/en/index) — fairness metrics for model outputs
- Content Safety: [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) — programmatic safety boundaries

**Hands-on:** Add guardrails to your Phase 6 RAG application. Red-team it with jailbreak attempts. Build a simple multi-agent workflow with LangGraph.

**M7: Agentic & Safety Ready** — You can build multi-agent systems, implement safety guardrails, and evaluate AI outputs for bias and harm. These skills differentiate you from 80% of LLM practitioners.

</details>

---

<details>
<summary><h2>Phase 8: Econometrics — Regression & Diagnostics</h2></summary>

| | |
|---|---|
| **Duration** | 6 Weeks |
| **Resources** | [Basic Econometrics, 5th Ed. — Gujarati & Porter](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf) (Ch. 1–13), Python: `statsmodels` |
| **Depth** | Gujarati Part I–II (~500pp). Gold-standard econometrics textbook. Sustainable pace: ~80pp/week + Python implementation. Complements ISLP (Phase 3) with rigorous assumption testing and diagnostic framework. |

### Week 42: Introduction & Simple Regression (Ch. 1–3)

**Study:**
- Introduction (I.1–I.4): What is econometrics, methodology, types of data, computer's role
- Ch. 1: The nature of regression analysis — statistical vs. deterministic, regression vs. causation vs. correlation, data types (time series, cross-section, pooled)
- Ch. 2: Two-variable regression — population regression function (PRF), stochastic specification, sample regression function (SRF), the meaning of linear
- Ch. 3: Two-variable OLS estimation — classical linear regression model (CLRM) 10 assumptions, Gauss-Markov theorem, coefficient of determination r², properties of OLS

**Hands-on:** Implement OLS from scratch with NumPy. Verify against `statsmodels.OLS`. Estimate a consumption function with real data.

---

### Week 43: Inference & Hypothesis Testing (Ch. 4–5)

**Study:**
- Ch. 4: Classical Normal Linear Regression Model (CNLRM) — normality assumption, why normality, maximum likelihood estimation (MLE) vs. OLS
- Ch. 5: Two-variable regression inference — confidence intervals (β₁, β₂, σ²), t-test, χ² test, p-values, significance vs. practical significance, ANOVA, prediction intervals

**Hands-on:** Replicate all Chapter 5 examples. Calculate confidence intervals manually + with `statsmodels`. Interpret p-values correctly.

---

### Week 44: Functional Forms & Multiple Regression (Ch. 6–7)

**Study:**
- Ch. 6: Extensions of two-variable regression — regression through origin, scaling/standardization, functional forms (log-linear, semilog, reciprocal, logarithmic reciprocal), Box-Cox transformation, growth rate formulas
- Ch. 7: Multiple regression — partial regression coefficients, R² and adjusted R², specification bias, Cobb-Douglas production function, polynomial regression, partial correlation

**Hands-on:** Test log-linear vs. linear specifications on real wage/income data. Build a Cobb-Douglas production function. Compare R² vs. adjusted R².

---

### Week 45: Multiple Regression Inference & Dummy Variables (Ch. 8–9)

**Study:**
- Ch. 8: Multiple regression inference — individual/overall significance (t and F tests), restricted least squares, Chow test for structural stability, LR/Wald/LM tests
- Ch. 9: Dummy variable regression — ANOVA/ANCOVA models, interaction effects, seasonal analysis, piecewise linear regression, interpretation in semilogarithmic models

**Hands-on:** Build a wage determination model with dummy variables (gender, education, region). Run Chow test for structural breaks. Check for interaction effects.

---

### Week 46: Multicollinearity & Heteroscedasticity (Ch. 10–11)

**Study:**
- Ch. 10: Multicollinearity — nature, theoretical vs. practical consequences, detection (VIF, condition index, auxiliary regressions), ridge regression (intro)
- Ch. 11: Heteroscedasticity — consequences of OLS, detection (Breusch-Pagan, White, Goldfeld-Quandt, Park, Glejser), weighted least squares (WLS), White's heteroscedasticity-consistent standard errors

**Hands-on:** Take your Phase 4 ML project. Run VIF analysis on all features. Test for heteroscedasticity with 3 methods. Compare OLS vs. robust SE vs. WLS results.

---

### Week 47: Autocorrelation & Model Specification (Ch. 12–13)

**Study:**
- Ch. 12: Autocorrelation — nature, OLS consequences, detection (Durbin-Watson d, Breusch-Godfrey LM, runs test), GLS/FGLS correction, Newey-West HAC standard errors, ARCH/GARCH intro
- Ch. 13: Model specification & diagnostic testing — omitted/irrelevant variable bias, RESET test, nested vs. non-nested models, model selection criteria (AIC, SIC, Mallow's Cp), outliers/leverage/influence, Chow prediction failure test

**Hands-on:** Apply full diagnostics to a time series regression. Run RESET, check AIC/SIC across specifications. Test for autocorrelation and correct with Newey-West SE.

**M8: Diagnostics Ready** — You can rigorously test regression assumptions and apply corrections. This bridges ML and econometrics — a rare and highly valued dual competency.

</details>

---

<details>
<summary><h2>Phase 9: Advanced Econometrics & Time Series</h2></summary>

| | |
|---|---|
| **Duration** | 5 Weeks |
| **Resources** | [Basic Econometrics, 5th Ed. — Gujarati & Porter](https://www.cbpbu.ac.in/userfiles/file/2020/STUDY_MAT/ECO/1.pdf) (Ch. 15–17, 21–22), Python: `statsmodels`, `arch` |
| **Depth** | Gujarati Part III–IV (~400pp). Sustainable pace: ~80pp/week + implementation. Covers qualitative response, panel data, dynamic models, and time series econometrics. Builds directly on Phase 8 diagnostics foundation. |

### Week 48: Qualitative Response Models (Ch. 15)

**Study:**
- Ch. 15: Linear Probability Model (LPM) — problems and limitations
- Logit and Probit models — estimation (MLE), interpretation (odds ratios, marginal effects)
- Tobit model (censored regression), Poisson regression (count data)
- Multinomial/ordinal logit and probit extensions

**Hands-on:** Build Logit and Probit models for binary outcomes (loan default, customer churn). Compare LPM vs. Logit marginal effects. Estimate a Tobit model.

---

### Week 49: Panel Data Regression (Ch. 16)

**Study:**
- Ch. 16: Why panel data, pooled OLS (constant coefficients)
- Fixed effects: LSDV model, within-group (WG) estimator, caveats
- Random effects model (REM), Breusch-Pagan LM test
- Hausman test: FE vs. RE, properties comparison, practical guidelines

**Hands-on:** Work with a cross-country panel dataset (World Bank, Penn World Table). Implement FE, RE, and pooled OLS. Run Hausman test to choose specification.

---

### Week 50: Dynamic Econometric Models (Ch. 17)

**Study:**
- Ch. 17: The role of time and lags in economics, reasons for lags
- Distributed-lag models: ad hoc estimation, Koyck transformation (median/mean lag)
- Adaptive expectations model, partial (stock) adjustment model
- Instrumental Variables (IV), Granger causality, Sims causality

**Hands-on:** Estimate Koyck distributed-lag models on macro data. Test Granger causality between money supply and GDP. Apply IV estimation with `statsmodels`.

---

### Week 51: Time Series Econometrics I (Ch. 21)

**Study:**
- Ch. 21: Stationary and nonstationary stochastic processes, trend stationary (TS) vs. difference stationary (DS)
- Unit root stochastic process, spurious regression
- Tests of stationarity: graphical, ACF/correlogram, statistical significance of AC
- Unit root tests: Augmented Dickey-Fuller (ADF), Phillips-Perron (PP)
- Cointegration: Engle-Granger (EG) two-step, Augmented Engle-Granger (AEG), error correction mechanism (ECM)

**Hands-on:** Test unit roots in financial/economic time series (stock prices, GDP, exchange rates). Run ADF tests with proper lag selection. Test cointegration between consumption-income. Build an ECM.

---

### Week 52: Time Series Econometrics II & Capstone (Ch. 22)

**Study:**
- Ch. 22: AR, MA, ARMA, ARIMA modeling, Box-Jenkins methodology (identification → estimation → diagnostic checking → forecasting)
- Vector Autoregression (VAR) — estimation, impulse response functions (IRF), forecast error variance decomposition (FEVD)
- Volatility modeling: ARCH and GARCH models, EGARCH, GARCH-M extensions

**Hands-on:** Fit ARIMA/SARIMA to seasonal data (Box-Jenkins full workflow). Build a VAR model and generate IRF plots. Estimate GARCH(1,1) volatility on financial returns with `arch` package.

**M9: Econometrics Ready** — You can specify, estimate, diagnose, and forecast with the full econometric toolkit. You bridge ML and econometrics — qualifying for quantitative analyst, econometrician, and forecasting roles.

</details>

---

<details>
<summary><h2>Phase 10: MLOps & Production</h2></summary>

| | |
|---|---|
| **Duration** | 6 Weeks |
| **Resources** | [MLOps Zoomcamp — DataTalksClub](https://github.com/DataTalksClub/mlops-zoomcamp), [Machine Learning Systems — mlsysbook.ai](https://mlsysbook.ai/) (Vol I: Foundations, Vol II: At Scale) |
| **Depth** | MLOps Zoomcamp (9 modules, hands-on), mlsysbook (two-volume textbook). Infrastructure-heavy, very practical. Includes LLMOps coverage. |

### Week 53: Docker & Containerized ML

**Study:**
- MLOps Zoomcamp: Module 1 (Introduction, MLOps maturity)
- mlsysbook: Volume I — Chapters 1–2 (ML lifecycle overview, system design)

---

### Week 54: Experiment Tracking & Model Registry

**Study:**
- MLOps Zoomcamp: Module 2 (Experiment Tracking — MLflow)
- mlsysbook: Volume I — Chapter 4 (Model Training)
- **LLMOps:** Set up [LangSmith](https://docs.langchain.com/langsmith) for LLM experiment tracking alongside MLflow for ML experiments

---

### Week 55: ML Pipelines & Orchestration

**Study:**
- MLOps Zoomcamp: Module 3 (Orchestration — Prefect or Mage)
- mlsysbook: Volume I — Chapter 5 (Model Deployment)

---

### Week 56: Model Monitoring & Drift Detection

**Study:**
- MLOps Zoomcamp: Module 5 (Model Monitoring)
- mlsysbook: Volume I — Chapter 7 (Model Monitoring)
- **LLMOps Monitoring:** LLM-specific metrics — hallucination rate, retrieval precision, response latency

---

### Week 57: CI/CD for ML & Cloud Deployment

**Study:**
- mlsysbook: Volume II — [ML Operations](https://mlsysbook.ai/), [Model Serving](https://mlsysbook.ai/) (navigate to Vol II → Ops, Vol I → Model Serving)
- Cloud deployment (choose one):
  - **AWS:** [SageMaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models.html) — deploy pre-trained models
  - **GCP:** [Vertex AI](https://cloud.google.com/vertex-ai) — model registry, online prediction
  - **Render/Railway:** [Deploy FastAPI on Render](https://render.com/docs/deploy-fastapi) — simpler alternative for portfolio projects
- Great Expectations: [Getting Started](https://docs.greatexpectations.io/docs/) — data validation, expectations, checkpoints

---

### Week 58: Phase 10 Capstone — Production ML System

**Study:**
- Review all MLOps Zoomcamp modules (1–5)
- mlsysbook: Production system case studies (Volume II — Case Studies)

**M10: Production Ready** — You can build, deploy, monitor, and maintain production ML and LLM systems.

</details>

---

<details>
<summary><h2>Phase 11: Capstone & Portfolio</h2></summary>

| | |
|---|---|
| **Duration** | 6 Weeks |
| **Resources** | Real-world datasets, [Kaggle](https://www.kaggle.com/), [HuggingFace Datasets](https://huggingface.co/datasets), industry benchmarks |
| **Depth** | Synthesis phase — no new textbooks. Apply everything to build an original, end-to-end project demonstrating job readiness. Trimmed to 6 weeks (from 8) — focused execution. |

### Week 59: Capstone Proposal & Data Collection

**Study:**
- Review industry job postings for your target role
- Study 3–5 successful DS portfolios on GitHub

---

### Week 60–61: Build — Core Implementation

---

### Week 62: Build — Production & Deployment

---

### Week 63: Portfolio Polish & Documentation

---

### Week 64: Interview Preparation & Final Review

**M11: Job Ready** — You have a portfolio of 12+ projects, deployed applications, and interview preparation. You are ready to apply for Data Scientist, ML Engineer, AI Engineer, LLM Engineer, or Quantitative Analyst roles.

</details>

---

## Buffer Weeks

Buffer weeks are **planned recovery periods** — not optional extras. Use them intentionally:

| Buffer Week | Timing | Recommended Use |
|---|---|---|
| **W9** | After Phase 1 | Catch up on SQL/pandas weak spots, extend Phase 1 capstone |
| **W18** | After Phase 2 | Review statistics, extend A/B test project |
| **W26** | After Phase 4 | Review ISLP labs + Scikit-learn, extend ML pipeline |
| **W34** | Mid Phase 6 | Extend RAG application, experiment with small models |
| **W42** | After Phase 7 | Practice agentic patterns, extend safety testing |
| **W53** | After Phase 9 | Review econometrics, extend time series capstone |
| **W64+** | After Phase 11 | Capstone extension, interview prep, rest |

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

- **12 portfolio projects** across the full data science, AI, agentic, and econometrics stack
- **From-scratch implementations** of critical algorithms: OLS, logistic regression, CART, basic neural network, attention mechanism, ARIMA
- **Practical statistics & econometrics**: hypothesis testing, confidence intervals, regression diagnostics (heteroscedasticity, autocorrelation, multicollinearity), time series analysis, panel data
- **Production engineering skills**: Docker, MLflow, FastAPI, CI/CD, monitoring (ML + LLM)
- **LLM/GenAI competency**: Advanced RAG, fine-tuning, prompt engineering, agentic patterns, AI safety
- **Modern data tooling**: pandas, DuckDB, Polars for analytical-scale data

### Portfolio Structure (12 Projects)

| # | Project | Phase |
|---|---------|-------|
| 01 | EDA + data quality report | Phase 1 |
| 02 | A/B test + business decision | Phase 2 |
| 03 | ISLP capstone — model comparison | Phase 3 |
| 04 | Feature engineering + SHAP | Phase 4 |
| 05 | CNN/Transformer project | Phase 5 |
| 06 | Production RAG application | Phase 6 |
| 07 | Fine-tuned model + benchmark | Phase 6 |
| 08 | Multi-agent system (LangGraph/CrewAI) | Phase 7 |
| 09 | Full econometric diagnostics report | Phase 8 |
| 10 | Time series forecasting + volatility modeling | Phase 9 |
| 11 | Deployed ML/LLM system + monitoring | Phase 10 |
| 12 | End-to-end original capstone project | Phase 11 |

### Project Quality Checklist

Each portfolio project should have:
- [ ] Professional README with problem statement, methodology, results
- [ ] Reproducible code (single command to run)
- [ ] Visualizations (minimum 3 per project)
- [ ] Written analysis/interpretation
- [ ] GitHub link (public repository)
- [ ] For deployed projects: live demo URL
- [ ] For LLM projects: evaluation results, safety considerations

### Continuing Education

After completing this roadmap:
- Read recent NeurIPS/ICML/ICLR proceedings in your specialization
- Contribute to open-source (scikit-learn, HuggingFace, LangChain, Ollama)
- Publish a capstone project as a blog post or technical article
- Join DS/AI communities (Kaggle, HuggingFace Discord, local meetups)
- Stay current: LLM/GenAI and agentic fields move fast — review the resource list quarterly
- Watch for: MCP ecosystem growth, new reasoning models, open-source AGI progress

### Job Readiness Timeline

| Milestone | When | What You Can Apply For |
|---|---|---|
| **M1: Data Ready** | Week 5 | Data Analyst, Junior Data Analyst |
| **M3: ML Ready** | Week 18 | Junior Data Scientist, ML Analyst |
| **M4: Applied ML Ready** | Week 23 | Data Scientist (entry-level) |
| **M6: LLM Ready** | Week 38 | AI Engineer, LLM Engineer |
| **M7: Agentic & Safety Ready** | Week 41 | Senior AI Engineer, AI Safety Engineer |
| **M8: Diagnostics Ready** | Week 47 | Data Scientist (econometrics-focused), Quantitative Analyst |
| **M9: Econometrics Ready** | Week 52 | Econometrician, Quantitative Researcher, Forecasting Analyst |
| **M10: Production Ready** | Week 58 | ML Engineer, MLOps Engineer, LLMOps Engineer |
| **M11: Job Ready** | Week 64 | Senior Data Scientist, ML Engineer, AI Engineer, LLM Engineer |

---

> *"The impediment to action advances action. What stands in the way becomes the way."*
> — Marcus Aurelius
>
> Start with Week 1. The rest will follow.

---

*Roadmap Version 4.0 (2026 Q3 Edition — Balanced Track) | Designed for 10–12 hrs/week | Total: ~64 weeks core + 6 buffer + 8 elective*
