# SE4050 Deep Learning Assignment — Project Brief

## Dataset: ApacheJIT

**Source:** Keshavarz & Nagappan (University of Waterloo), MSR 2022. Publicly available on Zenodo (DOI: 10.5281/zenodo.5907002), CC BY 4.0 license.

**What it is:** 106,674 commits from 14 popular Apache open-source projects (Spark, Kafka, Hadoop, Cassandra, HBase, Hive, etc.), each labeled as `buggy` (bug-inducing) or clean.
- 28,239 bug-inducing commits (26%)
- 78,435 clean commits (74%)

## The Real-World Problem (Just-In-Time Defect Prediction)

When a developer commits a code change, that change might silently introduce a bug that surfaces later. Traditional bug detection works at the file level ("this file has bugs somewhere"), which is too coarse to act on.

**Just-In-Time (JIT) defect prediction** flags risky changes at commit time — before the code merges — so a reviewer or CI pipeline can catch it early, while the change is still fresh and cheap to fix. This is a real practice used in industry for code review triage and CI gating.

## How the Labels Were Built (important for your report's "Dataset Description" section)

1. Collected 56,929 fixed bug reports from JIRA across the 14 projects.
2. Matched each bug report to the commit that fixed it (the "fixing commit"), using issue-key references in commit messages.
3. Used the **SZZ algorithm** to trace backward from each fixing commit to the earlier commit(s) that introduced the buggy lines — these are labeled `bug-inducing`.
4. Applied filtering (removed huge commits, non-Java commits, trivial whitespace-only changes, suspicious fix-counts) to reduce label noise.
5. All remaining unflagged commits are labeled `clean`.

**Known limitation (good critical-analysis material):** SZZ is imperfect and introduces some label noise — worth mentioning explicitly in your report's limitations section.

## Why We Chose This Dataset

- **Not from a lecture or tutorial** — satisfies the rubric's originality requirement for dataset selection (5%).
- **Large enough for deep learning** (106K+ commits) — most JIT datasets are far smaller (~37K), which the original paper explicitly identifies as a reason deep learning underperforms on this task elsewhere.
- **Real, well-documented labeling methodology** — not synthetic, not arbitrary.
- **Two distinct data types available** (structured metrics + text), letting us build genuinely different model architectures rather than 4 variations of the same idea.
- **Directly relevant to software engineering / backend development career interests** — commit risk prediction is used in real CI/CD and code review tooling.

## Dataset Columns (verified from actual files)

Each CSV has 18 columns:

| Column | Meaning |
|---|---|
| `commit_id` | Git commit hash — used to join with commit message text |
| `project` | Which Apache repo (e.g. `apache/hbase`, `apache/kafka`) |
| `buggy` | **Label: True = bug-inducing, False = clean** |
| `fix` | Whether this commit is itself a bug-fixing commit |
| `year` | Commit year |
| `author_date` | Unix timestamp of the commit |
| `la` | Lines added |
| `ld` | Lines deleted |
| `nf` | Files touched |
| `nd` | Directories touched |
| `ns` | Subsystems touched |
| `ent` | Change entropy (how spread out the change is across files) |
| `ndev` | Distinct developers who previously touched these files |
| `age` | Average time since last change to touched files |
| `nuc` | Unique changes in files |
| `aexp` | Author experience (overall) |
| `arexp` | Author recent experience |
| `asexp` | Author subsystem experience |

**Not included in the CSV:** the actual code diff or commit message text. We fetch commit messages separately (via `git log` on cloned repos), keyed on `commit_id` + `project`, to build our text-based models.

**Files available (all confirmed present, extracted from the replication zip's `dataset/` folder):**

| File | Rows |
|---|---|
| `apachejit_total.csv` | 106,674 |
| `apachejit_train.csv` (balanced, 2003–2016) | 44,834 |
| `apachejit_test_large.csv` (unbalanced, 2017–2019) | 30,111 |
| `apachejit_test_small.csv` (unbalanced subset, 2017–2019) | 7,526 |

## Models We Suggest (pending confirmation of group-size model count)

All four are **supervised deep learning** — the label (`buggy`) already exists, so the task is binary classification, not unsupervised pattern discovery.

| # | Model | Input | Type | Why This Model / What It Tests |
|---|---|---|---|---|
| 1 | **Deep MLP** (Multi-Layer Perceptron) | 13 structured metrics | Supervised, deep learning | Simplest deep architecture. Tests whether numeric change-characteristics alone (size, entropy, author experience) can predict risk. Acts as our deep-learning baseline. |
| 2 | **1D-CNN on structured features** | 13 structured metrics | Supervised, deep learning | Tests whether local feature interactions / combinations (not just individual metrics) improve prediction — a genuinely different way of processing the same tabular data. |
| 3 | **BiLSTM (with attention)** | Commit message text (token sequence) | Supervised, deep learning | Tests whether the *order and sequence* of words a developer writes about their change (e.g., "quick fix", "refactor", "urgent patch") carries predictive signal. |
| 4 | **Transformer Encoder** | Commit message text (+ optionally fused with structured metrics) | Supervised, deep learning | Tests whether long-range semantic understanding of commit messages outperforms sequential models, and whether combining text + metrics beats either alone. |

**Simple summary of the research question:** *Do commit messages (what developers say about their change) add predictive power beyond commit metrics (what the change actually looks like structurally)? And does model complexity (MLP → CNN → LSTM → Transformer) meaningfully improve results, or does it just cost more compute for similar accuracy?*

This gives you a natural, rubric-friendly critical-analysis angle: comparing not just accuracy, but **generalization, training stability, computational cost, and whether added complexity is justified** — all explicitly required by the marking scheme.

## Evaluation Metrics (rubric requires multiple, not just accuracy)

- Accuracy, Precision, Recall, F1-score (classification is imbalanced — F1 matters more than raw accuracy)
- ROC-AUC
- Confusion matrix
- Training time / inference latency (for the "computational efficiency" comparison)

## Data Split (leakage-safe)

Use the dataset's own recommended structure — all 4 files confirmed present:
- **Train:** `apachejit_train.csv` — balanced set, 2003–2016 (44,834 rows)
- **Test:** `apachejit_test_small.csv` — 2017–2019, unbalanced, real-world scenario (7,526 rows; use this over `_large.csv` for faster iteration given the 2-week timeline)

This is a **chronological split** (train on past, test on future) — the correct leakage-safe approach for this dataset, since it mirrors how the model would actually be used.
