# Initial Steps — What We're Going To Do

## Step 1: Confirm with lecturer
- Wait for reply on whether 2-person groups get a reduced model count.
- Until confirmed, plan for 4 models (safer assumption).

## Step 2: Set up the project skeleton
- Create GitHub repository (both members added as collaborators).
- Add `.gitignore`, `requirements.txt`, folder structure:
  ```
  /data
  /notebooks
  /src
  /reports
  ```
- Both members make an initial commit this week (rubric requires weekly commits from each member).

## Step 3: Get the data
- Download `apachejit_total.csv`, `apachejit_train.csv`, `apachejit_test_large.csv` (or `_small.csv`) from Zenodo.
- Clone the 14 Apache project repos (shallow clone, or only needed commit ranges) to extract commit messages via `git log`.
- Merge commit messages into the metrics CSV using the commit identifier as the key.

## Step 4: Exploratory Data Analysis (EDA)
- Class balance (buggy vs clean).
- Distribution of each of the 13 metrics.
- Check for missing values, outliers.
- Basic text stats on commit messages (length, common words).
- This feeds directly into the rubric's "Dataset Description" section (5%).

## Step 5: Preprocessing
- Structured features: normalize/scale numeric columns.
- Text features: tokenize commit messages, build vocabulary, handle length (padding/truncation).
- Confirm chronological train/test split has no leakage (no future data in training).
- Document every preprocessing decision (rubric explicitly checks this, 10%).

## Step 6: Build baseline first
- Train a simple traditional ML baseline (e.g., Logistic Regression) — allowed by rubric, doesn't count toward the 4 required deep models, but gives a sanity-check reference point.

## Step 7: Build the deep learning models (one at a time, not all at once)
- Model 1: Deep MLP (structured metrics)
- Model 2: 1D-CNN (structured metrics)
- Model 3: BiLSTM + attention (commit text)
- Model 4: Transformer Encoder (commit text, possibly fused with metrics)

## Step 8: Evaluate all models under the same conditions
- Same train/test split, same metrics (F1, ROC-AUC, confusion matrix, latency).
- Record training time, parameter count, inference speed for the efficiency comparison.

## Step 9: Critical analysis
- Compare all 4 models: which generalizes best, which overfits, cost vs benefit of complexity.
- Discuss SZZ label noise as a limitation.
- Propose specific, feasible improvements.

## Step 10: Report + GitHub + video
- Write the report following the 10-section structure in the assignment PDF.
- Ensure README.md has setup/run instructions.
- Record the 10-minute demo video.
- Final check: Members.txt, Report.pdf, Turnitin report, Submission.txt (GitHub + YouTube links).

---

**Next immediate action:** wait for lecturer reply, then start Step 2 (repo setup) in parallel — that doesn't depend on the model-count answer.
