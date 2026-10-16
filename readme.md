# Just-In-Time Software Defect Prediction on ApacheJIT

Predicts whether a Git commit introduces a bug, from structural change metrics
available at commit time. The repository has two parts:

1. **Research notebooks** comparing deep learning models on the ApacheJIT dataset
   (SE4050 Deep Learning Assignment, 2026, SLIIT).
2. **A production path** for the chosen model: a prediction API and a VS Code
   extension that shows a risk score every time you commit.

## Dataset

**ApacheJIT** (Keshavarz & Nagappan, 2022): 106,674 commits from 14 Apache
projects, each with 12 change metrics and a `buggy` label derived with the SZZ
algorithm. Source: https://zenodo.org/records/5907002 (CC BY 4.0). The CSV
files are in `dataset/`.

## Repository layout

```
notebooks/            Preprocessing (v1, v2), Deep MLP (v1, v2), TabNet (v1, v2)
dataset/              ApacheJIT CSV files
jit/                  features.py   v2 feature engineering, shared by training and serving
                      model.py      Deep MLP architecture
                      train.py      trains the model and exports it to ONNX
                      artifacts.py  loads and verifies the exported model
app/main.py           FastAPI service
artifacts/            model.onnx and metadata.json (output of jit.train)
tests/                service tests
vscode-extension/     the VS Code extension
deploy/huggingface/   files for hosting the API on Hugging Face Spaces
config.json           hyperparameters and split boundaries used in the notebooks
```

## Results

All models use a chronological split (train 2003–2014, validation 2015–2016,
test 2017–2019), class weighting, and a decision threshold tuned on validation
data only. Test-set results from the notebooks:

| Model | Features | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Deep MLP | v1 (12 raw) | 0.4778 | 0.7775 | 0.4713 |
| Deep MLP | v2 (17 engineered) | 0.4788 | 0.7908 | 0.4863 |
| TabNet | v1 | 0.4756 | 0.7835 | 0.4530 |
| TabNet | v2 | 0.4743 | 0.7804 | 0.4659 |

The F1 differences are within seed-to-seed noise (about ±0.005 in the MLP seed
sweep). **Deep MLP on v2 features** is the model served in production: it has
the highest ROC-AUC and PR-AUC.

## Production service

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements-dev.txt
```

`requirements.txt` is what the server needs. `requirements-train.txt` adds
PyTorch for training, and `requirements-dev.txt` adds the test tools.

### Train

```bash
python -m jit.train --data dataset/apachejit_total.csv --out artifacts
```

This reproduces `notebooks/Deep_MLP (V2).ipynb` and writes `artifacts/model.onnx`
with the scaler, network and softmax in one graph. The committed model scores
ROC-AUC 0.7910, F1 0.4778, precision 0.3423 and recall 0.7912 on the test set.

### Run

```bash
uvicorn app.main:app --port 8000
# or
docker build -t jit-risk . && docker run -p 8000:8000 jit-risk
```

| Endpoint | Returns |
|---|---|
| `GET /health` | status and model version |
| `POST /predict` | `risk_score`, `is_risky`, `threshold`, `model_version` |
| `GET /docs` | interactive request schema |

`POST /predict` takes the 12 metrics (`la`, `ld`, `nf`, `nd`, `ns`, `ent`,
`ndev`, `age`, `nuc`, `aexp`, `arexp`, `asexp`). Invalid input returns 422. Set
the `API_KEY` environment variable to require an `X-API-Key` header.

### Test

```bash
pytest
```

### Deploy to Hugging Face Spaces

Create a Space with the Docker SDK, clone it, then:

```powershell
.\deploy\huggingface\build_space.ps1 -SpaceDir C:\path\to\your-space
```

Commit and push inside the Space folder. Add `API_KEY` as a Space secret if you
want the endpoint protected.

## VS Code extension

See [vscode-extension/README.md](vscode-extension/README.md). In short:

```bash
cd vscode-extension
npm install
npm run package
code --install-extension jit-commit-risk-0.1.0.vsix
```

Then set `jitRisk.apiUrl` to your API's address.

## How the metrics are computed

The extension computes the 12 metrics with `git`, following what the dataset
actually contains rather than the textbook definitions. These were checked
against 300 real Apache ZooKeeper commits (`npm run calibrate`):

- A directory (`nd`) is identified by its name, not its full path.
- A subsystem (`ns`) is the first path component, so each root-level file
  counts as its own subsystem.
- `ndev` is the number of distinct earlier developers across all touched
  files, divided by the number of files.
- `age` uses whole days since each file's last change, averaged over files.
- `asexp` is the author's earlier commits in the subsystem only for commits
  touching one subsystem. For commits touching several (26% of the dataset)
  the dataset holds an unrelated value between 0 and 2, which cannot be
  reproduced from git. The extension sends the dataset median (0.5) for those
  commits; on the test set this changes ROC-AUC by less than 0.001.

## Limitations

- Precision is about 0.34: roughly two of three flagged commits are fine. Use
  the score as a review hint, not a merge gate.
- The model was trained on Apache Java projects from 2003–2014. Scores on
  other kinds of repository are indicative only.
- SZZ labels are noisy, which caps how well any model can do on this data.

## References

- H. Keshavarz and M. Nagappan, "ApacheJIT: A large dataset for just-in-time
  defect prediction," MSR 2022. https://zenodo.org/records/5907002
- Y. Kamei et al., "A large-scale empirical study of just-in-time quality
  assurance," IEEE Transactions on Software Engineering, 2013.
- J. Śliwerski, T. Zimmermann, and A. Zeller, "When do changes induce fixes?,"
  MSR 2005.
- S. Ö. Arık and T. Pfister, "TabNet: Attentive interpretable tabular
  learning," AAAI 2021.

## AI Assistance Disclosure

Portions of the code, experimental design, and report text were developed with
assistance from Claude (Anthropic). All results were produced by running the
code on the ApacheJIT dataset, and the authors reviewed and take responsibility
for the content.
