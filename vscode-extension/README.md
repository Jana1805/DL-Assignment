# JIT Commit Risk

Shows a bug-risk score in the status bar each time you commit.

The extension computes 12 change metrics from your local git history (lines
changed, files touched, how often and by whom those files changed before, and
the author's experience) and sends those 12 numbers to a prediction API. No
source code, file names or commit messages leave your machine.

## Use

- Commit as usual, from the Source Control view or a terminal. The status bar
  shows `Risk 62%` for the new commit, and a warning appears if it is flagged.
- **JIT Risk: Score staged changes** scores what you are about to commit.
- **JIT Risk: Score last commit** rescores `HEAD`.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `jitRisk.apiUrl` | `http://localhost:8000` | Base URL of the prediction API |
| `jitRisk.apiKey` | empty | Sent as `X-API-Key` if the API requires one |
| `jitRisk.notify` | `risky` | Notify for `risky` commits, `always`, or `never` |
| `jitRisk.timeoutMs` | `8000` | How long to wait for the API |

## Reading the score

The score is a review hint. At the model's threshold, about one in three
flagged commits in the test data was actually bug-inducing. The model was
trained on 14 Apache Java projects, so treat scores on other kinds of
repository as indicative only.

## Develop

```bash
npm install
npm test          # unit tests and a test against a temporary git repository
npm run package   # builds the .vsix
```

Press F5 in this folder to try the extension in a development window.

`npm run calibrate -- <repo> <apachejit_total.csv> <project>` compares the
metrics computed here with the dataset's values for a cloned Apache project.
