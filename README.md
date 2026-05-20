# PmcArticleLabel

An automated pipeline for classifying PubMed Central (PMC) articles using a vLLM inference server. Each article is labeled as **WHITE**, **BLACK**, or **GRAY** based on its relevance to Human Complementary and Alternative Medicine (CAM) and Human Nutrition.

---

## How It Works

```
Article Server  ──►  ArticleLabelHelper  ──►  vLLM Server
     ▲                  (prompt + parse)            │
     └──────────── update labels ◄──────────────────┘
```

1. Fetch a batch of unlabeled articles from the article management server
2. Build a structured prompt for each article (title + abstract)
3. Send the prompt to a vLLM server and parse the JSON response
4. Map the response to a label: `WHITE=1`, `BLACK=2`, `GRAY=3`, `unknown=0`
5. Push the labeled results back to the server
6. Repeat until interrupted

---

## Labels

| Label | Value | Meaning |
|-------|-------|---------|
| `WHITE` | `1` | Clearly within scope of human CAM or human nutrition |
| `BLACK` | `2` | Clearly outside scope |
| `GRAY`  | `3` | Borderline or ambiguous relevance |
| unknown | `0` | Could not be parsed |

---

## Project Structure

```
PmcArticleLabel/
├── main.py                    # Entry point — runs the full labeling loop
├── ArticleLabelHelper.py      # Prompt builder, vLLM caller, response parser
├── ApiCall.py                 # HTTP client for the article management server
├── CheckLibraries.py          # Validates and auto-installs required packages
├── CheckVLLM.py               # Validates vLLM server, model, and inference
├── requirements.txt           # Python dependencies
├── PmcArticleLabel_Colab.ipynb # Google Colab notebook
└── DTO/
    ├── SimpleArticleLabelDTO.py  # Input: article data from the server
    └── ArticleLlmResponse.py     # Output: labeling result sent back to server
```

---

## Requirements

- Python 3.10+
- A running **article management server** (REST API)
- A running **vLLM server** with a loaded chat model

### Python dependencies

```
openai>=1.0.0
requests>=2.31.0
pydantic>=2.0.0
tqdm>=4.66.0
```

Install manually:

```bash
pip install -r requirements.txt
```

Or let the pipeline install them automatically on first run.

---

## Configuration

Edit the constants at the top of `main.py`:

```python
BATCH_SIZE         = 100                        # Articles per batch
POLL_INTERVAL_SECONDS = 5                       # Seconds to wait when queue is empty
VLLM_BASE_URL      = "http://localhost:8001"    # vLLM server address
MODEL_NAME         = "Qwen/Qwen3.5-4B"          # Model loaded in vLLM
API_BASE_URL       = "http://localhost:8000"    # Article management server
```

---

## Running Locally

```bash
git clone https://github.com/EasternPharma/PmcArticleLabel.git
cd PmcArticleLabel
pip install -r requirements.txt
python main.py
```

On startup the pipeline:
1. Checks all required libraries (auto-installs if missing)
2. Pings the vLLM server, verifies the model is loaded, and runs a smoke test
3. Starts the labeling loop — press **Ctrl+C** to stop

### Running checks independently

```bash
# Check Python libraries only
python CheckLibraries.py

# Check vLLM server and model
python CheckVLLM.py --url http://localhost:8001 --model Qwen/Qwen3.5-4B
```

---

## Running in Google Colab

Open `PmcArticleLabel_Colab.ipynb` in Colab or click below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EasternPharma/PmcArticleLabel/blob/main/PmcArticleLabel_Colab.ipynb)

> **Note:** Colab does not have access to `localhost`. Use [ngrok](https://ngrok.com/) or a similar tunnel to expose your vLLM and article servers to a public URL, then update the configuration cell in the notebook.

The notebook handles:
- Cloning (or pulling) the repo automatically
- Installing all dependencies
- Running all pre-flight checks
- Executing the labeling loop with a visual progress bar

---

## API Contract

### `GET /api/v1/articles/unlabeled`

Fetch a batch of unlabeled articles.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 100 | Number of articles to return |

**Response** — array of:
```json
{
  "PmcId": 12345678,
  "Title": "Article title",
  "AbstractText": "Article abstract..."
}
```

### `PUT /api/v1/articles/labels`

Submit labeling results.

**Request body** — array of:
```json
{
  "PmcId": 12345678,
  "Label": 1,
  "Confidence": 0.92,
  "Reasoning": "This article discusses herbal supplements in human subjects."
}
```
