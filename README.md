# Graph-Based Scientific Paper Recommendation

**Graph learning and metadata-aware recommendation over a scientific citation network**

This project studies scientific-paper recommendation by combining graph representations, learned node embeddings and bibliographic metadata. It includes exploratory data mining, classical machine-learning baselines, Graph Neural Network experiments and an interactive Streamlit application.

## What the project explores

The experimental workflow compares several approaches:

- classical ML baselines;
- structural graph features;
- PCA / t-SNE analysis;
- Graph Convolutional Networks;
- GraphSAGE;
- Graph Attention Networks;
- learned node embeddings;
- ensemble comparisons.

The recommendation application uses learned graph embeddings together with publication metadata to rank papers related to a user-selected title.

## Recommendation pipeline

```text
Paper / title query
        ↓
Exact or fuzzy matching
        ↓
Graph embedding lookup
        ↓
Cosine similarity
        ↓
Metadata-aware re-ranking
        ↓
Interactive recommendations
```

Ranking can incorporate publication category, shared authors, PageRank, citation thresholds and scientific concepts.

## Interactive application

The Streamlit application is implemented in `streamlit_app.py` and provides:

- GitHub OAuth login;
- exact and fuzzy title search;
- embedding-based similarity;
- configurable ranking weights;
- metadata filters;
- recommendation cards;
- favourites;
- CSV export.

OAuth credentials are read from Streamlit secrets and are never committed.

## Representative results

### Model comparison

![Model comparison](docs/figures/model_comparison.png)

### Citation-network neighbourhood

![Citation subgraph](docs/figures/subgraph_khop.png)

Additional figures retained for documentation cover degree distribution and XGBoost feature importance.

## Data enrichment

`scripts/openalex_enrichment.py` enriches publication identifiers through the OpenAlex API with metadata including:

- title;
- authors;
- institutions;
- concepts;
- citation count;
- publication date.

## Repository structure

```text
Data-Mining-Streamlit/
├── streamlit_app.py
├── notebooks/
│   └── graph_recommendation_analysis.ipynb
├── scripts/
│   └── openalex_enrichment.py
├── data/
│   └── README.md
├── checkpoints/
│   └── README.md
├── docs/
│   └── figures/
├── requirements.txt
├── runtime.txt
└── README.md
```

Large experiment outputs, prediction arrays, training checkpoints, generated HTML visualizations and intermediate datasets are intentionally excluded from the portfolio repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure GitHub OAuth:

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then provide the two runtime artifacts described in:

- `data/README.md`
- `checkpoints/README.md`

Run the app with:

```bash
streamlit run streamlit_app.py
```

## Tech stack

Python · PyTorch · Graph Neural Networks · scikit-learn · Pandas · NumPy · NetworkX · Streamlit · OpenAlex · OAuth · RapidFuzz

## Author

**Paolo Pangallo**  
M.Sc. Computer Engineering — Artificial Intelligence  
University of Calabria
