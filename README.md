# Graph-Based Scientific Paper Recommendation

**A data-mining and graph-learning project built around scientific publication networks**

This project studies scientific-paper recommendation using graph structure, learned node representations and bibliographic metadata. The repository includes data exploration, classical machine-learning baselines, Graph Neural Network experiments and an interactive Streamlit application.

## Project overview

The work uses a scientific citation graph and enriches publication data with metadata such as authors, institutions, concepts, citation counts and publication dates.

The experimental part explores several families of models and representations, including:

- classical machine-learning baselines;
- structural graph features;
- dimensionality reduction and visualization;
- Graph Convolutional Networks;
- GraphSAGE;
- Graph Attention Networks;
- learned node embeddings;
- ensemble comparisons.

The final Streamlit application uses learned embeddings together with metadata-aware scoring to recommend papers similar to a selected publication.

## Recommendation pipeline

```text
Paper / title query
        ↓
Exact or fuzzy title matching
        ↓
Learned node embedding
        ↓
Cosine similarity over candidate papers
        ↓
Metadata-aware re-ranking
        ↓
Interactive recommendations
```

The interface supports additional ranking signals and filters including:

- publication category;
- shared authors;
- PageRank;
- citation threshold;
- scientific concepts.

## Data enrichment

The repository also contains tooling for enriching publication records through **OpenAlex**, collecting fields such as:

- title;
- authors;
- institutions;
- concepts;
- citation count;
- publication date.

## Interactive application

The Streamlit app is implemented in `streamlit_app.py`.

It includes:

- GitHub OAuth login;
- fuzzy paper-title search;
- cosine-similarity recommendation from graph embeddings;
- configurable ranking weights;
- metadata filters;
- recommendation cards and favourites.

## Tech stack

- Python
- PyTorch
- Graph Neural Networks
- scikit-learn
- Pandas / NumPy
- NetworkX
- Streamlit
- OpenAlex API
- OAuth
- RapidFuzz

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The application expects two local artifacts that are intentionally not committed because of their size/data role:

- `df_final3.csv`
- `checkpoints/deepgcn_node_embeddings.pt`

Place them at those paths before launching the app. GitHub OAuth credentials are configured through Streamlit secrets and must not be committed.

## Repository contents

Besides the application, the repository includes notebooks, trained checkpoints, prediction arrays and plots used to compare classical models and GNN variants.

Selected artifacts include:

- GCN / GraphSAGE / GAT checkpoints;
- PCA and t-SNE visualizations;
- model-comparison plots;
- structural graph analyses;
- interactive subgraph visualizations;
- validation and test predictions.

## Author

**Paolo Pangallo**  
M.Sc. candidate in Computer Engineering — Artificial Intelligence  
University of Calabria
