# Runtime embeddings

The Streamlit application expects the trained node embeddings at:

```text
checkpoints/deepgcn_node_embeddings.pt
```

Training checkpoints and intermediate model files are intentionally excluded from the portfolio repository.

The application needs only the final node-embedding tensor used for recommendation.
