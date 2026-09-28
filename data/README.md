# Runtime data

The Streamlit application expects the enriched publication table at:

```text
data/df_final3.csv
```

This generated dataset is intentionally not committed to Git.

The analysis notebook documents the data-mining workflow, while `scripts/openalex_enrichment.py` contains the OpenAlex enrichment utility used to add bibliographic metadata.
