#!/bin/sh
if [ ! -f labor_law.index ]; then
    echo "Building FAISS index..."
    python ingest_labor_law.py
fi
exec streamlit run app.py --server.address=0.0.0.0
