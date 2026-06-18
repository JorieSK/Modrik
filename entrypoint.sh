#!/bin/sh
if [ ! -f index/labor_law.index ]; then
    echo "Building FAISS index..."
    python scripts/ingest_labor_law.py
fi
echo "Pre-warming embedding model..."
python -c "from core.vector_store import load_from_disk, _get_model; load_from_disk(); _get_model()"
exec streamlit run app.py --server.address=0.0.0.0
