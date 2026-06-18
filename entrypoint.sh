#!/bin/sh
mkdir -p index
if [ ! -f index/labor_law.index ]; then
    echo "Building FAISS index..."
    python -m scripts.ingest_labor_law
fi
echo "Pre-warming embedding model..."
python -c "from core.vector_store import load_from_disk, _get_model; load_from_disk(); _get_model()"
exec streamlit run app.py --server.address=0.0.0.0
