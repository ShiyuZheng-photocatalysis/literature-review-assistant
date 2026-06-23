---
title: LitReview Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Literature Review Writing Assistant

综述写作辅助器 — automatically analyze dozens of academic papers to identify shared methods, routine expressions, similar backgrounds, research problems, figure patterns, and unsolved questions.

## Features

- **Shared Method Detection** — Clusters similar methodology paragraphs across papers to show which techniques are commonly used
- **Boilerplate Phrase Detection** — Finds near-identical sentences and templated expressions shared across papers
- **Introduction Clustering** — Groups papers with similar research backgrounds and context
- **Problem Extraction** — Extracts the research problem each paper addresses; clusters similar problems
- **Figure Analysis** — Identifies similar figures by caption; detects templated discussion patterns for common figure types
- **Open Questions Aggregation** — Finds unsolved problems mentioned across papers; highlights converging research gaps

## Quick Start

### Local Installation

```bash
# Clone or download this repository
cd literature-review-assistant

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Usage

1. **Load papers**: Upload PDFs or enter arXiv IDs (e.g., `2301.12345` or `arxiv.org/abs/2301.12345`)
2. **Click "Load Papers"** to extract and segment content
3. **Load at least 2 papers**, then click **"Run Cross-Paper Analysis"**
4. Explore the results across 7 tabs:
   - **Overview** — Summary statistics, section coverage, method-paper intersection
   - **Methods** — Shared method clusters with representative paragraphs
   - **Boilerplate** — Routine expressions grouped by section
   - **Introductions** — Similarity heatmap and background clusters
   - **Problems** — Problem clusters and unique paper focuses
   - **Figures** — Figure type analysis and discussion patterns
   - **Open Questions** — Convergent and unique research gaps
5. **Export** reports as Markdown or HTML

## Deployment (Shareable URL)

### Option 1: Streamlit Community Cloud (Easiest)

1. Push this repository to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account, select the repo, and deploy
4. Share the generated URL with others

### Option 2: Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Set `sdk: docker` (for GPU support) or `sdk: streamlit`
3. Push this repo to the Space
4. Share the Space URL

### Option 3: Local Network

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Others on the same network can access via `http://<your-ip>:8501`.

## Requirements

- Python 3.10+
- First run will download the embedding model (~120MB, cached for subsequent runs)
- For 30+ papers, 4GB+ RAM recommended

## Project Structure

```
literature-review-assistant/
├── app.py                          # Main Streamlit application
├── src/
│   ├── pdf_processor.py            # PDF text extraction + section segmentation
│   ├── arxiv_fetcher.py            # arXiv API client
│   ├── embeddings.py               # Sentence embeddings + TF-IDF
│   ├── utils.py                    # Text cleaning, caching, parallel processing
│   ├── visualizer.py               # Plotly/pyvis visualization helpers
│   ├── report.py                   # Markdown/HTML report generation
│   └── cross_analysis/
│       ├── method_similarity.py    # Shared method clustering
│       ├── boilerplate_finder.py   # Routine expression detection
│       ├── intro_clusterer.py      # Introduction background clustering
│       ├── problem_extractor.py    # Problem statement extraction
│       ├── figure_analyzer.py      # Figure similarity + discussion patterns
│       └── open_questions.py       # Unsolved problem aggregation
├── .streamlit/
│   └── config.toml                 # Streamlit theme & settings
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages for deployment
└── README.md
```

## Limitations

- **Section segmentation** is rule-based; papers with unusual formatting may not segment perfectly
- **Figure similarity** relies primarily on caption text analysis; visual similarity requires GPU
- **Chinese support**: the multilingual embedding model handles both Chinese and English
- **Large corpora** (>50 papers): embedding generation becomes memory-intensive; consider batch processing
- **No GROBID integration** in this version for simpler deployment; can be added for improved structure parsing

## License

MIT
