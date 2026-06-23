"""Literature Review Writing Assistant — Main Streamlit App.

A tool for analyzing dozens of academic papers to identify:
- Shared methods and techniques
- Templated/routine expressions
- Similar introduction backgrounds
- Research problems
- Figure similarity and discussion patterns
- Unsolved problems and open questions

Shareable via Streamlit Cloud or Hugging Face Spaces.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import tempfile
import time

from src.pdf_processor import extract_paper_from_pdf
from src.arxiv_fetcher import parse_arxiv_id, fetch_paper_by_id
from src.embeddings import Embedder
from src.cross_analysis.method_similarity import analyze_method_similarity, analyze_shared_methods_intersection
from src.cross_analysis.boilerplate_finder import find_boilerplate_phrases
from src.cross_analysis.intro_clusterer import cluster_introductions, analyze_background_overlap
from src.cross_analysis.problem_extractor import extract_problems, cluster_problems_across_papers, compare_problem_focus
from src.cross_analysis.figure_analyzer import analyze_figures, find_similar_figure_types
from src.cross_analysis.open_questions import aggregate_open_questions, find_research_gaps
from src.visualizer import (
    similarity_heatmap, method_cluster_chart, presence_heatmap,
    problem_comparison_chart, open_questions_chart, figure_type_chart,
    section_coverage_chart,
)
from src.report import generate_report, generate_html_report

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LitReview Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ─────────────────────────────────────────────────────

if "papers" not in st.session_state:
    st.session_state.papers = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def add_message(msg: str, msg_type: str = "info"):
    st.session_state.messages.append({"text": msg, "type": msg_type})


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("LitReview Assistant")
    st.markdown("综述写作辅助器")
    st.markdown("---")

    st.markdown("### Input Papers")

    # ArXiv ID input
    arxiv_input = st.text_input(
        "arXiv IDs / URLs",
        placeholder="e.g., 2301.12345 or arxiv.org/abs/2301.12345",
        help="Enter one or more arXiv IDs or URLs, separated by commas or newlines.",
    )

    # PDF upload
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files.",
    )

    col1, col2 = st.columns(2)
    with col1:
        load_btn = st.button("Load Papers", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear All", use_container_width=True)

    if clear_btn:
        st.session_state.papers = []
        st.session_state.analysis_done = False
        st.session_state.analysis_results = {}
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Analysis")
    analyze_btn = st.button(
        "Run Cross-Paper Analysis",
        type="primary",
        use_container_width=True,
        disabled=len(st.session_state.papers) < 2,
    )

    if len(st.session_state.papers) < 2:
        st.caption(f"Loaded: {len(st.session_state.papers)} papers (need ≥2 for analysis)")

    st.markdown("---")
    st.markdown("### Export")
    if st.session_state.analysis_done:
        if st.button("Download Markdown Report", use_container_width=True):
            report_md = generate_report(st.session_state.analysis_results)
            st.download_button(
                "Click to Download",
                report_md,
                file_name="literature_review_report.md",
                mime="text/markdown",
            )
        if st.button("Download HTML Report", use_container_width=True):
            report_html = generate_html_report(st.session_state.analysis_results)
            st.download_button(
                "Click to Download",
                report_html,
                file_name="literature_review_report.html",
                mime="text/html",
            )

    st.markdown("---")
    st.caption("Share via Streamlit Cloud or HF Spaces")
    st.caption("https://github.com/streamlit/streamlit")

# ── Main content ──────────────────────────────────────────────────────────

st.title("Literature Review Writing Assistant")
st.markdown("Upload papers, run cross-paper analysis, and discover shared methods, routine expressions, and research gaps.")

# Messages
for msg in st.session_state.messages[-5:]:
    if msg["type"] == "success":
        st.success(msg["text"])
    elif msg["type"] == "warning":
        st.warning(msg["text"])
    elif msg["type"] == "error":
        st.error(msg["text"])
    else:
        st.info(msg["text"])

# ── Handle paper loading ──────────────────────────────────────────────────

if load_btn:
    papers_to_add = []

    # Process arXiv IDs
    if arxiv_input.strip():
        ids = [s.strip() for s in arxiv_input.replace("\n", ",").split(",") if s.strip()]
        with st.spinner(f"Fetching {len(ids)} papers from arXiv..."):
            for source in ids:
                aid = parse_arxiv_id(source)
                if aid:
                    try:
                        meta = fetch_paper_by_id(aid, download_pdf=True)
                        if meta.get("pdf_path"):
                            paper = extract_paper_from_pdf(meta["pdf_path"], source_name=f"arXiv:{aid}")
                            paper.title = meta.get("title", paper.title)
                            paper.authors = meta.get("authors", [])
                            paper.year = meta.get("year")
                            paper.abstract = meta.get("abstract", paper.abstract)
                            if meta["pdf_path"]:
                                Path(meta["pdf_path"]).unlink(missing_ok=True)
                        else:
                            # No PDF, use metadata only
                            from src.pdf_processor import Paper
                            paper = Paper(source=f"arXiv:{aid}")
                            paper.title = meta.get("title", "")
                            paper.authors = meta.get("authors", [])
                            paper.year = meta.get("year")
                            paper.abstract = meta.get("abstract", "")
                            paper.introduction = meta.get("abstract", "")
                            paper.full_text = meta.get("abstract", "")
                        papers_to_add.append(paper)
                        add_message(f"Fetched: {paper.title[:80]}", "success")
                    except Exception as e:
                        add_message(f"Failed to fetch {aid}: {e}", "error")
                else:
                    add_message(f"Could not parse arXiv ID: {source}", "warning")

    # Process uploaded PDFs
    if uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} PDFs..."):
            for uf in uploaded_files:
                try:
                    pdf_bytes = uf.read()
                    paper = extract_paper_from_pdf(pdf_bytes, source_name=uf.name)
                    papers_to_add.append(paper)
                    add_message(f"Loaded: {paper.title[:80]}", "success")
                except Exception as e:
                    add_message(f"Failed to process {uf.name}: {e}", "error")

    if papers_to_add:
        existing_sources = {p.source for p in st.session_state.papers}
        for p in papers_to_add:
            if p.source not in existing_sources:
                st.session_state.papers.append(p)
                existing_sources.add(p.source)

    if st.session_state.papers:
        add_message(f"Total papers loaded: {len(st.session_state.papers)}", "info")
    st.rerun()

# ── Run analysis ──────────────────────────────────────────────────────────

if analyze_btn and len(st.session_state.papers) >= 2:
    papers = st.session_state.papers
    n = len(papers)

    progress = st.progress(0, "Initializing...")
    status = st.empty()

    # Initialize embedder (lazy-loaded, first use will download model)
    if st.session_state.embedder is None:
        status.text("Loading embedding model (first time may download ~120MB)...")
        st.session_state.embedder = Embedder()

    embedder = st.session_state.embedder

    analysis = {}
    analysis["stats"] = {"n_papers": n}
    analysis["papers"] = [
        {
            "title": p.title,
            "source": p.source,
            "authors": p.authors,
            "year": p.year,
            "language": p.language,
        }
        for p in papers
    ]

    # 1. Method similarity
    status.text(f"Analyzing method similarity ({n} papers)...")
    progress.progress(0.1, "Method similarity...")
    analysis["method_similarity"] = analyze_method_similarity(papers, embedder)
    analysis["method_intersection"] = analyze_shared_methods_intersection(papers, embedder)
    progress.progress(0.25)

    # 2. Boilerplate detection
    status.text("Detecting routine expressions...")
    progress.progress(0.35, "Boilerplate detection...")
    analysis["boilerplate"] = find_boilerplate_phrases(papers)
    progress.progress(0.45)

    # 3. Introduction clustering
    status.text("Clustering introduction backgrounds...")
    progress.progress(0.55, "Intro clustering...")
    analysis["intro_clusters"] = cluster_introductions(papers, embedder)
    analysis["background_overlap"] = analyze_background_overlap(papers, embedder)
    progress.progress(0.65)

    # 4. Problem extraction
    status.text("Extracting research problems...")
    progress.progress(0.70, "Problem extraction...")
    analysis["problems"] = cluster_problems_across_papers(papers, embedder)
    analysis["problem_focus"] = compare_problem_focus(papers)
    progress.progress(0.78)

    # 5. Figure analysis
    status.text("Analyzing figures and discussion patterns...")
    progress.progress(0.82, "Figure analysis...")
    analysis["figures"] = analyze_figures(papers, embedder)
    analysis["figure_types"] = find_similar_figure_types(papers, embedder)
    progress.progress(0.90)

    # 6. Open questions
    status.text("Aggregating open questions...")
    progress.progress(0.94, "Open questions...")
    analysis["open_questions"] = aggregate_open_questions(papers, embedder)
    analysis["research_gaps"] = find_research_gaps(papers)
    progress.progress(1.0)

    status.text("Analysis complete!")
    st.session_state.analysis_results = analysis
    st.session_state.analysis_done = True

    add_message(f"Analysis complete: {n} papers processed across 6 dimensions", "success")
    time.sleep(0.5)
    st.rerun()

# ── Display results ────────────────────────────────────────────────────────

if not st.session_state.analysis_done:
    # Show loaded papers
    papers = st.session_state.papers
    if papers:
        st.markdown(f"### Loaded Papers ({len(papers)})")
        data = []
        for p in papers:
            data.append({
                "Title": p.title[:100] if p.title else "(unknown)",
                "Source": p.source[:40],
                "Language": p.language,
                "Sections": sum(1 for s in ["abstract", "introduction", "methods", "results", "discussion", "conclusion"] if p.get_section(s)),
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.markdown("### Get Started")
        st.markdown("""
        1. **Upload PDFs** using the sidebar uploader, or enter **arXiv IDs**
        2. Click **Load Papers** to extract and segment each paper
        3. Load at least 2 papers, then click **Run Cross-Paper Analysis**

        The system will automatically:
        - Extract and segment paper content
        - Find shared methods across papers
        - Detect templated/routine expressions
        - Group papers with similar introductions
        - Identify research problems and gaps
        - Compare figures and their discussion patterns
        - Aggregate unsolved problems
        """)
    st.stop()

# ── Analysis results tabs ─────────────────────────────────────────────────

results = st.session_state.analysis_results
papers = st.session_state.papers

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Methods", "Boilerplate",
    "Introductions", "Problems", "Figures",
    "Open Questions",
])

# ── Tab 1: Overview ───────────────────────────────────────────────────────

with tab1:
    st.header("Analysis Overview")

    stats = results.get("stats", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Papers", stats.get("n_papers", 0))
    with col2:
        ms = results.get("method_similarity", {})
        st.metric("Shared Methods", len(ms.get("clusters", [])))
    with col3:
        bp = results.get("boilerplate", {})
        st.metric("Routine Expressions", bp.get("summary", {}).get("total_groups", 0))
    with col4:
        oq = results.get("open_questions", {})
        st.metric("Convergent Gaps", oq.get("stats", {}).get("convergent", 0))

    st.markdown("---")

    # Section coverage
    st.subheader("Section Extraction Coverage")
    fig = section_coverage_chart(papers)
    st.plotly_chart(fig, use_container_width=True)

    # Method intersection heatmap
    st.markdown("---")
    st.subheader("Method-Paper Intersection")
    mi = results.get("method_intersection", {})
    if mi.get("method_names"):
        fig = presence_heatmap(
            mi["method_names"], mi["paper_labels"], mi["presence_matrix"],
            "Common Methods Detected per Paper"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No common method patterns detected with keyword matching.")

    # Method similarity heatmap
    ms_data = results.get("method_similarity", {})
    if ms_data.get("similarity_matrix") and len(ms_data["similarity_matrix"]) > 1:
        st.subheader("Paper Method Similarity")
        fig = similarity_heatmap(
            ms_data["similarity_matrix"],
            ms_data.get("paper_labels", []),
            "Method Section Similarity"
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Methods ────────────────────────────────────────────────────────

with tab2:
    st.header("Shared Methods & Techniques")

    ms_data = results.get("method_similarity", {})
    clusters = ms_data.get("clusters", [])

    if clusters:
        fig = method_cluster_chart(clusters)
        st.plotly_chart(fig, use_container_width=True)

        for i, c in enumerate(clusters):
            with st.expander(f"{i+1}. {c['label']} ({c['paper_count']} papers)"):
                st.markdown(f"**Key terms**: {', '.join(c['key_terms'][:8])}")
                st.markdown(f"**Papers**: {', '.join(c['papers'])}")
                st.markdown("**Representative paragraphs:**")
                for para in c.get("paragraphs", [])[:5]:
                    st.markdown(f"> *{para['paper_label']}*: {para['text'][:400]}")
                    st.markdown("")
    else:
        st.info("No shared method clusters found. Try adding more papers from the same subfield.")

# ── Tab 3: Boilerplate ────────────────────────────────────────────────────

with tab3:
    st.header("Routine / Templated Expressions")
    st.caption("Near-identical phrases detected across papers, indicating templated writing patterns.")

    bp_data = results.get("boilerplate", {})
    by_section = bp_data.get("by_section", {})

    for section_name in ["introduction", "related_work", "methods", "results", "discussion", "conclusion"]:
        groups = by_section.get(section_name, [])
        if not groups:
            continue
        with st.expander(f"{section_name.title()} ({len(groups)} groups)", expanded=(section_name == "introduction")):
            for g in groups[:10]:
                st.markdown(f"**\"{g['shared_phrase'][:150]}\"** — {g['paper_count']} papers")
                for item in g.get("items", [])[:3]:
                    st.caption(f"[{item['paper_label']}] {item['text'][:250]}")
                st.markdown("---")

    bp_summary = bp_data.get("summary", {})
    if bp_summary.get("total_groups", 0) == 0:
        st.info("No routine expressions detected. This might indicate diverse writing styles or a small corpus.")

# ── Tab 4: Introductions ──────────────────────────────────────────────────

with tab4:
    st.header("Introduction Background Clusters")
    st.caption("Papers grouped by similar introduction backgrounds.")

    intro_data = results.get("intro_clusters", {})

    # Intro similarity heatmap
    if intro_data.get("similarity_matrix") and len(intro_data["similarity_matrix"]) > 1:
        fig = similarity_heatmap(
            intro_data["similarity_matrix"],
            intro_data.get("paper_labels", []),
            "Introduction Section Similarity"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Clusters
    clusters = intro_data.get("clusters", [])
    if clusters:
        for i, c in enumerate(clusters):
            with st.expander(f"Cluster {i+1}: {c['label']} ({c['paper_count']} papers)"):
                st.markdown(f"**Key terms**: {', '.join(c['key_terms'][:8])}")
                st.markdown(f"**Papers**: {', '.join(c['papers'])}")
                if c.get("representative_excerpt"):
                    st.markdown(f"> {c['representative_excerpt'][:500]}")
    else:
        st.info("No introduction clusters found.")

    # Background overlap
    st.markdown("---")
    st.subheader("Sentence-Level Background Overlap")
    overlap = results.get("background_overlap", [])
    if overlap:
        for o in overlap[:10]:
            st.markdown(f"**{o['paper_a']}** ↔ **{o['paper_b']}** (similarity: {o['similarity']})")
            st.caption(f"\"{o['text_a'][:200]}\"")
            st.caption(f"\"{o['text_b'][:200]}\"")
            st.markdown("---")
    else:
        st.info("No sentence-level background overlap found.")

# ── Tab 5: Problems ───────────────────────────────────────────────────────

with tab5:
    st.header("Research Problems")

    problem_data = results.get("problems", {})

    # Per-paper problem counts
    per_paper = problem_data.get("per_paper", [])
    if per_paper:
        fig = problem_comparison_chart(per_paper)
        st.plotly_chart(fig, use_container_width=True)

    # Problem clusters
    clusters = problem_data.get("problem_clusters", [])
    if clusters:
        st.subheader("Cross-Paper Problem Clusters")
        for i, c in enumerate(clusters):
            with st.expander(f"{i+1}. {c['label'][:120]} ({c['paper_count']} papers)"):
                for item in c.get("items", []):
                    st.markdown(f"- **{item['paper_label']}**: {item['text'][:300]}")
    else:
        st.info("No cross-paper problem clusters found.")

    # Problem focus comparison
    st.markdown("---")
    st.subheader("Unique Problem Focus per Paper")
    focus_data = results.get("problem_focus", [])
    if focus_data:
        for f in focus_data:
            terms_str = ", ".join([f"{t[0]} ({t[1]})" for t in f.get("top_terms", [])[:5]])
            st.markdown(f"- **{f['paper_label']}**: {terms_str}")
    else:
        st.info("Run analysis to see unique problem focus per paper.")

# ── Tab 6: Figures ────────────────────────────────────────────────────────

with tab6:
    st.header("Figure Analysis")
    st.caption("Figure similarity based on caption analysis and discussion pattern detection.")

    fig_data = results.get("figures", {})
    fig_types = results.get("figure_types", [])

    # Common figure types
    if fig_types:
        fig = figure_type_chart(fig_types)
        st.plotly_chart(fig, use_container_width=True)

    # Similar figure groups
    groups = fig_data.get("figure_groups", [])
    if groups:
        st.subheader("Similar Figure Groups (by Caption)")
        for g in groups:
            with st.expander(f"{g['label'][:100]} ({g['paper_count']} papers, {g['figure_count']} figures)"):
                for fi in g.get("figures", []):
                    st.markdown(f"- **{fi['paper_label']}** Fig.{fi['figure_index']}: {fi['caption'][:250]}")
    else:
        st.info("No similar figure groups detected.")

    # Discussion patterns
    patterns = fig_data.get("discussion_patterns", [])
    if patterns:
        st.subheader("Templated Figure Discussion Patterns")
        for dp in patterns:
            with st.expander(f"{dp['figure_group'][:80]}"):
                for pat in dp.get("patterns", [])[:5]:
                    st.markdown(f"**Shared phrase**: \"{pat['shared_phrase'][:200]}\"")
                    st.caption(f"Similarity: {pat['similarity']}")
                    st.markdown("---")

    fig_stats = fig_data.get("stats", {})
    if fig_stats:
        st.markdown("---")
        st.caption(f"Total figures: {fig_stats.get('total_figures', 0)} | "
                   f"Figure groups: {fig_stats.get('figure_groups_found', 0)} | "
                   f"Discussion patterns: {fig_stats.get('discussion_patterns_found', 0)}")

# ── Tab 7: Open Questions ─────────────────────────────────────────────────

with tab7:
    st.header("Unsolved Problems & Open Questions")

    oq_data = results.get("open_questions", {})
    convergent = oq_data.get("convergent_questions", [])
    unique = oq_data.get("unique_questions", [])

    # Chart
    if convergent or unique:
        fig = open_questions_chart(convergent, unique)
        st.plotly_chart(fig, use_container_width=True)

    # Convergent (most important)
    if convergent:
        st.subheader("Convergent Open Questions (Identified by ≥2 Papers)")
        for i, q in enumerate(convergent):
            with st.expander(f"{i+1}. {q['summary'][:150]} ({q['paper_count']} papers)"):
                for item in q.get("items", []):
                    st.markdown(f"- **{item['paper_label']}**: {item['text'][:400]}")
                    st.markdown("")
    else:
        st.info("No convergent open questions found. Papers may address different research gaps.")

    # Unique
    if unique:
        st.markdown("---")
        st.subheader("Unique Open Questions (Per-Paper)")
        for i, q in enumerate(unique[:15]):
            st.markdown(f"- **{q['items'][0]['paper_label']}**: {q['summary'][:200]}")

    # Research gaps
    gaps = results.get("research_gaps", [])
    if gaps:
        st.markdown("---")
        st.subheader("Research Gaps Mentioned")
        for g in gaps[:20]:
            st.markdown(f"- **{g['paper_label']}**: {g['text'][:300]}")
