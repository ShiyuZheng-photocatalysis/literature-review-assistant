"""Structured report generation for the literature review assistant.

Generates a markdown report summarizing all cross-paper analysis findings.
"""

from datetime import datetime


def generate_report(analysis_results: dict) -> str:
    """Generate a complete markdown report from all analysis results.

    Args:
        analysis_results: dict with keys for each analysis module output.

    Returns markdown string.
    """
    sections = []

    # Header
    sections.append(_report_header(analysis_results))

    # Executive summary
    sections.append(_executive_summary(analysis_results))

    # Methods comparison
    sections.append(_methods_section(analysis_results))

    # Boilerplate phrases
    sections.append(_boilerplate_section(analysis_results))

    # Introduction background clusters
    sections.append(_intro_section(analysis_results))

    # Problems
    sections.append(_problems_section(analysis_results))

    # Figures
    sections.append(_figures_section(analysis_results))

    # Open questions
    sections.append(_open_questions_section(analysis_results))

    # Footer
    sections.append(_report_footer())

    return "\n\n".join(sections)


def generate_html_report(analysis_results: dict) -> str:
    """Generate an HTML report with embedded styling."""
    md = generate_report(analysis_results)

    # Simple markdown-to-HTML conversion
    html = _markdown_to_html(md)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Literature Review Analysis Report</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    line-height: 1.6;
    color: #333;
}}
h1 {{ color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: 10px; }}
h2 {{ color: #2471a3; border-bottom: 1px solid #aed6f1; padding-bottom: 6px; margin-top: 30px; }}
h3 {{ color: #2e86c1; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background-color: #eaf2f8; }}
tr:nth-child(even) {{ background-color: #f8f9fa; }}
blockquote {{ border-left: 4px solid #2980b9; margin: 15px 0; padding: 10px 20px; background: #eaf2f8; }}
code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
.meta {{ color: #888; font-size: 0.9em; }}
.header-box {{ background: linear-gradient(135deg, #1a5276, #2980b9); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; }}
.header-box h1 {{ color: white; border: none; margin: 0; }}
</style>
</head>
<body>
{html}
</body>
</html>"""


def _report_header(results: dict) -> str:
    stats = results.get("stats", {})
    n_papers = stats.get("n_papers", len(results.get("papers", [])))
    return f"""# Literature Review Analysis Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Papers Analyzed**: {n_papers}
**Analysis Modules**: Method Similarity | Boilerplate Detection | Introduction Clustering | Problem Extraction | Figure Analysis | Open Questions
"""


def _executive_summary(results: dict) -> str:
    lines = ["## Executive Summary", ""]
    stats = results.get("stats", {})

    n_papers = stats.get("n_papers", 0)

    # Method summary
    method_data = results.get("method_similarity", {})
    n_method_clusters = len(method_data.get("clusters", []))
    lines.append(f"- **{n_papers} papers** were analyzed across 6 dimensions")
    lines.append(f"- **{n_method_clusters} shared method clusters** identified across papers")

    # Boilerplate summary
    bp_data = results.get("boilerplate", {})
    bp_summary = bp_data.get("summary", {})
    lines.append(f"- **{bp_summary.get('total_groups', 0)} routine expression groups** detected")

    # Intro clusters
    intro_data = results.get("intro_clusters", {})
    lines.append(f"- **{len(intro_data.get('clusters', []))} introduction background clusters** found")

    # Problems
    problem_data = results.get("problems", {})
    lines.append(f"- **{len(problem_data.get('problem_clusters', []))} problem clusters** across papers")

    # Figures
    fig_data = results.get("figures", {})
    lines.append(f"- **{len(fig_data.get('figure_groups', []))} similar figure groups** identified")

    # Open questions
    oq_data = results.get("open_questions", {})
    oq_stats = oq_data.get("stats", {})
    lines.append(f"- **{oq_stats.get('convergent', 0)} convergent open questions** (identified by ≥2 papers)")
    lines.append(f"- **{oq_stats.get('unique', 0)} unique open questions** (identified by single papers)")

    return "\n".join(lines)


def _methods_section(results: dict) -> str:
    data = results.get("method_similarity", {})
    clusters = data.get("clusters", [])
    lines = ["## Shared Methods & Techniques", ""]

    if not clusters:
        lines.append("*No shared method clusters found across papers.*")
        return "\n".join(lines)

    for i, c in enumerate(clusters):
        lines.append(f"### {i + 1}. {c['label']}")
        lines.append(f"**Papers**: {', '.join(c['papers'])} ({c['paper_count']} papers)")
        lines.append(f"**Key terms**: {', '.join(c['key_terms'][:5])}")
        lines.append("")
        lines.append("**Representative excerpts:**")
        for para in c.get("paragraphs", [])[:3]:
            lines.append(f"> [{para['paper_label']}] {para['text'][:300]}")
            lines.append("")
        lines.append("")

    return "\n".join(lines)


def _boilerplate_section(results: dict) -> str:
    data = results.get("boilerplate", {})
    by_section = data.get("by_section", {})
    lines = ["## Routine / Templated Expressions", ""]

    total = 0
    for section_name, groups in by_section.items():
        if not groups:
            continue
        lines.append(f"### {section_name.title()} ({len(groups)} groups)")
        lines.append("")
        for g in groups[:5]:
            lines.append(f"- **\"{g['shared_phrase'][:120]}\"** — {g['paper_count']} papers")
            total += 1
        lines.append("")

    if total == 0:
        lines.append("*No significant routine expressions detected across papers.*")

    return "\n".join(lines)


def _intro_section(results: dict) -> str:
    data = results.get("intro_clusters", {})
    clusters = data.get("clusters", [])
    lines = ["## Introduction Background Clusters", ""]

    if not clusters:
        lines.append("*No introduction background clusters found.*")
        return "\n".join(lines)

    for i, c in enumerate(clusters):
        lines.append(f"### Cluster {i + 1}: {c['label']}")
        lines.append(f"**Papers ({c['paper_count']})**: {', '.join(c['papers'])}")
        lines.append(f"**Key terms**: {', '.join(c['key_terms'][:5])}")
        lines.append("")
        if c.get("representative_excerpt"):
            lines.append(f"> {c['representative_excerpt'][:400]}")
        lines.append("")

    return "\n".join(lines)


def _problems_section(results: dict) -> str:
    data = results.get("problems", {})
    clusters = data.get("problem_clusters", [])
    per_paper = data.get("per_paper", [])
    lines = ["## Research Problems Addressed", ""]

    # Problem clusters
    if clusters:
        lines.append("### Cross-Paper Problem Clusters")
        lines.append("")
        lines.append("| Problem | Papers | Instances |")
        lines.append("|---------|--------|-----------|")
        for c in clusters:
            lines.append(f"| {c['label'][:100]} | {c['paper_count']} | {c['problem_count']} |")
        lines.append("")

    # Per-paper problems
    lines.append("### Problems by Paper")
    lines.append("")
    for pp in per_paper:
        lines.append(f"**{pp['paper_label']}** ({pp['problem_count']} problems)")
        for prob in pp.get("problems", [])[:5]:
            lines.append(f"- [{prob['section']}] {prob['sentence'][:200]}")
        if len(pp.get("problems", [])) > 5:
            lines.append(f"- *... and {len(pp['problems']) - 5} more*")
        lines.append("")

    return "\n".join(lines)


def _figures_section(results: dict) -> str:
    data = results.get("figures", {})
    groups = data.get("figure_groups", [])
    patterns = data.get("discussion_patterns", [])
    fig_types = results.get("figure_types", [])
    lines = ["## Figure Analysis", ""]

    # Similar figure groups
    if groups:
        lines.append("### Similar Figure Groups")
        lines.append("")
        for g in groups:
            lines.append(f"- **{g['label'][:100]}** ({g['paper_count']} papers, {g['figure_count']} figures)")
        lines.append("")

    # Common figure types
    if fig_types:
        lines.append("### Common Figure Types Across Papers")
        lines.append("")
        lines.append("| Figure Type | Papers | Total Figures |")
        lines.append("|------------|--------|---------------|")
        for ft in fig_types:
            lines.append(f"| {ft['figure_type']} | {ft['paper_count']} | {ft['total_figures']} |")
        lines.append("")

    # Discussion patterns
    if patterns:
        lines.append("### Templated Figure Discussion Patterns")
        lines.append("")
        for dp in patterns:
            lines.append(f"**{dp['figure_group'][:80]}**")
            for pat in dp.get("patterns", [])[:3]:
                lines.append(f"> \"{pat['shared_phrase'][:150]}\"")
            lines.append("")

    if not groups and not fig_types:
        lines.append("*No cross-paper figure analysis results available.*")

    return "\n".join(lines)


def _open_questions_section(results: dict) -> str:
    data = results.get("open_questions", {})
    convergent = data.get("convergent_questions", [])
    unique = data.get("unique_questions", [])
    lines = ["## Unsolved Problems & Open Questions", ""]

    if convergent:
        lines.append("### Convergent Open Questions (Identified by ≥2 Papers)")
        lines.append("")
        for i, q in enumerate(convergent):
            lines.append(f"**{i + 1}. {q['summary'][:200]}**")
            lines.append(f"- Papers: {q['paper_count']}")
            for item in q.get("items", []):
                lines.append(f"  - [{item['paper_label']}] \"{item['text'][:200]}\"")
            lines.append("")

    if unique:
        lines.append("### Unique Open Questions (Per-Paper)")
        lines.append("")
        for i, q in enumerate(unique[:10]):
            lines.append(f"- [{q['items'][0]['paper_label']}] {q['summary'][:200]}")
        if len(unique) > 10:
            lines.append(f"- *... and {len(unique) - 10} more*")
        lines.append("")

    if not convergent and not unique:
        lines.append("*No open questions extracted.*")

    return "\n".join(lines)


def _report_footer() -> str:
    return f"""---

*Report generated by Literature Review Assistant on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.*
"""


def _markdown_to_html(md: str) -> str:
    """Simple markdown to HTML converter."""
    import re

    lines = md.split("\n")
    html = []
    in_list = False
    in_blockquote = False
    in_table = False

    for line in lines:
        # Headers
        if line.startswith("### "):
            if in_list:
                html.append("</ul>")
                in_list = False
            if in_blockquote:
                html.append("</blockquote>")
                in_blockquote = False
            html.append(f"<h3>{line[4:]}</h3>")
            continue
        if line.startswith("## "):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<h2>{line[3:]}</h2>")
            continue
        if line.startswith("# "):
            html.append(f"<h1>{line[2:]}</h1>")
            continue

        # Blockquote
        if line.startswith("> "):
            if not in_blockquote:
                html.append("<blockquote>")
                in_blockquote = True
            html.append(line[2:] + "<br>")
            continue
        elif in_blockquote:
            html.append("</blockquote>")
            in_blockquote = False

        # Tables
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                html.append("<table>")
                in_table = True
            cells = [c.strip() for c in line.split("|")[1:-1]]
            is_sep = all(re.match(r"^-+$", c) for c in cells)
            if is_sep:
                continue
            tag = "th" if "---" not in line else "td"
            html.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        elif in_table and not line.startswith("|"):
            html.append("</table>")
            in_table = False

        # Lists
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            if not in_list:
                html.append("<ul>")
                in_list = True
            content = line.strip()[2:]
            # Bold
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            html.append(f"<li>{content}</li>")
            continue
        elif in_list and not line.strip().startswith(("- ", "* ")):
            html.append("</ul>")
            in_list = False

        # Bold
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        # Empty line
        if not line.strip():
            html.append("<br>")
        else:
            html.append(f"<p>{line}</p>")

    if in_list:
        html.append("</ul>")
    if in_blockquote:
        html.append("</blockquote>")
    if in_table:
        html.append("</table>")

    return "\n".join(html)
