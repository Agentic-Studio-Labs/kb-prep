"""Generate interactive local HTML reports from manifest data."""

from __future__ import annotations

import json
from pathlib import Path


def write_web_report(output_path: str, manifest_data: dict) -> str:
    """Write a standalone HTML dashboard for one run."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(manifest_data, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IngestGate Run Dashboard</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #131b33;
      --text: #e8eefc;
      --muted: #9bb0df;
      --pass: #22c55e;
      --notes: #f59e0b;
      --remediate: #f97316;
      --hold: #ef4444;
      --accent: #60a5fa;
      --border: #283252;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: linear-gradient(180deg, #0a0f1e 0%, #0b1020 100%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .title {{
      margin: 0 0 4px;
      font-size: 28px;
    }}
    .sub {{
      margin: 0 0 20px;
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
    }}
    .k {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .v {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 4px;
    }}
    .section {{
      margin-top: 20px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }}
    .row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .decision-label {{
      min-width: 280px;
      white-space: nowrap;
      font-size: 14px;
    }}
    .decision-pct {{
      width: 48px;
      text-align: right;
      white-space: nowrap;
      color: var(--muted);
      font-size: 13px;
    }}
    .bar {{
      height: 10px;
      border-radius: 999px;
      background: #1f2a48;
      overflow: hidden;
      flex: 1;
    }}
    .fill {{
      height: 100%;
    }}
    .tbl {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 14px;
    }}
    .tbl th, .tbl td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
    }}
    .tbl tr[data-clickable="1"] {{
      cursor: pointer;
    }}
    .tbl tr[data-clickable="1"]:hover {{
      background: rgba(96,165,250,.08);
    }}
    .tbl tr.selected {{
      background: rgba(96,165,250,.15);
    }}
    .tag {{
      display: inline-block;
      border-radius: 999px;
      font-size: 12px;
      padding: 2px 8px;
      font-weight: 600;
    }}
    .pass {{ background: rgba(34,197,94,.2); color: var(--pass); }}
    .notes {{ background: rgba(245,158,11,.2); color: var(--notes); }}
    .remediate {{ background: rgba(249,115,22,.2); color: var(--remediate); }}
    .hold {{ background: rgba(239,68,68,.2); color: var(--hold); }}
    input {{
      width: 100%;
      border: 1px solid var(--border);
      background: #0e152b;
      color: var(--text);
      border-radius: 8px;
      padding: 8px;
      outline: none;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 180px 1fr;
      gap: 8px;
      margin-top: 8px;
      font-size: 14px;
    }}
    .kv .label {{
      color: var(--muted);
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
    }}
    .inline-tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .tbl-mini {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .tbl-mini th, .tbl-mini td {{
      border-bottom: 1px solid var(--border);
      padding: 4px 6px;
      text-align: left;
    }}
    @media (max-width: 900px) {{
      .decision-label {{
        min-width: 220px;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">IngestGate Dashboard</h1>
    <p class="sub">Interactive local run summary and triage view.</p>

    <div class="grid" id="summary"></div>

    <div class="section">
      <h3>Gate Decision Distribution</h3>
      <div id="decision-bars"></div>
    </div>

    <div class="section">
      <h3>Document Triage</h3>
      <input id="q" placeholder="Filter by file, decision, mode..." />
      <table class="tbl">
        <thead>
          <tr>
            <th>File</th>
            <th>Score</th>
            <th>Gate Decision</th>
            <th>Retrieval Mode</th>
          </tr>
        </thead>
        <tbody id="docs"></tbody>
      </table>
    </div>

    <div class="section">
      <h3>Document Drill-In</h3>
      <div id="detail" class="sub">Click a row in Document Triage to inspect details.</div>
    </div>
  </div>

  <script>
    const DATA = {payload};

    const corpus = DATA.corpus || {{}};
    const docs = DATA.documents || [];

    const summary = [
      ["Documents", corpus.total_documents ?? docs.length],
      ["Chunks", corpus.total_chunks ?? 0],
      ["Avg Score", corpus.avg_score ?? 0],
      ["Benchmarks", (DATA.benchmarks || []).length],
    ];

    document.getElementById("summary").innerHTML = summary.map(([k, v]) => `
      <div class="card"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>
    `).join("");

    const dist = corpus.gate_decision_distribution || {{}};
    const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
    const color = (d) => d === "PASS" ? "var(--pass)"
      : d === "PASS_WITH_NOTES" ? "var(--notes)"
      : d === "REMEDIATION_RECOMMENDED" ? "var(--remediate)"
      : "var(--hold)";

    document.getElementById("decision-bars").innerHTML = Object.keys(dist).map((k) => {{
      const pct = Math.round((dist[k] / total) * 100);
      return `<div class="row">
        <div class="decision-label">${{k}} (${{dist[k]}})</div>
        <div class="bar"><div class="fill" style="width:${{pct}}%;background:${{color(k)}}"></div></div>
        <div class="decision-pct">${{pct}}%</div>
      </div>`;
    }}).join("");

    function cls(decision) {{
      if (decision === "PASS") return "pass";
      if (decision === "PASS_WITH_NOTES") return "notes";
      if (decision === "REMEDIATION_RECOMMENDED") return "remediate";
      return "hold";
    }}

    function modeOf(doc) {{
      return doc.retrieval_quality_gate?.retrieval_mode_hint?.recommended_mode || "";
    }}

    const tbody = document.getElementById("docs");
    const detail = document.getElementById("detail");
    let selectedFile = "";

    function esc(v) {{
      return String(v ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    }}

    function renderDetail(doc) {{
      if (!doc) {{
        detail.innerHTML = "Click a row in Document Triage to inspect details.";
        return;
      }}
      const rqg = doc.retrieval_quality_gate || {{}};
      const hint = rqg.retrieval_mode_hint || {{}};
      const mod = rqg.modality_readiness || {{}};
      const evidence = rqg.evidence || {{}};
      const reasons = hint.reasons || [];
      const topics = doc.topics || [];
      const sev = doc.issues_by_severity || {{}};
      const criteria = doc.criteria_scores || {{}};
      const criteriaRows = Object.entries(criteria)
        .sort(([, a], [, b]) => (b.score || 0) - (a.score || 0))
        .map(([k, v]) => `
          <tr>
            <td>${{esc(v.label || k)}}</td>
            <td>${{esc(v.score ?? "")}}</td>
            <td>${{esc(v.weight ?? "")}}</td>
            <td>${{esc(v.issue_count ?? 0)}}</td>
          </tr>
        `)
        .join("");
      const rows = [
        ["File", doc.source_file],
        ["Score", doc.overall_score],
        ["Gate Decision", doc.gate_decision || ""],
        ["Legacy Readiness", doc.readiness || ""],
        ["Retrieval Mode", hint.recommended_mode || ""],
        ["Confidence", hint.confidence || ""],
        ["Domain", doc.domain || ""],
        ["Topics", topics.join(", ")],
        ["Chunk Count", doc.chunk_count ?? ""],
        ["Entity Count", doc.entity_count ?? ""],
        ["Relationship Count", doc.relationship_count ?? ""],
      ];
      detail.innerHTML = `
        <div class="kv">
          ${{rows.map(([k, v]) => `<div class="label">${{esc(k)}}</div><div>${{esc(v)}}</div>`).join("")}}
          <div class="label">Reasons</div>
          <div>${{
            reasons.length
              ? reasons.map((r) => `<span class="tag notes mono">${{esc(r)}}</span>`).join(" ")
              : "<span class='sub'>None</span>"
          }}</div>
          <div class="label">Issues by Severity</div>
          <div class="inline-tags">
            <span class="tag hold">critical: ${{esc(sev.critical ?? 0)}}</span>
            <span class="tag notes">warning: ${{esc(sev.warning ?? 0)}}</span>
            <span class="tag pass">info: ${{esc(sev.info ?? 0)}}</span>
          </div>
          <div class="label">Criteria Scores</div>
          <div>
            <table class="tbl-mini">
              <thead>
                <tr>
                  <th>Criterion</th>
                  <th>Score</th>
                  <th>Weight</th>
                  <th>Issues</th>
                </tr>
              </thead>
              <tbody>${{criteriaRows || "<tr><td colspan='4'>No criteria data</td></tr>"}}</tbody>
            </table>
          </div>
          <div class="label">Modality Flags</div>
          <div class="mono">${{esc(JSON.stringify(mod, null, 2))}}</div>
          <div class="label">Evidence</div>
          <div class="mono">${{esc(JSON.stringify(evidence, null, 2))}}</div>
        </div>
      `;
    }}

    function render(filter = "") {{
      const q = filter.toLowerCase();
      const rows = docs.filter((d) => {{
        const h = `${{d.source_file}} ${{d.gate_decision || ""}} ${{modeOf(d)}}`.toLowerCase();
        return h.includes(q);
      }});
      tbody.innerHTML = rows.map((d) => `
        <tr
          data-clickable="1"
          data-file="${{d.source_file}}"
          class="${{d.source_file === selectedFile ? "selected" : ""}}"
        >
          <td>${{d.source_file}}</td>
          <td>${{d.overall_score ?? ""}}</td>
          <td><span class="tag ${{cls(d.gate_decision)}}">${{d.gate_decision || ""}}</span></td>
          <td>${{modeOf(d)}}</td>
        </tr>
      `).join("");
      tbody.querySelectorAll("tr[data-clickable='1']").forEach((row) => {{
        row.addEventListener("click", () => {{
          selectedFile = row.getAttribute("data-file") || "";
          const doc = docs.find((d) => d.source_file === selectedFile);
          render();
          renderDetail(doc);
        }});
      }});
    }}
    render();
    if (docs.length) {{
      selectedFile = docs[0].source_file;
      render();
      renderDetail(docs[0]);
    }}
    document.getElementById("q").addEventListener("input", (e) => render(e.target.value));
  </script>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return str(out)
