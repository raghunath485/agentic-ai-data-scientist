"""
Executive Presentation & Interactive HTML Slide Deck Generator.
Creates a self-contained, high-impact HTML slide deck with CSS presentation controls.
"""

from typing import Dict, Any
from core.state import DataScienceState


class SlideDeckGenerator:
    @staticmethod
    def generate_html_deck(state: DataScienceState) -> str:
        """Constructs a responsive dark-mode cyber presentation slide deck."""
        profile = state.profiling_summary
        best_name = state.best_model_name
        best_res = state.model_results.get(best_name) if best_name else None
        metric_str = ", ".join([f"{k}: {v}" for k, v in (best_res.metrics.items() if best_res else []) if k != "Confusion Matrix" and v is not None])

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Executive Briefing: {state.file_name or 'Autonomous Investigation'}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono&display=swap');
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: #090d16;
            color: #f1f5f9;
            font-family: 'Outfit', sans-serif;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }}
        .deck-container {{
            max-width: 900px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 40px;
        }}
        .slide {{
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid rgba(0, 242, 254, 0.25);
            border-radius: 16px;
            padding: 45px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
            position: relative;
        }}
        .slide-num {{
            position: absolute;
            top: 25px;
            right: 30px;
            font-size: 0.85rem;
            color: #64748b;
            font-family: 'JetBrains Mono', monospace;
        }}
        h1 {{ font-size: 2.2rem; color: #00f2fe; margin-bottom: 12px; }}
        h2 {{ font-size: 1.6rem; color: #c084fc; margin-bottom: 20px; }}
        p {{ font-size: 1.05rem; line-height: 1.6; color: #cbd5e1; margin-bottom: 15px; }}
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: rgba(0, 242, 254, 0.15);
            color: #00f2fe;
            border: 1px solid rgba(0, 242, 254, 0.3);
            margin-bottom: 20px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 25px 0;
        }}
        .kpi-card {{
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-val {{ font-size: 1.8rem; font-weight: 700; color: #34d399; }}
        .kpi-label {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; margin-top: 5px; }}
        ul {{ padding-left: 20px; margin: 15px 0; }}
        li {{ margin-bottom: 8px; color: #cbd5e1; font-size: 1rem; }}
        .btn-print {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #00f2fe;
            color: #090d16;
            font-weight: 700;
            border: none;
            padding: 12px 24px;
            border-radius: 30px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(0, 242, 254, 0.4);
            font-size: 0.95rem;
        }}
        .btn-print:hover {{ background: #38bdf8; }}
        @media print {{
            .btn-print {{ display: none; }}
            body {{ background: white; color: black; }}
            .slide {{ page-break-after: always; border: 1px solid #ccc; background: white; color: black; box-shadow: none; }}
            h1, h2 {{ color: #0284c7; }}
            p, li {{ color: #333; }}
        }}
    </style>
</head>
<body>
    <div class="deck-container">
        <!-- Slide 1: Title -->
        <div class="slide">
            <span class="slide-num">SLIDE 01</span>
            <span class="badge">EXECUTIVE INVESTIGATION BRIEFING</span>
            <h1>Autonomous AI Data Science Findings</h1>
            <p>Dataset Artifact: <code>{state.file_name or 'Uploaded Data'}</code></p>
            <p>Target Objective: <b>{state.target_column}</b> ({state.problem_type.replace('_', ' ').title()})</p>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-val">{profile.get('health_score', 100)}/100</div>
                    <div class="kpi-label">Data Health Score</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-val">{best_name or 'Pending'}</div>
                    <div class="kpi-label">Winning Production Model</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-val">{profile.get('total_rows', 0):,}</div>
                    <div class="kpi-label">Observations Analyzed</div>
                </div>
            </div>
        </div>

        <!-- Slide 2: Data Health & Forensics -->
        <div class="slide">
            <span class="slide-num">SLIDE 02</span>
            <h2>1. Data Health & Anomaly Profiling</h2>
            <p>Comprehensive statistical diagnostics performed across all features:</p>
            <ul>
                <li><b>Total Observations:</b> {profile.get('total_rows', 0):,} rows across {profile.get('total_cols', 0)} dimensions.</li>
                <li><b>Duplicate Rows:</b> {profile.get('duplicate_rows', 0)} ({profile.get('duplicate_pct', 0)}%).</li>
                <li><b>High-Cardinality Filter:</b> Filtered out {len(profile.get('high_cardinality_cols', []))} identifier / noise columns.</li>
                <li><b>Zero-Variance Filter:</b> Dropped {len(profile.get('constant_cols', []))} constant attributes.</li>
            </ul>
        </div>

        <!-- Slide 3: Model Arena Results -->
        <div class="slide">
            <span class="slide-num">SLIDE 03</span>
            <h2>2. Model Arena Benchmarking</h2>
            <p>Algorithms evaluated using 5-Fold Stratified Cross-Validation on holdout partitions:</p>
            <div style="background:rgba(15,23,42,0.8); padding:15px; border-radius:8px; border-left:4px solid #34d399; margin-bottom:15px;">
                <b style="color:#34d399;">Champion Algorithm:</b> {best_name}<br/>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.9rem; color:#94a3b8;">{metric_str}</span>
            </div>
            <p>The model was selected based on optimal generalization and resistance to overfitting.</p>
        </div>

        <!-- Slide 4: Strategic Recommendations -->
        <div class="slide">
            <span class="slide-num">SLIDE 04</span>
            <h2>3. Production Strategy & Operational Next Steps</h2>
            <ul>
                <li><b>Deploy Microservice:</b> Automated FastAPI REST API generated and containerized with Docker.</li>
                <li><b>Covariate Shift Monitoring:</b> Monitor features with Population Stability Index (PSI) > 0.10.</li>
                <li><b>Explainability Audit:</b> Review SHAP attribution waterfalls for regulated decision flows.</li>
            </ul>
        </div>
    </div>

    <button class="btn-print" onclick="window.print()">🖨️ Print / Save as PDF</button>
</body>
</html>
"""
