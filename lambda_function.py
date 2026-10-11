"""
LLM-KVOps: AWS Lambda Serverless Entrypoint & Function URL Handler
Serves both the interactive web dashboard (GET) and runs the multi-tier KVOps engine (POST).
100% Serverless, runs on AWS Free Tier with $0.00 idle cost.
"""

import json
import base64
import os
import sys

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.inference_simulator import InferenceSimulator, MODEL_PRESETS
from src.engine.aws_cloudwatch import AWSCloudWatchLogger


def get_html_dashboard():
    """Returns the standalone, high-performance HTML/CSS/JS frontend for the Lambda Function URL."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cloud-KVOps: Serverless LLM Inference on AWS</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-green: #4ade80;
            --accent-red: #f87171;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --border: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background-color: var(--bg); color: var(--text); padding: 24px; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
        .title { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
        .badge { background: #0369a1; color: white; padding: 4px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
        .grid-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
        .card h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim); margin-bottom: 8px; }
        .card .value { font-size: 1.8rem; font-weight: 800; color: var(--text); }
        .card .subtext { font-size: 0.85rem; margin-top: 6px; }
        .positive { color: var(--accent-green); }
        .negative { color: var(--accent-red); }
        .controls { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; display: grid; grid-template-columns: 1fr 1fr 1fr auto; gap: 16px; align-items: flex-end; }
        .control-group label { display: block; font-size: 0.85rem; color: var(--text-dim); margin-bottom: 6px; font-weight: 600; }
        select, input, button { width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border); background: #0f172a; color: var(--text); font-size: 0.9rem; }
        button { background: #0284c7; color: white; font-weight: 700; cursor: pointer; border: none; transition: background 0.2s; }
        button:hover { background: #0369a1; }
        .chart-container { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; height: 380px; position: relative; }
        .token-map { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; max-height: 250px; overflow-y: auto; line-height: 2.2; }
        .token-sink { background: #854d0e; color: #fef08a; padding: 3px 6px; border-radius: 4px; font-size: 0.85rem; font-weight: 700; margin: 2px; display: inline-block; }
        .token-ret { background: #14532d; color: #bbf7d0; padding: 3px 6px; border-radius: 4px; font-size: 0.85rem; margin: 2px; display: inline-block; }
        .token-evi { background: #334155; color: #64748b; padding: 3px 6px; border-radius: 4px; font-size: 0.85rem; text-decoration: line-through; margin: 2px; display: inline-block; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <div class="title">⚡ Cloud-KVOps: Serverless LLM Inference Optimizer</div>
                <div style="font-size: 0.85rem; color: var(--text-dim); margin-top: 4px;">AWS Lambda Serverless Execution • Real-Time VRAM & FinOps Profiler</div>
            </div>
            <div class="badge">AWS Lambda Free Tier ($0.00 Idle Cost)</div>
        </header>

        <div class="controls">
            <div class="control-group">
                <label>Target Architecture</label>
                <select id="modelSelect">
                    <option value="LLaMA-3-70B (80L, 8H, 128D)">LLaMA-3-70B (80 Layers, 8 GQA Heads)</option>
                    <option value="Mistral-7B (32L, 8H, 128D)">Mistral-7B (32 Layers, 8 GQA Heads)</option>
                    <option value="LLaMA-3-8B (32L, 8H, 128D)">LLaMA-3-8B (32 Layers, 8 GQA Heads)</option>
                </select>
            </div>
            <div class="control-group">
                <label>Context Length (Tokens)</label>
                <select id="contextSelect">
                    <option value="2048">2,048 Tokens (Standard)</option>
                    <option value="4096" selected>4,096 Tokens (Long Document)</option>
                    <option value="8192">8,192 Tokens (Extended)</option>
                    <option value="16384">16,384 Tokens (Heavy Load)</option>
                </select>
            </div>
            <div class="control-group">
                <label>VRAM Budget Ratio</label>
                <select id="budgetSelect">
                    <option value="0.30">30% (70% Reduction)</option>
                    <option value="0.40" selected>40% (60% Reduction)</option>
                    <option value="0.50">50% (50% Reduction)</option>
                </select>
            </div>
            <button onclick="runOptimization()" id="runBtn">⚡ Run Simulation</button>
        </div>

        <div style="background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 24px;">
            <label style="display:block; font-size: 0.85rem; color: var(--text-dim); margin-bottom: 6px; font-weight: 600;">💬 Prompt / Document Input (Compared Side-by-Side on both Models):</label>
            <textarea id="promptText" rows="2" style="width:100%; padding:10px 12px; border-radius:8px; border:1px solid var(--border); background:#0f172a; color:var(--text); font-size:0.9rem;">This Master Services Agreement is entered into regarding enterprise cloud computing infrastructure and data privacy compliance under ISO 27001 standards with high-throughput autoregressive inference...</textarea>
        </div>

        <div class="grid-cards">
            <div class="card">
                <h3>Baseline VRAM (Standard AWS)</h3>
                <div class="value negative" id="baseVram">-- GB</div>
                <div class="subtext">Unbounded O(N) linear growth</div>
            </div>
            <div class="card">
                <h3>Cloud-KVOps VRAM (Optimized)</h3>
                <div class="value positive" id="kvopsVram">-- GB</div>
                <div class="subtext" id="vramSavedPct">--% Memory Freed</div>
            </div>
            <div class="card">
                <h3>Estimated AWS FinOps Savings</h3>
                <div class="value positive" id="finopsSaved">--%</div>
                <div class="subtext" id="monthlySavedText">Save $-- / mo on EC2 GPU bill</div>
            </div>
            <div class="card">
                <h3>Algorithmic Decision Latency</h3>
                <div class="value" id="latency">-- ms</div>
                <div class="subtext">Sub-millisecond decision time</div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="vramChart"></canvas>
        </div>

        <h3 style="margin-bottom: 12px; font-size: 1rem; color: var(--text-dim);">Live Token Classification Map (Knapsack & Attention Sinks)</h3>
        <div class="token-map" id="tokenMap">
            Click <b>"Run Simulation"</b> above to process context tokens with 0-1 Knapsack Dynamic Programming...
        </div>
    </div>

    <script>
        let chartInstance = null;

        async function runOptimization() {
            const btn = document.getElementById("runBtn");
            btn.innerText = "⏳ Computing...";
            btn.disabled = true;

            const payload = {
                model_name: document.getElementById("modelSelect").value,
                token_count: parseInt(document.getElementById("contextSelect").value),
                budget_ratio: parseFloat(document.getElementById("budgetSelect").value),
                sample_text: document.getElementById("promptText").value
            };

            try {
                const res = await fetch(window.location.href, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                // Update Cards
                document.getElementById("baseVram").innerText = data.baseline.vram_gb.toFixed(2) + " GB";
                document.getElementById("kvopsVram").innerText = data.kvops.vram_gb.toFixed(2) + " GB";
                document.getElementById("vramSavedPct").innerText = "-" + data.kvops.memory_saved_pct.toFixed(1) + "% Memory Freed";
                document.getElementById("finopsSaved").innerText = data.finops.percentage_savings + "% Saved";
                document.getElementById("monthlySavedText").innerText = "Save $" + data.finops.monthly_savings_usd + "/mo on AWS GPU bill";
                document.getElementById("latency").innerText = data.kvops.latency_overhead_ms.toFixed(2) + " ms";

                // Update Chart
                const ts = data.time_series;
                const ctx = document.getElementById('vramChart').getContext('2d');
                if (chartInstance) chartInstance.destroy();
                chartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: ts.steps,
                        datasets: [
                            {
                                label: 'Baseline VRAM (Without Eviction)',
                                data: ts.baseline_vram_mb,
                                borderColor: '#f87171',
                                backgroundColor: 'rgba(248, 113, 113, 0.1)',
                                tension: 0.1,
                                borderWidth: 3
                            },
                            {
                                label: 'Cloud-KVOps VRAM (0-1 Knapsack + Geometry)',
                                data: ts.kvops_vram_mb,
                                borderColor: '#4ade80',
                                backgroundColor: 'rgba(74, 222, 128, 0.1)',
                                tension: 0.1,
                                borderWidth: 3
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { labels: { color: '#f8fafc' } }
                        },
                        scales: {
                            x: { title: { display: true, text: 'Context Length (Tokens)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
                            y: { title: { display: true, text: 'VRAM Footprint (MB)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
                        }
                    }
                });

                // Update Token Map
                let html = "";
                data.token_texts.slice(0, 150).forEach((t, i) => {
                    const st = data.token_status_sample[i];
                    if (st === "SINK") html += `<span class="token-sink">★ ${t}</span> `;
                    else if (st === "RETAINED") html += `<span class="token-ret">${t}</span> `;
                    else html += `<span class="token-evi">${t}</span> `;
                });
                document.getElementById("tokenMap").innerHTML = html;

            } catch (err) {
                alert("Error calling Lambda API: " + err);
            } finally {
                btn.innerText = "⚡ Run Simulation";
                btn.disabled = false;
            }
        }

        // Run once on initial load
        window.onload = runOptimization;
    </script>
</body>
</html>"""


def lambda_handler(event, context):
    """AWS Lambda entrypoint handler."""
    http_method = None
    if "requestContext" in event and "http" in event["requestContext"]:
        http_method = event["requestContext"]["http"].get("method")
    if not http_method:
        http_method = event.get("httpMethod", "GET")

    # 1. Handle GET request: Return full interactive Web UI
    if http_method == "GET":
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "text/html; charset=utf-8",
                "Access-Control-Allow-Origin": "*"
            },
            "body": get_html_dashboard()
        }

    # 2. Handle POST request: Run multi-tier KVOps engine
    try:
        body_raw = event.get("body", "{}")
        if event.get("isBase64Encoded", False):
            body_raw = base64.b64decode(body_raw).decode("utf-8")
        body = json.loads(body_raw) if body_raw else {}

        model_name = body.get("model_name", "LLaMA-3-70B (80L, 8H, 128D)")
        token_count = int(body.get("token_count", 4096))
        budget_ratio = float(body.get("budget_ratio", 0.40))

        sample_text = body.get("sample_text")

        # Run multi-tier KVOps simulation
        simulator = InferenceSimulator(model_key=model_name, target_budget_ratio=budget_ratio)
        sim_data = simulator.run_side_by_side_simulation(token_count=token_count, sample_text=sample_text)

        # Calculate AWS FinOps Cost Savings
        cw = AWSCloudWatchLogger(enabled=False)
        finops = cw.calculate_cost_savings(
            baseline_vram_gb=sim_data["baseline"]["vram_gb"],
            kvops_vram_gb=sim_data["kvops"]["vram_gb"]
        )

        sim_data["finops"] = finops

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(sim_data)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }
