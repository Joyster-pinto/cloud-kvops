import os
import sys
import time

# -------------------------------------------------------------
# Python 3.12 / Streamlit Threading Fix for Windows
# Prevents 'can't register atexit after shutdown' during transformers import
# -------------------------------------------------------------
import threading
if hasattr(threading, "_register_atexit"):
    _orig_reg = threading._register_atexit
    def _safe_register_atexit(*args, **kwargs):
        try:
            return _orig_reg(*args, **kwargs)
        except RuntimeError:
            pass
    threading._register_atexit = _safe_register_atexit

try:
    import concurrent.futures.process
    import joblib
except Exception:
    pass

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.engine.real_llm_runner import RealLLMRunner
from src.engine.inference_simulator import InferenceSimulator, MODEL_PRESETS


st.set_page_config(
    page_title="LLM-KVOps: Real-Time Offline GPU LLM Execution",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0d6efd;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #495057;
        margin-bottom: 18px;
    }
    .gpu-badge {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 15px;
    }
    .stream-box {
        background-color: #f8f9fa;
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 15px;
        min-height: 220px;
        max-height: 350px;
        overflow-y: auto;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.95rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Hardware Detection
has_cuda = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if has_cuda else 0.0

st.markdown("<div class='main-header'>⚡ LLM-KVOps: Real-Time Offline GPU LLM Execution & KV-Cache Pruning</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'><b>Vellore Institute of Technology</b> • DAA Course Project (BCSE204L/P) • Live Hardware Resource Profiler</div>", unsafe_allow_html=True)

if has_cuda:
    st.markdown(f"<div class='gpu-badge'>🎮 Active Hardware: <b>{device_name}</b> ({total_vram_gb:.2f} GB VRAM, CUDA Acceleration Active)</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='gpu-badge' style='background-color:#fff3cd; color:#664d03;'>💻 Active Hardware: <b>CPU Mode</b> (PyTorch Local Execution)</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.header("⚙️ Execution Mode & Settings")

exec_mode = st.sidebar.radio(
    "Inference Engine Mode",
    options=["🎮 Real Local PyTorch GPU Inference", "⚡ High-Throughput Long-Context Simulator"],
    index=0
)

if exec_mode == "🎮 Real Local PyTorch GPU Inference":
    model_category = st.sidebar.selectbox(
        "Model Family",
        options=[
            "Alibaba Qwen (Cached & Ready)",
            "Classic GPT-2 / DistilGPT2 (Lightweight)",
            "Google Gemma (Requires HF Token)",
            "Meta LLaMA / TinyLlama",
            "Microsoft Phi",
            "Custom HuggingFace Model"
        ],
        index=0
    )

    if model_category == "Alibaba Qwen (Cached & Ready)":
        model_name_input = st.sidebar.selectbox(
            "Select Qwen Variant",
            options=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B"],
            index=0
        )
    elif model_category == "Classic GPT-2 / DistilGPT2 (Lightweight)":
        model_name_input = st.sidebar.selectbox(
            "Select GPT-2 Variant",
            options=["distilbert/distilgpt2", "gpt2", "openai-community/gpt2-medium"],
            index=0
        )
    elif model_category == "Google Gemma (Requires HF Token)":
        model_name_input = st.sidebar.selectbox(
            "Select Gemma Variant",
            options=["google/gemma-2-2b-it", "google/gemma-2b-it", "google/gemma-2b"],
            index=0
        )
    elif model_category == "Meta LLaMA / TinyLlama":
        model_name_input = st.sidebar.selectbox(
            "Select LLaMA Variant",
            options=["meta-llama/Llama-3.2-1B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
            index=1
        )
    elif model_category == "Microsoft Phi":
        model_name_input = st.sidebar.selectbox(
            "Select Phi Variant",
            options=["microsoft/Phi-3-mini-4k-instruct", "microsoft/phi-2"],
            index=0
        )
    else:
        model_name_input = st.sidebar.text_input("Enter HuggingFace Model Repo ID:", value="Qwen/Qwen2.5-0.5B-Instruct")

    hf_token_input = st.sidebar.text_input(
        "HuggingFace Access Token (Optional)",
        type="password",
        help="Required for gated repositories like Google Gemma 2 or Meta LLaMA."
    )

    device_choice = st.sidebar.radio(
        "Execution Device",
        options=["cuda", "cpu"] if has_cuda else ["cpu"],
        index=0
    )

    max_tokens = st.sidebar.slider("Max Generated Tokens", min_value=20, max_value=300, value=80, step=10)
    budget_slots = st.sidebar.slider("KV-Cache Memory Capacity (B Tokens)", min_value=10, max_value=80, value=25, step=5,
                                    help="Maximum number of historical token KV slots allowed in GPU RAM before DAA 0-1 Knapsack pruning activates.")

    temperature = st.sidebar.slider("Sampling Temperature", min_value=0.1, max_value=1.2, value=0.7, step=0.1)


    # Prompt Presets
    sample_prompts = [
        "Explain dynamic programming in two simple sentences.",
        "In modern generative artificial intelligence, the Key-Value cache memory bottleneck occurs when",
        "Design and Analysis of Algorithms is a foundational computer science discipline that",
        "The primary difference between greedy algorithms and dynamic programming lies in how decisions are made,"
    ]

    selected_prompt = st.selectbox("📝 Choose a Starter Prompt or Type Your Own Below:", sample_prompts, index=0)
    user_prompt = st.text_area("Input Prompt to Feed Both LLMs:", value=selected_prompt, height=80)

    # Load model in session state
    if ("runner" not in st.session_state or 
        st.session_state.get("current_model") != model_name_input or 
        st.session_state.get("current_device") != device_choice or
        st.session_state.get("current_token") != hf_token_input):
        with st.spinner(f"⏳ Initializing offline model '{model_name_input}' on {device_choice.upper()}..."):
            try:
                st.session_state["runner"] = RealLLMRunner(model_name=model_name_input, device=device_choice, hf_token=hf_token_input)
                st.session_state["current_model"] = model_name_input
                st.session_state["current_device"] = device_choice
                st.session_state["current_token"] = hf_token_input
                st.toast(f"Model {model_name_input} successfully loaded on {device_choice.upper()}!", icon="✅")
            except Exception as e:
                st.error(f"Could not load '{model_name_input}' locally: {e}. If using Gemma, please ensure you provided a valid HF Token.")

    run_button = st.button("🚀 Run Live Side-by-Side Comparison on GPU", type="primary", use_container_width=True)

    if run_button:
        runner = st.session_state.get("runner")
        if not runner:
            st.error("Model initializing. Please click Run once more.")
        else:
            st.markdown("---")
            st.subheader("🖥️ Live Side-by-Side Model Outputs & Hardware Telemetry")

            # Two columns for the models
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("### ❌ 1. Without Eviction (Standard LLM)")
                st.caption("Standard HuggingFace / PyTorch inference where KV tensors grow infinitely $O(N)$ with every generated token.")
                kpi_l1, kpi_l2, kpi_l3 = st.columns(3)
                box_l1 = kpi_l1.empty()
                box_l2 = kpi_l2.empty()
                box_l3 = kpi_l3.empty()
                text_placeholder_left = st.empty()

            with col_right:
                st.markdown("### ✅ 2. With LLM-KVOps (Combinatorial Eviction)")
                st.caption(f"Real-time 0-1 Knapsack DP pruning keeping KV cache strictly bounded under budget $B = {budget_slots}$ tokens.")
                kpi_r1, kpi_r2, kpi_r3 = st.columns(3)
                box_r1 = kpi_r1.empty()
                box_r2 = kpi_r2.empty()
                box_r3 = kpi_r3.empty()
                text_placeholder_right = st.empty()

            # Placeholders for Real-Time Graphs
            st.markdown("### 📈 Live Real-Time Hardware Resource Graphs")
            graph_col1, graph_col2 = st.columns(2)
            with graph_col1:
                graph_placeholder_vram = st.empty()
            with graph_col2:
                graph_placeholder_latency = st.empty()

            steps_history = []
            baseline_kv_mb = []
            baseline_gpu_mb = []
            baseline_latencies = []
            
            kvops_kv_mb = []
            kvops_gpu_mb = []
            kvops_latencies = []
            tokens_evicted_hist = []

            # ---------------------------------------------------------
            # Simultaneous Side-by-Side Streaming Execution
            # ---------------------------------------------------------
            gen_base = runner.generate_streaming_baseline(
                prompt=user_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            gen_kvops = runner.generate_streaming_kvops(
                prompt=user_prompt,
                max_new_tokens=max_tokens,
                budget_capacity=budget_slots,
                temperature=temperature
            )

            step = 0
            base_done = False
            kvops_done = False

            with st.spinner("Streaming both LLMs side-by-side on GPU in real-time..."):
                while not (base_done and kvops_done):
                    step += 1
                    steps_history.append(step)

                    # Step Baseline
                    if not base_done:
                        try:
                            step_data_b = next(gen_base)
                            baseline_kv_mb.append(step_data_b["kv_cache_mb"])
                            baseline_gpu_mb.append(step_data_b["gpu_allocated_mb"])
                            baseline_latencies.append(step_data_b["step_time_ms"])

                            box_l1.metric("KV Cache Size", f"{step_data_b['kv_cache_mb']:.3f} MB")
                            box_l2.metric("GPU Memory", f"{step_data_b['gpu_allocated_mb']:.1f} MB")
                            box_l3.metric("Step Latency", f"{step_data_b['step_time_ms']:.1f} ms")
                            text_placeholder_left.markdown(f"<div class='stream-box'>{step_data_b['full_text']}</div>", unsafe_allow_html=True)
                            if step_data_b.get("is_eos"):
                                base_done = True
                        except StopIteration:
                            base_done = True

                    # Step KVOps
                    if not kvops_done:
                        try:
                            step_data_k = next(gen_kvops)
                            kvops_kv_mb.append(step_data_k["kv_cache_mb"])
                            kvops_gpu_mb.append(step_data_k["gpu_allocated_mb"])
                            kvops_latencies.append(step_data_k["step_time_ms"])
                            tokens_evicted_hist.append(step_data_k["tokens_evicted"])

                            curr_base_mb = baseline_kv_mb[-1] if baseline_kv_mb else step_data_k["kv_cache_mb"]
                            saved_pct = ((curr_base_mb - step_data_k["kv_cache_mb"]) / curr_base_mb * 100.0) if curr_base_mb > 0 else 0.0

                            box_r1.metric("KV Cache Size", f"{step_data_k['kv_cache_mb']:.3f} MB", delta=f"-{saved_pct:.1f}% Saved")
                            box_r2.metric("GPU Memory", f"{step_data_k['gpu_allocated_mb']:.1f} MB")
                            box_r3.metric("Tokens Evicted", f"{step_data_k['tokens_evicted']}", delta=f"Budget: {budget_slots}")
                            text_placeholder_right.markdown(f"<div class='stream-box'>{step_data_k['full_text']}</div>", unsafe_allow_html=True)
                            if step_data_k.get("is_eos"):
                                kvops_done = True
                        except StopIteration:
                            kvops_done = True

                    # Live Graph Update every 2 steps or on completion
                    if step % 2 == 0 or (base_done and kvops_done):
                        # Graph 1 (VRAM & KV-Cache Size)
                        fig_vram = go.Figure()
                        fig_vram.add_trace(go.Scatter(
                            x=steps_history[:len(baseline_kv_mb)], y=baseline_kv_mb,
                            mode="lines+markers", name="❌ Without Eviction (Baseline)",
                            line=dict(color="#dc3545", width=3)
                        ))
                        fig_vram.add_trace(go.Scatter(
                            x=steps_history[:len(kvops_kv_mb)], y=kvops_kv_mb,
                            mode="lines+markers", name="✅ With LLM-KVOps (DP Pruned)",
                            line=dict(color="#198754", width=3)
                        ))
                        fig_vram.update_layout(
                            title="<b>Live KV-Cache VRAM Allocation (MB) vs Tokens</b>",
                            xaxis_title="Generated Token Step",
                            yaxis_title="KV-Cache Size (MB)",
                            legend=dict(x=0.02, y=0.98),
                            height=380,
                            margin=dict(l=40, r=20, t=40, b=40)
                        )
                        graph_placeholder_vram.plotly_chart(fig_vram, use_container_width=True)

                        # Graph 2 (Latency)
                        fig_lat = go.Figure()
                        fig_lat.add_trace(go.Scatter(
                            x=steps_history[:len(baseline_latencies)], y=baseline_latencies,
                            mode="lines", name="Baseline Latency (ms)",
                            line=dict(color="#6c757d", width=2, dash="dot")
                        ))
                        fig_lat.add_trace(go.Scatter(
                            x=steps_history[:len(kvops_latencies)], y=kvops_latencies,
                            mode="lines+markers", name="LLM-KVOps Latency (ms)",
                            line=dict(color="#0d6efd", width=2)
                        ))
                        fig_lat.update_layout(
                            title="<b>Per-Token Generation Latency (ms)</b>",
                            xaxis_title="Generated Token Step",
                            yaxis_title="Latency (ms)",
                            legend=dict(x=0.02, y=0.98),
                            height=380,
                            margin=dict(l=40, r=20, t=40, b=40)
                        )
                        graph_placeholder_latency.plotly_chart(fig_lat, use_container_width=True)

                    time.sleep(0.01)

            evicted_count = tokens_evicted_hist[-1] if tokens_evicted_hist else 0
            st.success(f"🎉 Live GPU Execution Complete! LLM-KVOps successfully pruned {evicted_count} tokens, keeping GPU memory strictly capped under your budget ceiling of {budget_slots} slots!")

else:
    # -------------------------------------------------------------
    # High-Throughput Long-Context Simulator Mode (up to 128k tokens)
    # -------------------------------------------------------------
    st.sidebar.subheader("Large-Scale Architecture Settings")
    model_sim_choice = st.sidebar.selectbox("Architecture Preset", list(MODEL_PRESETS.keys()), index=0)
    ctx_sim_len = st.sidebar.select_slider("Context Sequence Length", [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], value=8192)
    sim_budget_pct = st.sidebar.slider("Memory Budget %", min_value=20, max_value=80, value=40, step=5)

    sim_prompt = st.text_area("Simulated Long Document Context Prompt:", 
                              value="In modern enterprise cloud infrastructure, processing a 128,000 token legal contract or software codebase causes extreme GPU VRAM memory bottlenecks due to uncompressed KV cache growth.",
                              height=70)

    if st.button("🚀 Run Large-Scale Long-Context Simulation", type="primary", use_container_width=True):
        sim = InferenceSimulator(model_key=model_sim_choice, target_budget_ratio=sim_budget_pct / 100.0)
        sim_res = sim.run_side_by_side_simulation(token_count=ctx_sim_len, sample_text=sim_prompt)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("❌ Baseline VRAM", f"{sim_res['baseline']['vram_gb']:.2f} GB" if sim_res['baseline']['vram_gb'] >= 1.0 else f"{sim_res['baseline']['vram_mb']:.1f} MB", delta="Unbounded O(N)", delta_color="inverse")
        col_b.metric("✅ LLM-KVOps VRAM", f"{sim_res['kvops']['vram_gb']:.2f} GB" if sim_res['kvops']['vram_gb'] >= 1.0 else f"{sim_res['kvops']['vram_mb']:.1f} MB", delta=f"-{sim_res['kvops']['memory_saved_pct']:.1f}% Saved")
        col_c.metric("🔥 Memory Saved", f"{(sim_res['baseline']['vram_gb'] - sim_res['kvops']['vram_gb']):.2f} GB")
        col_d.metric("⚡ Decision Latency", f"{sim_res['kvops']['latency_overhead_ms']:.2f} ms")

        # Plotly Time-Series Graph
        ts = sim_res["time_series"]
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=ts["steps"], y=ts["baseline_vram_mb"], mode="lines+markers", name="Without Eviction (Full Baseline)", line=dict(color="#dc3545", width=3)))
        fig_sim.add_trace(go.Scatter(x=ts["steps"], y=ts["sliding_window_vram_mb"], mode="lines+markers", name="Sliding Window Baseline", line=dict(color="#fd7e14", width=2, dash="dash")))
        fig_sim.add_trace(go.Scatter(x=ts["steps"], y=ts["kvops_vram_mb"], mode="lines+markers", name="With LLM-KVOps (Combinatorial Eviction)", line=dict(color="#198754", width=3)))

        fig_sim.update_layout(
            title=f"<b>GPU VRAM Footprint vs Generation Sequence Length ({model_sim_choice})</b>",
            xaxis_title="Generation Sequence Length (Tokens)",
            yaxis_title="GPU VRAM Allocated (MB)",
            legend=dict(x=0.02, y=0.98),
            height=450
        )
        st.plotly_chart(fig_sim, use_container_width=True)
