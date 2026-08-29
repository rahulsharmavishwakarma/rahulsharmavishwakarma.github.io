---
layout: post
title: "Serving LLMs: what actually decides your latency"
date: 2026-08-30 01:30:00 +0530
description: "TTFT, prefill vs decode, the KV-cache memory bill, quantization, multi-LoRA serving, and how vLLM, SGLang, llama.cpp and MLX run them across NVIDIA, AMD and Apple silicon."
tags: [inference, serving, llm]
categories: [articles]
featured: true
---

Training a model is a throughput problem. Serving one is a latency problem under concurrency — and almost every interesting engineering decision in an inference stack falls out of that difference. This is a working note on the pieces that actually decide what your users feel: the phases of a request, the metrics worth measuring, the memory the KV cache quietly eats, what changes when you serve fine-tuned models instead of base ones, and how the popular runtimes — vLLM, SGLang, llama.cpp, MLX — approach the same problem.

> **TL;DR** — Prefill is compute-bound; decode is bandwidth-bound. **TTFT** is a prefill problem, **TPOT** is a bandwidth problem, the **KV cache** is a memory bill that grows with context × concurrency, and **multi-LoRA** decides whether your GPU count scales with your number of fine-tuned variants. Everything below is those four sentences, unpacked.

## 1. A request has two phases, and they are not alike

Every LLM request runs in two distinct phases.

**Prefill** processes the entire prompt — system message, context, user question — in one pass. Because all prompt tokens are available at once, this is a large parallel matrix multiply: **compute-bound**. Its cost grows roughly linearly with prompt length for the dense work, plus the quadratic attention term over the sequence.

**Decode** generates the reply one token at a time. Each new token needs a full pass over the model, which means reading *every weight* from GPU memory to produce a single token. That makes decode **memory-bandwidth-bound** — the arithmetic units spend most of their time waiting on data.

| | Prefill | Decode |
|---|---|---|
| **What runs** | whole prompt, one pass | one token per pass |
| **Work shape** | large parallel matmuls | full weight read per token |
| **Bottleneck** | GPU *compute* | GPU *memory bandwidth* |
| **Cost scales with** | prompt length (linear + quadratic attention) | output length × weight size |

This split explains most of what follows: optimizations for serving are really optimizations for one phase or the other.

<figure class="paper-figure">
<svg viewBox="0 0 760 242" role="img" aria-label="Request timeline: prefill phase, TTFT marker, decode tokens with TPOT, end-to-end span" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
  <!-- phase labels -->
  <text x="115" y="88" text-anchor="middle" font-size="12" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">PREFILL</text>
  <text x="115" y="104" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">compute-bound</text>
  <text x="455" y="88" text-anchor="middle" font-size="12" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">DECODE</text>
  <text x="455" y="104" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">memory-bandwidth-bound</text>

  <!-- prefill block -->
  <rect x="42" y="118" width="146" height="30" rx="4" fill="var(--global-theme-color)"/>
  <text x="115" y="138" text-anchor="middle" font-size="11.5" font-family="JetBrains Mono, monospace" fill="var(--site-bg, #161617)" font-weight="600">1,000 prompt tokens</text>

  <!-- TTFT marker -->
  <line x1="196" y1="60" x2="196" y2="176" stroke="var(--global-text-color-light)" stroke-width="1.2" stroke-dasharray="4 4"/>
  <text x="196" y="52" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">TTFT</text>

  <!-- decode tokens -->
  <g fill="var(--global-theme-color)">
    <rect x="212" y="125" width="11" height="16" rx="2.5"/><rect x="234" y="125" width="11" height="16" rx="2.5"/>
    <rect x="256" y="125" width="11" height="16" rx="2.5"/><rect x="278" y="125" width="11" height="16" rx="2.5"/>
    <rect x="300" y="125" width="11" height="16" rx="2.5"/><rect x="322" y="125" width="11" height="16" rx="2.5"/>
    <rect x="344" y="125" width="11" height="16" rx="2.5"/><rect x="366" y="125" width="11" height="16" rx="2.5"/>
    <rect x="388" y="125" width="11" height="16" rx="2.5"/><rect x="410" y="125" width="11" height="16" rx="2.5"/>
    <rect x="432" y="125" width="11" height="16" rx="2.5"/>
  </g>
  <text x="462" y="138" font-size="13" fill="var(--global-text-color-light)">···</text>
  <rect x="486" y="125" width="11" height="16" rx="2.5" fill="none" stroke="var(--global-theme-color)" stroke-width="1.5"/>
  <text x="508" y="138" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">last token</text>

  <!-- TPOT bracket -->
  <path d="M234,164 L234,170 L267,170 L267,164" fill="none" stroke="var(--global-text-color-light)" stroke-width="1.2"/>
  <text x="250" y="186" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">TPOT ≈ 20 ms</text>

  <!-- E2E bracket -->
  <path d="M42,206 L42,212 L697,212 L697,206" fill="none" stroke="var(--global-text-color)" stroke-width="1.2"/>
  <text x="370" y="228" text-anchor="middle" font-size="11.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">end-to-end ≈ TTFT + (n_out − 1) × TPOT</text>
</svg>
<figcaption>Figure 1 — Anatomy of a request: prefill runs the whole prompt as one parallel, compute-bound pass; decode then emits tokens one at a time, bounded by memory bandwidth.</figcaption>
</figure>

## 2. The metrics that matter

The four numbers worth putting on a dashboard:

- **TTFT — time to first token.** Request sent → first streamed token. Dominated by prefill (plus queueing). This is *perceived* latency: chat UIs feel instant at 300ms TTFT even if the full answer takes twenty seconds.
- **TPOT — time per output token** (sometimes ITL, inter-token latency). The steady-state rhythm of the stream after the first token. Its inverse is your per-request decode speed.
- **Throughput** — total tokens/second or requests/second across all concurrent users. What the *operator* optimizes — in direct tension with per-request latency, because the trick that raises throughput (bigger batches) usually raises TPOT too.
- **Goodput** — throughput that *counts*: only requests meeting your latency SLO. Serving at 95% GPU utilization with 30% of requests over TTFT budget can be worse than running cooler with better scheduling.

E2E / TTOT — end-to-end time — is roughly:

$$
t_{e2e} \approx t_{TTFT} + (n_{out} - 1) \cdot t_{TPOT}
$$

**Worked example.** Say your stack reports TTFT 250ms and TPOT 20ms for a request with a 1,000-token prompt and a 512-token completion:

- TTFT = 250ms
- decode of 511 remaining tokens ≈ 511 × 20ms ≈ 10.2s
- E2E ≈ 10.5s

One number — "tokens per second" — hides all of this. A stack can halve TPOT and leave TTFT untouched; whether that helps depends on whether your users are reading a stream or waiting for a JSON blob.

## 3. Model size: the memory bill

The weights set the floor. A useful rule: **gigabytes ≈ parameters × bytes-per-parameter**.

| Model | FP16 (2B) | INT8 (1B) | INT4 (0.5B) |
|---|---|---|---|
| 7B | 14 GB | 7 GB | ~3.5 GB |
| 13B | 26 GB | 13 GB | ~6.5 GB |
| 70B | 140 GB | 70 GB | ~35 GB |

Quantization trades a little quality for a lot of memory — and memory buys *speed* here, because decode is bandwidth-bound. A back-of-envelope decode ceiling for one stream:

$$
\text{tokens/s} \;\lesssim\; \frac{\text{memory bandwidth}}{\text{weight bytes}}
$$

A 7B model at FP16 (14 GB) on an A100 (2 TB/s) tops out near 140 tokens/s *per stream* — real-world numbers land lower. Quantize to INT4 (3.5 GB) and the same bandwidth now feeds ~4× more streams, or a faster single stream. This is why quantization is a serving decision, not just a "fit it on the GPU" decision.

**The quantization formats you'll actually meet:**

- **GPTQ / AWQ** — 4-bit weight formats built for GPU serving; behind most INT4 checkpoints on the Hub.
- **GGUF k-quants** — llama.cpp's format family, tuned for CPU / Metal execution from one file.
- **FP8** — hardware-accelerated on Hopper-class NVIDIA chips and newer; near-FP16 quality at half the memory and double the effective bandwidth.
- **MLX 4-bit** — the Apple-side equivalent, computed on unified memory.

The caveat that matters: quantization quality is *eval-dependent*. Always re-run your own evals on the quantized checkpoint — a 4-bit model that aces its benchmarks can still lose your production edge cases.

## 4. The KV cache: the hidden state that eats your GPU

Transformers are autoregressive: each new token attends over *everything before it*. Recomputing keys and values for the whole history at every step would be quadratic and absurd, so runtimes cache them per sequence. That cache is the **KV cache**, and after the weights it is the largest consumer of GPU memory — often *the* largest at real context lengths.

Its size per token:

$$
\text{KV bytes/token} = 2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times \text{bytes}
$$

(the 2 is K and V; modern models use GQA/MQA, so `n_kv_heads` is much smaller than query heads).

**Worked numbers:**

- **Llama-2-7B**: 32 layers × 32 heads × 128 head-dim, FP16 → 2 × 32 × 32 × 128 × 2 ≈ **512 KiB/token**. A full 4,096-token context: ~2 GB *per sequence*.
- **Llama-3-70B** (GQA, 8 KV heads): 2 × 80 × 8 × 128 × 2 ≈ **320 KiB/token** → ~1.25 GB at 4k tokens — *per concurrent request*.

The key mental model: **weights are a fixed cost; the KV cache scales with context × concurrency.** Serve ten streams of 8k context on that 7B and the cache alone is nearly triple the weights. This is why "what's the max context I can serve?" is really "how much memory is left after weights and KV at my target batch size?"

<figure class="paper-figure">
<svg viewBox="0 0 760 318" role="img" aria-label="Stacked VRAM bars: fixed weights versus KV cache growing with context and concurrency" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
  <!-- gridlines: 0,20,40,60,80 GB  (2.95 px/GB, baseline y=270) -->
  <g stroke="var(--global-divider-color)" stroke-width="1">
    <line x1="70" y1="270" x2="720" y2="270"/>
    <line x1="70" y1="211" x2="720" y2="211" stroke-dasharray="3 5"/>
    <line x1="70" y1="152" x2="720" y2="152" stroke-dasharray="3 5"/>
    <line x1="70" y1="93"  x2="720" y2="93"  stroke-dasharray="3 5"/>
  </g>
  <g font-size="10.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">
    <text x="62" y="274" text-anchor="end">0</text>
    <text x="62" y="215" text-anchor="end">20</text>
    <text x="62" y="156" text-anchor="end">40</text>
    <text x="62" y="97"  text-anchor="end">60</text>
    <text x="70" y="28"  text-anchor="start">GB of VRAM</text>
  </g>

  <!-- legend (left side, clear of bar labels) -->
  <g font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">
    <rect x="170" y="14" width="12" height="12" fill="#8a8a8f"/><text x="188" y="24">weights (7B FP16)</text>
    <rect x="368" y="14" width="12" height="12" fill="var(--global-theme-color)"/><text x="386" y="24">KV cache</text>
  </g>

  <!-- bar A: 1 req, 4k ctx — weights 14GB (h41), KV 2GB (h6) -->
  <rect x="150" y="229" width="90" height="41" fill="#8a8a8f"/>
  <rect x="150" y="223" width="90" height="6"  fill="var(--global-theme-color)"/>
  <text x="195" y="216" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">16 GB</text>
  <text x="195" y="292" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">1 req · 4k ctx</text>

  <!-- bar B: 8 reqs, 8k ctx — weights 14GB (h41), KV 32GB (h94) -->
  <rect x="350" y="229" width="90" height="41" fill="#8a8a8f"/>
  <rect x="350" y="135" width="90" height="94" fill="var(--global-theme-color)"/>
  <text x="395" y="128" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">46 GB</text>
  <text x="395" y="180" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--site-bg, #161617)" font-weight="600">KV 32 GB</text>
  <text x="395" y="292" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">8 reqs · 8k ctx</text>

  <!-- bar C: 16 reqs, 16k ctx, FP8 KV — weights 14GB (h41), KV 64GB (h189) -->
  <rect x="550" y="229" width="90" height="41" fill="#8a8a8f"/>
  <rect x="550" y="40"  width="90" height="189" fill="var(--global-theme-color)"/>
  <text x="595" y="30" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">78 GB</text>
  <text x="595" y="122" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--site-bg, #161617)" font-weight="600">KV 64 GB</text>
  <text x="595" y="138" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="var(--site-bg, #161617)">FP8</text>
  <text x="595" y="292" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">16 reqs · 16k ctx</text>

  <text x="370" y="312" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">one 80 GB GPU is full before you reach 16k × 16 — and this is a 7B model</text>
</svg>
<figcaption>Figure 2 — The VRAM bill for a 7B FP16 model: weights are a fixed cost, the KV cache scales with context × concurrency until it dwarfs everything else. FP8 KV cache halves the slope.</figcaption>
</figure>

The runtime-side mitigations, briefly:

- **PagedAttention** (vLLM) — store the cache in fixed-size pages instead of contiguous blocks, eliminating the fragmentation that used to waste 60–80% of KV memory; also the basis of its prefix caching.
- **GQA / MQA** — fewer KV heads (an architecture choice) shrink the cache 4–8× for a small quality cost.
- **Quantized KV cache** — 8-bit (or lower) K/V halves the bill again.
- **Sliding-window attention** — cap how far back a token attends; bounded cache by construction.

## 5. Context length at inference time

"Supports 128k context" is a model property; *serving* 128k is an economics problem:

- **Prefill compute** grows quadratically with sequence length — the first token of a 100k-token prompt costs far more than a thousand 1k prompts.
- **KV memory** grows linearly, and stays resident for the whole generation.
- Long prompts *amplify* every queueing effect, because a long prefill blocks the compute everyone else needs.

Positional behavior matters too: models trained at 4k don't magically attend well at 32k. Techniques like RoPE scaling (linear, NTK-aware, YaRN) stretch positional encodings so the model *extrapolates* — this is what most "long-context" releases actually ship.

The practical lever is **prefix caching**: identical prompt prefixes (system prompts, few-shot blocks, retrieved RAG context) produce identical KV, so cache and reuse it instead of recomputing per request. For repeated-prefix workloads this is the single biggest TTFT win available — measured results on Apple Silicon showed cached-context TTFT dropping from 21.7s to 0.78s. SGLang generalizes the idea with **RadixAttention**: cached prefixes live in a radix tree, so any two requests sharing any prefix branch reuse it automatically.

## 6. Continuous batching

Classic serving batches a fixed set of requests and waits for the longest generation before admitting new ones — every straggler holds the whole batch hostage. **Continuous batching** (iteration-level scheduling) admits and retires requests *at every decode step*: finished sequences free their KV pages immediately and queued prompts slot in. It's the reason modern stacks report several-fold higher throughput than naive batching, and by now it's table stakes — every runtime below does it.

<figure class="paper-figure">
<svg viewBox="0 0 780 372" role="img" aria-label="Gantt comparison: static batching holds the batch for stragglers; continuous batching admits requests as slots free" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
  <!-- ===== static ===== -->
  <text x="130" y="28" font-size="12" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">STATIC BATCHING</text>
  <text x="730" y="28" text-anchor="end" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">batch ends →</text>
  <line x1="730" y1="38" x2="730" y2="158" stroke="var(--global-text-color-light)" stroke-width="1.2" stroke-dasharray="4 4"/>
  <g font-size="10.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">
    <text x="120" y="68" text-anchor="end">R1</text><text x="120" y="93" text-anchor="end">R2</text>
    <text x="120" y="118" text-anchor="end">R3</text><text x="120" y="143" text-anchor="end">R4</text>
  </g>
  <g fill="var(--global-theme-color)">
    <rect x="130" y="55"  width="280" height="16" rx="3"/>
    <rect x="130" y="80"  width="380" height="16" rx="3"/>
    <rect x="130" y="105" width="490" height="16" rx="3"/>
    <rect x="130" y="130" width="600" height="16" rx="3"/>
  </g>
  <g fill="none" stroke="var(--global-divider-color)" stroke-width="1.2">
    <rect x="414" y="55"  width="312" height="16" rx="3"/>
    <rect x="514" y="80"  width="212" height="16" rx="3"/>
    <rect x="624" y="105" width="102" height="16" rx="3"/>
  </g>
  <text x="570" y="67" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">idle · GPU waits</text>
  <text x="680" y="178" text-anchor="end" font-size="10.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">R5 queued behind the straggler</text>

  <!-- ===== continuous ===== -->
  <text x="130" y="212" font-size="12" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">CONTINUOUS BATCHING</text>
  <text x="730" y="212" text-anchor="end" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">same wall clock →</text>
  <g font-size="10.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">
    <text x="120" y="252" text-anchor="end">R1</text><text x="120" y="277" text-anchor="end">R2</text>
    <text x="120" y="302" text-anchor="end">R3</text><text x="120" y="327" text-anchor="end">R4</text>
  </g>
  <g fill="var(--global-theme-color)">
    <rect x="130" y="239" width="280" height="16" rx="3"/>
    <rect x="130" y="264" width="380" height="16" rx="3"/>
    <rect x="130" y="289" width="490" height="16" rx="3"/>
    <rect x="130" y="314" width="600" height="16" rx="3"/>
  </g>
  <g fill="var(--site-bg, #161617)" stroke="var(--global-theme-color)" stroke-width="1.5">
    <rect x="418" y="239" width="172" height="16" rx="3"/>
    <rect x="518" y="264" width="182" height="16" rx="3"/>
  </g>
  <g font-size="10" font-family="JetBrains Mono, monospace" fill="var(--global-theme-color)">
    <text x="424" y="251">R5 admitted</text>
    <text x="524" y="276">R6 admitted</text>
  </g>

  <!-- legend -->
  <g font-size="10.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">
    <rect x="130" y="350" width="12" height="10" fill="var(--global-theme-color)"/><text x="148" y="359">generating</text>
    <rect x="238" y="350" width="12" height="10" fill="var(--site-bg, #161617)" stroke="var(--global-theme-color)" stroke-width="1.5"/><text x="256" y="359">admitted mid-cycle</text>
    <rect x="392" y="350" width="12" height="10" fill="none" stroke="var(--global-divider-color)" stroke-width="1.2"/><text x="410" y="359">idle slot</text>
  </g>
</svg>
<figcaption>Figure 3 — Static vs continuous batching: with iteration-level scheduling a finished sequence hands its slot to the next request immediately, instead of every row waiting on the longest one.</figcaption>
</figure>

## 7. Serving fine-tuned models: LoRA and the multi-adapter problem

Fine-tuning produces variants; serving them is a separate problem. **LoRA** freezes the base weights and learns a low-rank delta: $\Delta W = BA$ with rank $r \ll d$, applied to attention (and sometimes MLP) projections. Practical adapters are 0.1–1% the size of the base — a rank-16 adapter on a 7B model is tens of megabytes against the base's 14 GB.

You can serve those variants two ways:

| Approach | How | Memory (50 variants) | Latency | Use when |
|---|---|---|---|---|
| **Merge** | bake ΔW into the weights | 50 × 14 GB ≈ **700 GB** | best — zero overhead | one or two variants |
| **Dynamic adapters** | one base, attach per request | 14 GB + 50 × ~40 MB ≈ **16 GB** | tiny routing overhead | many per-tenant variants |

That second shape is **multi-LoRA serving**, and doing it well is genuinely hard: requests on different adapters must batch *together* on the same base weights, adapter matrices must be resident and swapped with near-zero overhead, and the KV cache must be shared/paged across everything. The S-LoRA paper framed the memory problem — a *unified pool* that pages base weights and adapters together — and modern engines implement versions of it:

- **vLLM** serves many adapters on one base with hot reload and per-request routing — the most mature option, and my default for this pattern.
- **SGLang** matches it and, in several published benchmarks, edges ahead at scale, especially when tenants share long system prompts (its radix tree amortizes those across adapters).

The rule of thumb: one or two fine-tunes → merge and serve as separate models. Many per-user or per-tenant variants → one base + multi-LoRA serving, or your GPU bill scales with your customer count.

<figure class="paper-figure">
<svg viewBox="0 0 760 348" role="img" aria-label="Merged variants versus shared base with per-request LoRA adapters" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
  <!-- ===== left: merged ===== -->
  <text x="185" y="26" text-anchor="middle" font-size="12" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">MERGED — one model per variant</text>
  <g>
    <rect x="75"  y="80" width="70" height="150" rx="6" fill="#8a8a8f"/>
    <rect x="175" y="80" width="70" height="150" rx="6" fill="#8a8a8f"/>
    <rect x="275" y="80" width="70" height="150" rx="6" fill="#8a8a8f"/>
  </g>
  <g fill="var(--global-theme-color)">
    <rect x="75"  y="66" width="70" height="10" rx="3"/>
    <rect x="175" y="66" width="70" height="10" rx="3"/>
    <rect x="275" y="66" width="70" height="10" rx="3"/>
  </g>
  <g font-size="10.5" font-family="JetBrains Mono, monospace" fill="var(--site-bg, #161617)" font-weight="600">
    <text x="110" y="150" text-anchor="middle">7B</text>
    <text x="210" y="150" text-anchor="middle">7B</text>
    <text x="310" y="150" text-anchor="middle">7B</text>
  </g>
  <g font-size="10.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)" text-anchor="middle">
    <text x="110" y="250">variant A</text><text x="210" y="250">variant B</text><text x="310" y="250">variant C</text>
  </g>
  <text x="185" y="286" text-anchor="middle" font-size="11.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">50 variants ≈ 700 GB</text>
  <text x="185" y="304" text-anchor="middle" font-size="10.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">zero runtime overhead, linear memory</text>

  <!-- divider -->
  <line x1="385" y1="40" x2="385" y2="300" stroke="var(--global-divider-color)" stroke-width="1"/>

  <!-- ===== right: shared base ===== -->
  <text x="572" y="26" text-anchor="middle" font-size="12" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">SHARED BASE + ADAPTERS</text>
  <!-- incoming requests -->
  <g font-size="10.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">
    <text x="402" y="70">req · A</text>
    <text x="402" y="103">req · B</text>
    <text x="402" y="136">req · C</text>
  </g>
  <!-- adapters -->
  <g fill="var(--global-theme-color)">
    <rect x="458" y="58"  width="76" height="18" rx="4"/>
    <rect x="458" y="91"  width="76" height="18" rx="4"/>
    <rect x="458" y="124" width="76" height="18" rx="4"/>
  </g>
  <g font-size="10" font-family="JetBrains Mono, monospace" fill="var(--site-bg, #161617)" font-weight="600">
    <text x="496" y="70"  text-anchor="middle">LoRA A</text>
    <text x="496" y="103" text-anchor="middle">LoRA B</text>
    <text x="496" y="136" text-anchor="middle">LoRA C</text>
  </g>
  <g stroke="var(--global-theme-color)" stroke-width="1.4" stroke-dasharray="3 3" fill="none">
    <path d="M534,67  L590,67  L590,148"/>
    <path d="M534,100 L578,100 L578,148"/>
    <path d="M534,133 L566,133 L566,148"/>
  </g>
  <g fill="var(--global-theme-color)">
    <path d="M586,148 L590,156 L594,148 Z"/>
    <path d="M574,148 L578,156 L582,148 Z"/>
    <path d="M562,148 L566,156 L570,148 Z"/>
  </g>
  <!-- base -->
  <rect x="500" y="158" width="150" height="110" rx="6" fill="#8a8a8f"/>
  <text x="575" y="208" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="var(--site-bg, #161617)" font-weight="600">7B base — 14 GB</text>
  <text x="575" y="226" text-anchor="middle" font-size="9.5" font-family="JetBrains Mono, monospace" fill="var(--site-bg, #161617)">shared KV pages</text>
  <text x="572" y="286" text-anchor="middle" font-size="11.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color)">50 variants ≈ 16 GB</text>
  <text x="572" y="304" text-anchor="middle" font-size="10.5" font-family="JetBrains Mono, monospace" fill="var(--global-text-color-light)">tiny per-request routing overhead</text>
</svg>
<figcaption>Figure 4 — Serving fine-tuned variants: merging duplicates the base model per variant; shared-base multi-LoRA keeps one base in memory and pages small adapters per request.</figcaption>
</figure>

## 8. The runtimes, at a high level

All four below are excellent; they optimize different points on the same curve.

### vLLM

The production default on NVIDIA hardware. PagedAttention for the KV cache, continuous batching, prefix caching, mature multi-LoRA serving with hot-swappable adapters. If you're serving a popular open model to many users on GPUs, start here.

### SGLang

Built around RadixAttention: cached prefixes in a radix tree shared across requests. That makes it exceptional for agentic and structured workloads — multi-turn tool loops, shared system prompts, RAG pipelines where everything shares a prefix. Also very fast at constrained decoding (guaranteed-JSON outputs), which agent stacks love. Benchmarks in 2026 frequently show it matching or beating vLLM, particularly on prefix-heavy and multi-LoRA-at-scale loads.

### llama.cpp

The everything-runs-everywhere runtime: GGUF quantized weights, CPU execution with GPU offload, Metal on macOS. Single-stream latency on modest hardware is remarkable, and it's the right answer for local tools, edge boxes, and embedded deployments. High-concurrency throughput is not its game.

### MLX

Apple's array framework for Apple Silicon (serve via mlx-lm or vllm-mlx). Unified memory means the RAM/GPU split doesn't exist: a 64 GB Mac can hold a 70B model that would need two data-center GPUs elsewhere. Recent benchmarks show it competitive natively, with prefix caching delivering dramatic TTFT wins on cached contexts. The right choice for Mac-local serving and Apple-first products.

| Pick | When |
|---|---|
| vLLM | Multi-user GPU serving, multi-LoRA, the safe default |
| SGLang | Agentic loops, shared prefixes, structured/JSON output |
| llama.cpp | Local, edge, CPU/mixed hardware, single-stream latency |
| MLX | Apple Silicon, unified-memory serving on Macs |

## 9. The hardware: the third axis

Runtimes don't run in the abstract. The same model, same version, same settings lands *different numbers* on different silicon, because kernel coverage and memory systems differ — so hardware is a first-class serving decision, not a procurement detail.

### NVIDIA (CUDA)

The default gravity of the ecosystem: every runtime above targets it first, kernel coverage is deepest, and the memory story (HBM bandwidth, NVLink for multi-GPU) is what most serving math assumes. Two NVIDIA-specific layers sit above the runtimes:

- **TensorRT-LLM** — NVIDIA's hand-tuned kernel and engine library; often the fastest raw CUDA path, paid for in a heavier build-and-version matrix.
- **Dynamo** — NVIDIA's datacenter-scale serving framework, and the clearest expression of the prefill/decode split from §1: **disaggregated serving** runs prefill and decode as *independently scalable GPU pools*, transfers the KV state between them, and routes with KV-cache awareness. It orchestrates TensorRT-LLM, vLLM, and SGLang as backends — prefill and decode simply stop fighting over the same GPUs. This only earns its complexity at serious scale, but it's where big fleets are going.

Newer chips (Hopper, Blackwell) add FP8 and FP4 tensor-core paths — the hardware half of the quantization story above.

### AMD (ROCm)

The credible second source. The MI300X puts **192 GB of HBM3 on one card**, which changes the sizing math — a 70B FP16 model fits with room for KV to spare. vLLM and SGLang ship ROCm builds, llama.cpp runs via ROCm/HIP or Vulkan, and PyTorch support is real. The honest caveat: kernel coverage and edge-case polish lag CUDA by quarters, so expect rougher edges at exotic configurations — and validate *your model on your card* before committing a fleet.

### Apple Silicon (MLX)

Covered in §8; the hardware note belongs here too — **unified memory removes the RAM/VRAM split entirely**, which is *why* a single Mac can serve models that need multi-GPU rigs elsewhere, at bandwidth numbers well below data-center GPUs.

| | NVIDIA | AMD | Apple |
|---|---|---|---|
| Ecosystem | CUDA — deepest, default | ROCm — maturing fast | MLX — Apple-only |
| Serve via | TensorRT-LLM, vLLM, SGLang, Dynamo | vLLM / SGLang ROCm builds | mlx-lm, vllm-mlx |
| Memory angle | HBM + NVLink pooling | huge HBM per card (MI300X: 192 GB) | unified memory |
| Quant paths | FP8 / FP4 tensor cores | growing | 4-bit MLX |

The practical rule: **pick the runtime and the hardware together, not sequentially.** "vLLM on the cheapest GPUs," "SGLang on MI300X," and "llama.cpp on a Mac mini" are three different products — and Dynamo-style disaggregation only earns its keep at datacenter scale.

## 10. The pluggable layer: cache extensions, quant backends, speculative decoding

Modern engines are platforms — the kernel is only the base, and a set of loadable optimizations rides on top. Three matter most in practice.

### KV-cache extensions — LMCache

**LMCache** adds a cache *layer* around the runtime's KV store: it persists and shares KV state **across requests and across instances**, tiered from GPU → CPU DRAM → local NVMe. The payoff is TTFT on repeated content — chat history, the same RAG corpus, multi-turn agent state — because the prefix's KV is *loaded*, not recomputed, and a second replica can reuse what the first one computed. vLLM integrates it through its KV-connector API, which is also the pattern disaggregated stores (Mooncake-style) build on. If your workload re-reads the same context all day — and most agent stacks do — this is the single highest-leverage extension to try.

### Pluggable quantization backends

vLLM's `--quantization` flag is a plugin point, not a single feature: `fp8`, `int8` (weight-and-activation), `awq`, `gptq`, `compressed-tensors`, even `gguf` checkpoints — each backed by different kernels (e.g. Marlin kernels for fast INT4 dequant). Two practical notes:

- **Match the format to the silicon** — FP8 wants Hopper-or-newer tensor cores; INT4/AWQ is the portable choice; `compressed-tensors` is the output of the llm-compressor pipeline and the smoothest path if you quantize yourself.
- **The backend changes the kernels, not just the weights** — the same INT4 checkpoint can serve at noticeably different speeds under different kernel paths, so benchmark the *flag*, not just the checkpoint.

### Speculative decoding

Decode is bandwidth-bound — so amortize it. **Speculative decoding** drafts *k* candidate tokens cheaply, then verifies them against the big model in one pass: accept what matches, regenerate from the first mismatch. Same output distribution (verification is exact, so it's lossless), but up to *k* tokens per forward pass.

Draft sources, cheapest first:

- **Prompt lookup / n-gram** — copy spans from the prompt itself; free, and shockingly good for summarization, editing, and code.
- **Medusa heads** — extra decoding heads on the base model predicting several future tokens at once.
- **EAGLE-style drafters** — a tiny model drafting in feature space; the current quality/acceptance frontier.
- **A small draft model** — the classic setup; pick one from the same family.

The catch: the win scales with the **acceptance rate**. Predictable output (code continuation, structured edits, summarization) accepts often and flies; high-entropy creative text rejects drafts and can even lose a little throughput. vLLM, SGLang, and llama.cpp (draft-model mode) all support it — measure on your traffic before betting on it.

## 11. A practical checklist

1. **Measure TTFT and TPOT separately** — one SLO each. A single "latency" number will mislead you.
2. **Do the KV math at real concurrency** — context × batch × KV-bytes-per-token, on top of weights. This decides your GPU before any tuning does.
3. **Enable prefix caching** — if your prompts share any structure, it's the cheapest TTFT win available; reach for LMCache when the sharing crosses requests or instances.
4. **Quantize deliberately** — weights (GPTQ/AWQ INT4, FP8) for bandwidth, KV (FP8) for long context; measure quality on *your* evals, not vibes.
5. **Try speculative decoding where output is predictable** — code, edits, summaries; near-free tokens when the acceptance rate is high.
6. **Decide merge-vs-adapter early** — it's an architecture decision, not a deployment detail.
7. **Load-test with realistic prompt mixes** — long prompts change the bottleneck from decode to prefill and queueing.
8. **Choose hardware and runtime as one decision** — model × quant × runtime × vendor all constrain each other; check the matrix before you rent.

The unifying idea: serving is memory management under a latency budget. Once you see prefill, decode, and the KV cache as separate line items, every runtime feature — paging, radix trees, adapter pooling — reads as a solution to a specific line of that bill.

## Further reading

- [vLLM — Efficient Memory Management for LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [NVIDIA Dynamo — disaggregated inference serving framework](https://github.com/ai-dynamo/dynamo)
- [LMCache — KV cache layer for LLM serving](https://github.com/LMCache/LMCache)
- [vLLM — speculative decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html)
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285)
- [SGLang — RadixAttention](https://github.com/sgl-project/sglang)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [MLX — array framework for Apple Silicon](https://github.com/ml-explore/mlx)
- [LLM inference servers compared — TensorFoundry](https://tensorfoundry.io/blog/llm-inference-servers-compared)
- [vLLM vs SGLang vs TensorRT-LLM vs Ollama (2026) — LeetLLM](https://leetllm.com/blog/llm-inference-engine-comparison-2026)
- [vLLM or llama.cpp? — Red Hat Developer](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case)
- [Native LLM inference at scale on Apple Silicon (arXiv)](https://arxiv.org/html/2601.19139v1)

## Cite this article

If this note was useful in your own writing, cite it as:

```bibtex
@misc{sharma2026servingllms,
  title        = {Serving LLMs: what actually decides your latency},
  author       = {Sharma, Rahul},
  year         = {2026},
  month        = {August},
  howpublished = {\url{https://rahulsharmavishwakarma.github.io/blog/2026/serving-llms-in-production/}},
  note         = {Online; accessed \today}
}
```

Or in APA style:

> Sharma, R. (2026, August 30). *Serving LLMs: what actually decides your latency*. Rahul Sharma. https://rahulsharmavishwakarma.github.io/blog/2026/serving-llms-in-production/
