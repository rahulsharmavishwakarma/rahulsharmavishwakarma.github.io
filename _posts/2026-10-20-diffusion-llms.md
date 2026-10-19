---
layout: post
title: "Diffusion LLMs: what changes when you stop generating one token at a time"
date: 2026-10-20 10:00:00 +0530
description: "How text diffusion models work, why they swap the memory-bound decode loop for a compute-bound one, and what the papers behind Mercury, LLaDA2.x and DiffusionGemma actually show."
tags: [llm, diffusion, inference]
categories: [articles]
featured: true
published: true
---

{% include diffusion/shared.html %}

Every mainstream LLM you have used writes left to right, one token at a time. That is a modeling choice, not a law of nature, and it is the reason decode is the slow, memory-bound half of the serving problem. Diffusion language models (dLLMs) make a different choice: start with a whole block of placeholder tokens and refine all of them in parallel over a few passes. In the last year this went from a research topic to a commercial API (Inception's Mercury family) and an open-weights release from Google (DiffusionGemma). This is a working note on how the idea works, what the papers show, and what is still only a vendor claim.

> **TL;DR** — Autoregressive decode reads every weight to emit *one* token, so it is **memory-bandwidth-bound**. A diffusion LM refines a whole block per pass, so each weight read produces many tokens and the bottleneck moves toward **compute**. That is why dLLMs are very fast at low concurrency. The costs: a quality gap to the autoregressive model they are derived from, awkward KV caching, and a throughput advantage that fades as concurrency grows: DiffusionGemma's own report has it ahead of the autoregressive baseline only up to about 32 concurrent users.

Throughout, I rely on papers and first-party technical material, and I label anything that is a vendor claim or my own back-of-envelope reasoning. The interactive figures are marked as *toy*, *illustrative* or *measured* so you can tell which is which.

## 1. The bottleneck diffusion is aimed at

If you have read [Serving LLMs: what actually decides your latency]({% post_url 2026-08-30-serving-llms-in-production %}), the setup is familiar. An autoregressive (AR) model generates a token, appends it, and runs the whole network again. Each step reads all (active) weights from GPU memory to produce **one** token, so the arithmetic units mostly wait on memory. Batching many users together hides this, but a single request still gets one token per forward pass, and the passes are strictly sequential.

Speculative decoding and multi-token prediction attack the same wall from inside the AR framework: guess several tokens, verify them in one pass. Diffusion attacks it from the other side, by changing how the text is produced in the first place. Figure 1 shows the difference on a toy sentence.

{% include diffusion/fig-race.html %}

## 2. How a text diffusion model works

Image diffusion starts from noise and denoises toward a picture. Text is discrete, so the dominant recipe for language is **masked diffusion**, studied in [D3PM](https://arxiv.org/abs/2107.03006)-style discrete diffusion and simplified in [MDLM (Sahoo et al., NeurIPS 2024)](https://arxiv.org/abs/2406.07524), which shows that a simple masked objective, reducing to a mixture of classical masked-language-modeling losses, approaches AR perplexity. Two stages define the method:

- **Forward (training) process.** Take a real sequence $x_0$. Pick a masking ratio $t \in (0, 1]$ at random and replace each token with a special `[MASK]` with probability $t$. At $t = 1$ everything is masked.
- **Reverse (generation) process.** A Transformer reads the partly masked sequence and predicts the original tokens at the masked positions. Generation starts from an all-masked block and repeatedly commits the positions it is most confident about, until nothing is masked.

[LLaDA](https://arxiv.org/abs/2502.09992) trains an 8B model from scratch this way, optimizing a likelihood bound that is a weighted cross-entropy on the masked positions only:

$$
\mathcal{L}(\theta) = -\,\mathbb{E}_{t,\,x_0,\,x_t}\left[\frac{1}{t}\sum_{i=1}^{L} \mathbf{1}[x_t^i = \texttt{[MASK]}]\;\log p_\theta\!\left(x_0^i \mid x_t\right)\right]
$$

The LLaDA authors report it competitive with their LLaMA3 8B baseline on in-context learning. Two properties fall out of this and drive everything that follows:

1. **Attention is bidirectional.** The model sees the whole block, left and right, while denoising. An AR model only sees the past.
2. **Many tokens can be committed per pass.** The sampler can commit several confident positions at once, so you get more than one token per forward pass. Tokens per forward pass (TPF) is the number to watch.

{% include diffusion/fig-mask.html %}

| | Autoregressive | Masked diffusion |
|---|---|---|
| **Order** | left to right, fixed | any order, chosen by confidence |
| **Tokens per forward pass** | 1 | many (TPF > 1) |
| **Attention** | causal | bidirectional within the block |
| **Can revise earlier output** | no | yes, in some designs |
| **Decode bottleneck** | memory bandwidth | shifts toward compute |

## 3. Why it is fast, and when it stops being fast

Take the AR decode loop and ask what a forward pass costs. The weights are read once either way. AR spends that read on one token; a diffusion model spends it on a whole block. The extra tokens are mostly extra arithmetic on data that is already on-chip, so the GPU stops idling on memory. Google's developer guide describes DiffusionGemma's speed-up in exactly these terms: the bottleneck shifts from memory bandwidth to compute.

That framing also tells you when the advantage should *disappear*. A diffusion pass processes the whole canvas but commits only TPF tokens, so it does roughly $256 / \mathrm{TPF}$ positions of work per committed token (my arithmetic from the report's numbers: about 13 for a 256-token canvas at TPF of about 20), where AR does 1. When the GPU is idle at batch size 1, that extra work is free. When a batch already saturates the GPU, it is not. Figure 4 is a deliberately simple model of this; play with the batch slider.

{% include diffusion/fig-bound.html %}

Two practical consequences:

- **Diffusion helps most at low concurrency**: interactive use, single-user, on-device, latency-sensitive agents.
- **At high concurrency the advantage fades.** The DiffusionGemma report's limitations section says it outperforms Gemma 4 with multi-token prediction on both per-user and total throughput up to about 32 concurrent users; beyond that, "the higher per-token compute cost causes AR models to gain a throughput advantage." These results are without targeted batch-size optimization, and the authors list analysis under realistic traffic as future work. Figure 4's toy model crosses over at a similar order of magnitude, but it is not calibrated to this result.

## 4. The cast, and what each one actually tells us

### 4.1 Block diffusion: the idea behind the practical models

Pure diffusion over a whole sequence has two practical problems: output length is fixed up front, and the KV cache is hard to use because every token can change at every step. [Block Diffusion (Arriola et al., ICLR 2025 oral)](https://arxiv.org/abs/2503.09573) interpolates between the two worlds: autoregressive *across* blocks, diffusion *within* each block. That gives KV caching, parallel token sampling and arbitrary-length output. Most of the practical systems below use some form of this idea.

### 4.2 DiffusionGemma (Google): the best-documented practical case

Released in June 2026 under Apache 2.0, [DiffusionGemma](https://developers.googleblog.com/diffusiongemma-the-developer-guide/) ships with weights and a [technical report](https://arxiv.org/html/2608.00146v1).

- **Converted from an AR model, not trained from scratch.** It is a fine-tune of the Gemma 4 26B-A4B mixture-of-experts model (about 3.8B active parameters), using less than 10% of the AR model's training tokens. [Dream 7B](https://arxiv.org/abs/2508.15487) uses the same trick at smaller scale, initializing a diffusion model from an AR LLM.
- **Block decoding.** It denoises a 256-token "canvas" in parallel; finished canvases are appended to the KV cache. Prefill is causal, denoising is bidirectional (Figure 3).
- **Few steps.** Up to 48 denoising steps, about 12 on average with adaptive stopping. A second training stage (reward maximization plus sampler distillation) is reported to raise tokens per forward pass from about 5 to nearly 20.

{% include diffusion/fig-arch.html %}

- **Speed (measured by the authors).** On one H100 (FP8, batch size 1, 4096 input and 1024 output tokens): about 1,479 tokens/s against 204 for AR Gemma 4, which the authors call 7.1x, and 4.8x over Gemma 4 with multi-token prediction. Google's blog uses the more conservative "up to 4x".

{% include diffusion/fig-speed.html %}

- **The price (measured by the authors).** In thinking mode it trails the AR model it was converted from:

| Benchmark | DiffusionGemma | Gemma 4 (AR) |
|---|---|---|
| AIME 2026 | 69.1 | 84.2 |
| GPQA Diamond | 73.2 | 79.8 |
| LiveCodeBench v6 | 69.1 | 71.4 |
| HumanEval | 94.5 | 98.2 |
| MMLU-Pro | 77.6 | 78.8 |
| IFEval | 97.4 | 97.2 |

The report states that the adaptation carries a performance penalty, and Google calls the model experimental. The shape is consistent: a large speed-up, a visible gap on the hardest reasoning tasks, and almost none on easier ones. Its limitations section also lists short, concise outputs and occasional token stuttering.

### 4.3 LLaDA2.x (Inclusion AI, Ant Group): open research at scale

The [LLaDA2.x](https://github.com/inclusionAI/LLaDA2.X) series is from Inclusion AI at Ant Group, released under Apache 2.0.

- **LLaDA2.0** ([paper: Scaling Up Diffusion Language Models to 100B](https://arxiv.org/abs/2512.15745), Nov 2025) ships 16B "mini" and 100B "flash" mixture-of-experts models, presented as the first diffusion LM scaled to the 100B level.
- **LLaDA2.1** ([paper: Speeding Up Text Diffusion via Token Editing](https://arxiv.org/abs/2602.08676), Feb 2026) adds **token editing**. Standard decoding is mask-to-token (M2T); LLaDA2.1 adds token-to-token (T2T) edits, so the model can revise tokens it already wrote. The paper describes a *speedy* and a *quality* mode and reports up to 892 tokens/s on HumanEval+ for the 100B model, plus the first large-scale reinforcement-learning framework for dLLMs. The paper evaluates the models on 33 benchmarks.

{% include diffusion/fig-edit.html %}

The editing idea is the most interesting conceptual step in this space: an AR model can never take back a token, a diffusion model in principle can.

### 4.4 Mercury (Inception Labs): the commercial family

Mercury is Inception's closed, API-only family of diffusion LLMs. The [Mercury paper (June 2025)](https://arxiv.org/abs/2506.17298) describes Transformer-based models trained to predict multiple tokens in parallel. For Mercury Coder it reports throughput on H100 GPUs of 1,109 tokens/s (Mini) and 737 tokens/s (Small), as measured by Artificial Analysis, and up to 10x faster on average than speed-optimized frontier models at comparable quality.

Mercury 3 launched on October 19, 2026. [Inception's announcement](https://www.inceptionlabs.ai/blog/introducing-mercury-3) claims about 1,068 tokens/s on NVIDIA GPUs and 4 to 8x the speed of GPT-6 Luna and Claude Haiku 5.5 at comparable quality. These are vendor claims: no benchmark scores or training details are published, and I could not find an independent measurement yet.

### 4.5 Beyond the big three: serving systems and hybrids

- **dInfer (Inclusion AI).** An open-source [inference framework for dLLMs](https://arxiv.org/abs/2510.08666) that splits the pipeline into a model, a diffusion iteration manager, a decoding strategy and a KV-cache manager. On LLaDA-MoE at batch size 1 it reports over 1,100 tokens/s on HumanEval (about 800 averaged over six benchmarks) on 8 H800 GPUs, about 10x faster than Fast-dLLM, and 2 to 3x faster than the autoregressive Qwen2.5-3B served with vLLM. This is the clearest evidence that the serving stack, not only the model, decides how fast a dLLM is.
- **Seed Diffusion.** A [large-scale discrete diffusion model for code](https://arxiv.org/abs/2508.02193) whose authors report 2,146 tokens/s on H20 GPUs while staying competitive on code benchmarks, and claim it is faster than Mercury and Gemini Diffusion. These are the authors' own measurements, on different hardware from the figures above, so they are not directly comparable.
- **Nemotron-Labs-Diffusion.** A [tri-mode model](https://arxiv.org/abs/2607.05722) (3B, 8B and 14B) trained to decode autoregressively, with diffusion, or in a self-speculation mode where diffusion drafts tokens and the AR path verifies them. The authors report about 6x the tokens per forward pass of Qwen3-8B at similar accuracy, and about 4x higher throughput on SPEED-Bench with SGLang on a GB200 GPU. The design choice is the interesting part: instead of picking a side, one model switches mode to suit the concurrency level, which fits the batch-size picture in Section 3.

## 5. What gets harder

- **KV caching.** AR caching works because past tokens never change. In a bidirectional denoiser, every token's representation can change at every step, so a naive cache is invalid. Block diffusion caches *finished* blocks. [Fast-dLLM](https://arxiv.org/abs/2505.22618) adds a training-free, block-wise approximate KV cache plus confidence-aware parallel decoding to LLaDA and Dream, reporting up to 27.6x throughput with minimal accuracy loss.
- **Quality vs steps.** Committing too many tokens in one pass breaks local coherence, because tokens decided simultaneously cannot condition on each other; Fast-dLLM attributes the quality loss to exactly this broken dependency, and fixes it by committing only tokens above a confidence threshold. The DiffusionGemma report mentions repetitive loops at low step counts before its second training stage. Fewer steps is faster and worse; the sampler is the dial.
- **Compute per committed token.** As Figure 4 shows, the work you save in memory traffic you partly spend in arithmetic. This is why DiffusionGemma's throughput advantage ends at moderate concurrency (about 32 users in its report), and it is the main open question for high-concurrency serving.
- **Variable length.** An AR model stops by emitting an end token. A fixed canvas has to decide its length up front or pad; block diffusion addresses this by chaining canvases.
- **Serving stacks.** Continuous batching, paged KV cache and scheduling were designed around one-token-per-step decode. DiffusionGemma's developer guide documents serving through vLLM, SGLang, MLX and Transformers, so support is arriving, but high-concurrency behavior is where I would look for surprises.
- **Post-training tooling.** RL, structured output and tool calling are mature for AR models. LLaDA2.1's RL framework shows dLLMs catching up, but expect rougher edges.

## 6. Where it fits

Diffusion looks strongest where the user feels latency directly and the task tolerates a small quality trade-off:

- **Interactive coding, completion and editing.** Infilling and next-edit suit a model that sees both sides of a position.
- **Voice and real-time agents.** Time to first token and short end-to-end latency dominate.
- **Agent loops with many sequential calls.** Per-call latency multiplies.
- **Small-batch or on-device inference**, where compute is spare.

AR models keep the edge when you need the highest reasoning accuracy, when you serve at high concurrency on saturated GPUs, or when you depend on the mature tooling around them.

## 7. A short checklist

- Is your bottleneck **per-request latency at low concurrency**? Diffusion is worth testing. Is it **aggregate throughput on a saturated GPU**? Expect the advantage to fade: DiffusionGemma's report puts the crossover with an MTP-accelerated AR model at about 32 concurrent users.
- Treat vendor speed numbers as ceilings. Ask for batch size, hardware, reasoning setting and sequence lengths.
- Expect a quality gap on the hardest reasoning tasks relative to the AR model of the same family, and measure it on *your* tasks.
- Prefer models with open weights or a technical report if you need to reason about behavior. DiffusionGemma and LLaDA2.x are inspectable; Mercury 3 is not.
- Watch the serving ecosystem and the caching literature. That is where the batch-size-above-1 question will get answered.

## Conclusion

Diffusion does not replace autoregressive decoding; it changes the trade. By committing a block of tokens per pass, it turns a memory-bound loop into a compute-bound one, which buys large speed-ups when the GPU would otherwise sit idle: low concurrency, single users, on-device, latency-sensitive agents. The open models back this up with numbers you can inspect. DiffusionGemma reaches about 7x the speed of its autoregressive parent at batch size 1, and pays for it with a quality gap on hard reasoning and a throughput advantage that ends at around 32 concurrent users.

Three things are still unsettled. Mercury 3's headline numbers are vendor claims with no independent measurement yet. High-concurrency serving is where the evidence is thinnest. And the most promising direction may be hybrids (block diffusion, token editing, self-speculation) that keep autoregressive strengths while borrowing parallelism. If you build latency-sensitive products, diffusion is worth a benchmark on your own workload today; if you serve saturated GPUs at scale, watch the serving research before betting on it.

## Further reading

Papers:

- [Simple and Effective Masked Diffusion Language Models (MDLM), NeurIPS 2024](https://arxiv.org/abs/2406.07524)
- [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models, ICLR 2025](https://arxiv.org/abs/2503.09573)
- [LLaDA: Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
- [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)
- [Mercury: Ultra-Fast Language Models Based on Diffusion (Inception Labs)](https://arxiv.org/abs/2506.17298)
- [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618)
- [LLaDA2.0: Scaling Up Diffusion Language Models to 100B](https://arxiv.org/abs/2512.15745)
- [LLaDA2.1: Speeding Up Text Diffusion via Token Editing](https://arxiv.org/abs/2602.08676)
- [DiffusionGemma Technical Report](https://arxiv.org/html/2608.00146v1)
- [A Survey on Diffusion Language Models](https://arxiv.org/abs/2508.10875)
- [dInfer: An Efficient Inference Framework for Diffusion Language Models](https://arxiv.org/abs/2510.08666)
- [Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference](https://arxiv.org/abs/2508.02193)
- [Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding](https://arxiv.org/abs/2607.05722)
- [Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)](https://arxiv.org/abs/2107.03006)

First-party material:

- [DiffusionGemma: the developer guide (Google)](https://developers.googleblog.com/diffusiongemma-the-developer-guide/)
- [Introducing Mercury 3 (Inception Labs)](https://www.inceptionlabs.ai/blog/introducing-mercury-3)
- [Inception Labs blog](https://www.inceptionlabs.ai/blog)
- [LLaDA2.X repository (Inclusion AI)](https://github.com/inclusionAI/LLaDA2.X)

## Cite this article

If this note was useful in your own writing, cite it as:

```bibtex
@misc{sharma2026diffusionllms,
  title        = {Diffusion LLMs: what changes when you stop generating one token at a time},
  author       = {Sharma, Rahul},
  year         = {2026},
  month        = {October},
  howpublished = {\url{https://rahulsharmavishwakarma.github.io/blog/2026/diffusion-llms/}},
  note         = {Online; accessed \today}
}
```

Or in APA style:

> Sharma, R. (2026, October 20). *Diffusion LLMs: what changes when you stop generating one token at a time*. Rahul Sharma. https://rahulsharmavishwakarma.github.io/blog/2026/diffusion-llms/
