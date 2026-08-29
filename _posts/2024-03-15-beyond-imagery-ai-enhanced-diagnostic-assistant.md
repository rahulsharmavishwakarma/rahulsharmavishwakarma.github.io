---
layout: post
title: "Beyond Imagery: AI-Enhanced Diagnostic Assistant for Cancer and Tumor Diagnosis using Radiology Imaging"
date: 2024-03-15 09:00:00 +0530
description: "Our IJETS 2024 paper — combining LLaVA-Med with Retrieval Augmented Generation to build a clinically grounded visual question answering assistant for CT and MRI imaging."
tags: [vision-language, medical-imaging, computer-vision, rag]
categories: [whitepapers]
featured: false
---

> This post is a technical walkthrough of our IJETS paper, [*Beyond Imagery: AI-Enhanced Diagnostic Assistant for Cancer and Tumor Diagnosis using Radiology Imaging*](https://ijets.in/Downloads/Published/E0202403007.pdf) (NandhaGopal S M, Rahul Sharma, Nithin M, Prajwal B R, Prashanth Kalgonda — International Journal On Engineering Technology and Sciences, Vol. XI, Issue III, March 2024). It was my final-year undergraduate research project at HKBK College of Engineering.

A radiologist looking at a CT scan is not just pattern-matching pixels against a list of diseases. She is interrogating the image — *Is there a mass in the upper lobe? Is it enhancing? Has it grown since the last scan?* — and answering by fusing what she sees with years of accumulated medical knowledge.

Most computer vision systems in medicine do none of that. They classify: this image is "malignant" or "benign", with a confidence score. Useful, but rigid. The clinician cannot ask a follow-up question, and the system cannot explain itself by pointing at the literature it "learned" from.

Our paper asks a simple question: **what if the diagnostic assistant could hold a conversation about the image — and ground its answers in biomedical literature at the same time?** The answer we landed on is a multimodal generator (LLaVA-Med) wired to a retrieval layer over biomedical literature. Below is the full architecture, how each part works, what it scores, and where it breaks.

## 1. The system at a glance

<figure class="paper-figure">
<svg viewBox="0 0 880 480" role="img" aria-label="Architecture of the RAG-augmented medical VQA system" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8a8a8f"/>
    </marker>
    <marker id="arrAccent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8a8a8f"/>
    </marker>
  </defs>

  <!-- ===== main inference path ===== -->
  <!-- image input -->
  <rect x="20" y="42" width="160" height="58" rx="6" fill="none" stroke="var(--global-text-color)" stroke-width="1.3"/>
  <text x="100" y="66" text-anchor="middle" font-size="13.5" fill="var(--global-text-color)">Radiology image</text>
  <text x="100" y="86" text-anchor="middle" font-size="11.5" fill="var(--global-text-color-light)">CT · MRI · X-ray</text>

  <!-- vision encoder -->
  <rect x="250" y="42" width="180" height="58" rx="6" fill="none" stroke="var(--global-text-color)" stroke-width="1.3"/>
  <text x="340" y="66" text-anchor="middle" font-size="13.5" fill="var(--global-text-color)">Vision encoder</text>
  <text x="340" y="86" text-anchor="middle" font-size="11.5" fill="var(--global-text-color-light)">CLIP ViT-L/14 (frozen)</text>

  <!-- adapter -->
  <rect x="500" y="42" width="170" height="58" rx="6" fill="var(--global-theme-color)" fill-opacity="0.08" stroke="var(--global-theme-color)" stroke-width="1.6"/>
  <text x="585" y="66" text-anchor="middle" font-size="13.5" fill="var(--global-text-color)">MLP adapter g&#8345;</text>
  <text x="585" y="86" text-anchor="middle" font-size="11.5" fill="var(--global-text-color-light)">projector — the only trained part</text>

  <!-- LLM -->
  <rect x="740" y="42" width="120" height="58" rx="6" fill="none" stroke="var(--global-text-color)" stroke-width="1.3"/>
  <text x="800" y="66" text-anchor="middle" font-size="13.5" fill="var(--global-text-color)">LLM decoder</text>
  <text x="800" y="86" text-anchor="middle" font-size="11.5" fill="var(--global-text-color-light)">Vicuna (frozen)</text>

  <!-- question path -->
  <rect x="20" y="160" width="160" height="58" rx="6" fill="none" stroke="var(--global-text-color)" stroke-width="1.3"/>
  <text x="100" y="184" text-anchor="middle" font-size="13.5" fill="var(--global-text-color)">Clinical question</text>
  <text x="100" y="204" text-anchor="middle" font-size="11.5" fill="var(--global-text-color-light)">natural language</text>

  <rect x="250" y="160" width="180" height="58" rx="6" fill="none" stroke="var(--global-text-color)" stroke-width="1.3"/>
  <text x="340" y="184" text-anchor="middle" font-size="13.5" fill="var(--global-text-color)">Question encoder</text>
  <text x="340" y="204" text-anchor="middle" font-size="11.5" fill="var(--global-text-color-light)">tokenizer + transformer</text>

  <!-- answer -->
  <rect x="740" y="160" width="120" height="58" rx="6" fill="var(--global-theme-color)" fill-opacity="0.08" stroke="var(--global-theme-color)" stroke-width="1.6"/>
  <text x="800" y="184" text-anchor="middle" font-size="13.5" fill="var(--global-text-color)">Answer</text>
  <text x="800" y="204" text-anchor="middle" font-size="11.5" fill="var(--global-text-color-light)">generated text</text>

  <!-- main path arrows -->
  <line x1="180" y1="71" x2="244" y2="71" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#arr)"/>
  <line x1="430" y1="71" x2="494" y2="71" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#arr)"/>
  <line x1="670" y1="71" x2="734" y2="71" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#arr)"/>
  <line x1="180" y1="189" x2="244" y2="189" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#arr)"/>
  <!-- question tokens into LLM -->
  <path d="M 430 189 L 585 189 L 585 130 L 734 130 L 734 100" fill="none" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#arr)"/>

  <!-- ===== RAG block ===== -->
  <rect x="20" y="280" width="840" height="180" rx="8" fill="none" stroke="var(--global-text-color-light)" stroke-width="1.1" stroke-dasharray="6 4"/>
  <text x="40" y="306" font-size="12.5" fill="var(--global-text-color-light)" letter-spacing="1">RETRIEVAL AUGMENTED GENERATION</text>

  <!-- corpus -->
  <rect x="45" y="330" width="230" height="70" rx="6" fill="none" stroke="var(--global-text-color)" stroke-width="1.3"/>
  <text x="160" y="357" text-anchor="middle" font-size="13.5" fill="var(--global-text-color)">Biomedical corpus 𝒞</text>
  <text x="160" y="377" text-anchor="middle" font-size="11.5" fill="var(--global-text-color-light)">literature · case studies (PMC-15M)</text>

  <!-- retriever -->
  <rect x="345" y="330" width="220" height="70" rx="6" fill="none" stroke="var(--global-text-color)" stroke-width="1.3"/>
  <text x="455" y="357" text-anchor="middle" font-size="13.5" fill="var(--global-text-color)">Retriever</text>
  <text x="455" y="377" text-anchor="middle" font-size="11.5" fill="var(--global-text-color-light)">top-k by cosine similarity</text>

  <!-- prompt builder -->
  <rect x="635" y="330" width="200" height="70" rx="6" fill="none" stroke="var(--global-text-color)" stroke-width="1.3"/>
  <text x="735" y="357" text-anchor="middle" font-size="13.5" fill="var(--global-text-color)">Prompt builder</text>
  <text x="735" y="377" text-anchor="middle" font-size="11.5" fill="var(--global-text-color-light)">context assembly</text>

  <!-- RAG arrows -->
  <line x1="275" y1="365" x2="339" y2="365" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#arr)"/>
  <line x1="565" y1="365" x2="629" y2="365" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#arr)"/>
  <!-- question embedding query to retriever -->
  <path d="M 340 218 L 340 250 L 455 250 L 455 324" fill="none" stroke="#8a8a8f" stroke-width="1.1" stroke-dasharray="5 4" marker-end="url(#arr)"/>
  <text x="468" y="262" font-size="11" fill="var(--global-text-color-light)">query embedding e&#8339;</text>
  <!-- retrieved passages up to LLM -->
  <path d="M 735 330 L 735 106" fill="none" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#arr)"/>
  <text x="748" y="240" font-size="11" fill="var(--global-text-color-light)">top-k passages 𝒟&#8339;</text>

  <!-- legend -->
  <line x1="40" y1="478" x2="70" y2="478" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#arr)"/>
  <text x="78" y="482" font-size="11" fill="var(--global-text-color-light)">inference path</text>
  <line x1="200" y1="478" x2="230" y2="478" stroke="#8a8a8f" stroke-width="1.1" stroke-dasharray="5 4" marker-end="url(#arr)"/>
  <text x="238" y="482" font-size="11" fill="var(--global-text-color-light)">retrieval query path</text>
</svg>
<figcaption><strong>Fig. 1</strong> — Architecture of the proposed RAG-augmented medical VQA system. Frozen pretrained components (CLIP ViT-L/14 vision tower, Vicuna language model) are connected by a lightweight trainable MLP adapter; a retrieval layer over biomedical literature conditions generation on external evidence at answer time.</figcaption>
</figure>

The design philosophy is the same one that made parameter-efficient fine-tuning take off: **keep the expensive pretrained machinery frozen, and train only the glue.** The vision tower and the LLM already know how to see and speak — the adapter's whole job is to translate image features into tokens the LLM is willing to attend to.

## 2. The multimodal core, piece by piece

### 2.1 Vision encoder — turning pixels into patch tokens

Medical images are not ImageNet photos: grayscale, high resolution, pathology in fine texture. The encoder is a pretrained vision transformer (CLIP ViT-L/14 in the LLaVA-Med lineage; ResNet-18-style CNNs are the classic baseline). It splits the scan into patches and emits one feature vector per patch:

$$\mathbf{Z}_v = f_v(I) \in \mathbb{R}^{N \times d_v}$$

where $N$ is the number of patches and $d_v$ the vision width. The encoder is trained/frozen against medical imaging sets such as VQA-RAD and PathVQA, optimizing cross-entropy — so the features it surfaces are the ones that separate pathologies, not natural-image categories.

### 2.2 Question encoder — parsing the interrogation

The question "Is there pleural effusion, and on which side?" is tokenized and passed through a transformer encoder (RNN/LSTM in older designs). Self-attention resolves what the question is actually about — the organ (pleura), the finding (effusion), the demanded output (a side). This produces the query representation used twice downstream: fused with the image for answering, and as the retrieval query for RAG.

### 2.3 Multimodal fusion — the MLP adapter

This is the heart of the parameter-efficient design. Rather than co-training a giant joint model, a small two-layer MLP projects vision tokens into the LLM's embedding space:

$$\mathbf{H}_v = \big(\mathrm{GELU}(\mathbf{Z}_v W_1 + \mathbf{b}_1)\big) W_2 + \mathbf{b}_2$$

with the hidden layer deliberately *smaller* than the input embeddings — an information bottleneck that forces the projection to keep only what matters for answering. The first layer compresses each patch token from the vision dimension $d_v$ down to the bottleneck width; GELU is the nonlinearity that lets the mapping bend around the geometry of the language embedding space rather than staying a flat linear rotation; the second layer expands back out to the LLM's input dimension $d_l$. After training, an LLM that has never seen an image attends to these projected tokens as if they were ordinary word embeddings — that is the whole trick.

Why an adapter and not fine-tuning the encoders themselves? Three reasons, all practical:

- **Parameter budget.** The adapter holds $O(d_v \cdot h + h \cdot d_l)$ weights — a few million — against tens of billions frozen across the towers. Training signal is concentrated where the modality gap actually lives.
- **Catastrophic forgetting.** The frozen LLM retains everything it knows about medical language and reasoning; fine-tuning it end-to-end on a few thousand QA pairs would overwrite exactly the knowledge we are borrowing.
- **Feature selection on top.** LLaVA-Med additionally ranks fused image/question features by cosine similarity to the query vector and keeps only the most relevant — a cheap attention-like filter that stops noisy patch tokens from diluting the question.

### 2.4 Answer decoder — generation under evidence

The decoder is a frozen transformer LLM (Vicuna in LLaVA-Med; the paper also discusses LLaMA-2 and GPT-3.5-class models). At each step it predicts:

$$P(y_t \mid y_{<t},\; q,\; \mathbf{H}_v,\; \mathcal{D}_q)$$

Note the $\mathcal{D}_q$ — the retrieved passages. That conditioning term is the entire difference between a chatbot that *remembers* medicine and one that *looks it up*. Training minimizes cross-entropy against ground-truth answers; because only the adapter receives gradients, the LLM's clinical language ability (and its failures) stay untouched.

## 3. The RAG layer — retrieval before generation

The retrieval pipeline runs in four steps:

**1. Index.** Biomedical literature and case studies (the PMC-15M corpus that LLaVA-Med itself was curriculum-trained on, plus curated medical databases) are chunked and embedded once, offline.

**2. Query.** For each incoming question, the question encoder's representation $e_q$ is matched against document embeddings by cosine similarity:

$$\mathcal{D}_q = \operatorname{TopK}_{d \in \mathcal{C}} \; \frac{e_q \cdot e_d}{\lVert e_q \rVert \, \lVert e_d \rVert}$$

**3. Assemble.** The prompt builder stitches together: image tokens from the adapter, the question, and the top-$k$ retrieved passages — in that order, so attention flows from evidence to question.

**4. Generate.** The LLM answers *conditioned on the passages*, which is what makes responses citable: claims in the answer trace back to retrievable sources rather than to compressed weights.

This matters double in oncology. First, for **credibility** — a generated claim a clinician can verify against its source is a different artifact than one she has to take on faith. Second, for **freshness** — treatment guidelines move faster than model retraining cycles, and retrieval absorbs that drift without touching a single weight.

## 4. Evaluation setup

The paper evaluates on the three standard biomedical VQA benchmarks:

| Dataset | Images | QA pairs | Character |
|---|---:|---:|---|
| **VQA-RAD** | 315 | 3,515 | Clinician-authored; head/chest/abdomen; 11 question categories; ~50% closed-ended |
| **SLAKE** | 642 | 7,000+ | Physician-annotated, knowledge-enhanced; segmentation masks + bounding boxes; bilingual (English subset used) |
| **PathVQA** | 4,998 | 32,799 | Pathology; location/shape/color/appearance; open- and closed-ended |

**Metrics.** Accuracy for closed-set questions; recall (fraction of ground-truth tokens present in the generated answer) for open-set questions.

**Base model provenance.** LLaVA-Med is curriculum-trained over the [PMC-15M](https://doi.org/10.48550/arXiv.2306.00890) figure–caption corpus and fine-tuned on studies from MIMIC-CXR, reaching BLEU-4 of 0.264 and precision of 0.311 on its report-generation objective — the numbers that establish the floor our VQA task builds on.

## 5. Results — and the honest part

Three findings from the paper's comparison against supervised and generative baselines:

1. **LLaVA-Med variants beat vanilla LLaVA** across the board, with margins shifting based on language-model and vision-encoder initialization. Biomedical curriculum training transfers.
2. **Closed-set questions: state of the art on VQA-RAD and PathVQA.** When the answer set is constrained, instruction-tuned multimodal models outperform fully supervised pipelines — they follow the task definition better.
3. **Open-set questions are where it gets uncomfortable.** The model achieves SoTA on SLAKE but lags elsewhere. Unconstrained biomedical questions are ambiguous; a generative decoder without answer options can produce fluent, *plausible-sounding* answers that are simply not in the image.

Point 3 is the most useful result in the paper. Closed evaluation hides failure modes — with five options, a lucky guess and grounded reasoning score identically. Open generation exposes the gap, and it is precisely the gap retrieval is meant to close: give the decoder something concrete to condition on, and fluency stops being a liability.

## 6. Where this goes next

The paper's limitation is its roadmap. Open-set performance is the real test of clinical usefulness, and the fix is not a bigger language model — it is better grounding: stronger retrieval (hybrid dense + keyword, reranking), domain-specific knowledge integration, and evaluation that rewards *correct-and-supported* answers over merely plausible ones. The first step of that direction is architectural: train only a lightweight projector on a frozen backbone, and let retrieval carry the domain knowledge.

## Reference

> NandhaGopal S M, **Rahul Sharma**, Nithin M, Prajwal B R, Prashanth Kalgonda, "Beyond Imagery: AI-Enhanced Diagnostic Assistant for Cancer and Tumor Diagnosis using Radiology Imaging," *International Journal On Engineering Technology and Sciences (IJETS)*, Vol. XI, Issue III, March 2024, pp. 41–46. ISSN (P) 2349-3968, ISSN (O) 2349-3976. Available: [PDF ↗](https://ijets.in/Downloads/Published/E0202403007.pdf)
