---
layout: page
title: Medi-care
description: Visual Question Answering for Medical Imaging.
img: assets/img/publication_preview/vqa.png
importance: 1
category: work
related_publications:
---
Medi-Care utilizes LLaVA-Med base weights for precise visual question answering in medical images, alongside the LLaMA model for accurate answer generation. This combination enables efficient and accurate diagnostic support, enhancing healthcare outcomes.
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/publication_preview/vqa.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A VQA model used in medical imaging typically comprises an image encoder to process visual input into feature vectors, often using CNNs or pre-trained models. A question encoder handles textual input and converts questions into feature representations, commonly using RNNs or transformers. A multimodal fusion component integrates the image and question embeddings, using techniques like concatenation, element-wise multiplication, or attention mechanisms. The fused representation is then processed by a classifier or sequence-to-sequence model to generate answers to the questions.
</div>

More information and code on our [repository](https://github.com/rahulsharmavishwakarma/medi-care).