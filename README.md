# slm-creativity

This repository contains the data, models, and analysis code used in the paper:

**Small Language Models can Outperform Humans in Short Creative Writing:  
A Study Comparing SLMs with Humans and LLMs**

The work explores whether **small, fine-tuned language models (SLMs)** can be competitive with both **human writers** and **large language models (LLMs)** in short creative writing tasks.

## Overview

We study a creative writing task consisting of **generating short movie synopses from a given title**.  
The main model under analysis is a **fine-tuned BART-large**, which is compared against:

- Human-written synopses (average, non-professional writers)
- GPT-3.5 (zero-shot)
- GPT-4o (zero-shot)

The evaluation combines:

1. **A large-scale human study**  
   - 68 participants  
   - More than 24,000 manual ratings  
   - Evaluation dimensions: readability, understandability, relevance, informativity, attractiveness, and creativity  
   - Several experimental settings to measure the effect of **authorial bias** (human vs. AI)

2. **A qualitative linguistic analysis**  
   - Focused on coherence, clichés, recurrent themes, and surprising associations  
   - Designed to understand *why* models differ beyond aggregate scores

The results show that the SLM:
- Outperforms average human writers in most quality dimensions
- Is slightly behind humans in perceived creativity (not statistically significant)
- Produces more surprising associations than larger models, which tend to be more fluent but more predictable

## Repository contents

This repository includes:
- Annotated datasets used in the human evaluation
- Model outputs (SLM, GPT-3.5, GPT-4o)
- Scripts for statistical analysis and qualitative annotation
- Figures and intermediate results used in the paper

All materials are released to support **reproducibility and further research** on creativity, evaluation, and model size.

## Citation

If you use this repository, please cite:

Marco, G., Rello, L., & Gonzalo, J. (2025).  
*Small Language Models can Outperform Humans in Short Creative Writing:  
A Study Comparing SLMs with Humans and LLMs*.  
Proceedings of the 31st International Conference on Computational Linguistics (COLING).

```bibtex
@inproceedings{marco2025slmcreativity,
  title     = {Small Language Models can Outperform Humans in Short Creative Writing: A Study Comparing SLMs with Humans and LLMs},
  author    = {Marco, Guillermo and Rello, Luz and Gonzalo, Julio},
  booktitle = {Proceedings of the 31st International Conference on Computational Linguistics},
  pages     = {6552--6570},
  year      = {2025},
  publisher = {Association for Computational Linguistics}
}
```
