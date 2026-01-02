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
@inproceedings{marco-etal-2025-small,
    title = "Small Language Models can Outperform Humans in Short Creative Writing: A Study Comparing {SLM}s with Humans and {LLM}s",
    author = "Marco, Guillermo  and
      Rello, Luz  and
      Gonzalo, Julio",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.437/",
    pages = "6552--6570",
    abstract = "In this paper, we evaluate the creative fiction writing abilities of a fine-tuned small language model (SLM), BART-large, and compare its performance to human writers and two large language models (LLMs): GPT-3.5 and GPT-4o. Our evaluation consists of two experiments: (i) a human study in which 68 participants rated short stories from humans and the SLM on grammaticality, relevance, creativity, and attractiveness, and (ii) a qualitative linguistic analysis examining the textual characteristics of stories produced by each model. In the first experiment, BART-large outscored average human writers overall (2.11 vs. 1.85), a 14{\%} relative improvement, though the slight human advantage in creativity was not statistically significant. In the second experiment, qualitative analysis showed that while GPT-4o demonstrated near-perfect coherence and used less cliche phrases, it tended to produce more predictable language, with only 3{\%} of its synopses featuring surprising associations (compared to 15{\%} for BART). These findings highlight how model size and fine-tuning influence the balance between creativity, fluency, and coherence in creative writing tasks, and demonstrate that smaller models can, in certain contexts, rival both humans and larger models."
}
```
