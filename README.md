<h3  align="center">⚔️ Adversarial Attacks against Closed-Source MLLMs via Feature Optimal Alignment</h3>
<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=jiaxiaojunQAQ.FOA-Attack" alt="访客统计" />
  <img src="https://img.shields.io/github/stars/jiaxiaojunQAQ/FOA-Attack?style=social" alt="GitHub stars" />
  <img alt="Static Badge" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
</p>

<p align="center">

> **FOA-Attack** is proposed to enhance adversarial transferability in multimodal large language models by optimizing both global and local feature alignments using cosine similarity and optimal transport.

## 💥 News

- **[2025-05-29]** We release the FOA-Attack code! 🚀

## 💻 Requirements

**Dependencies**: To install requirements:

```bash
pip install -r requirements.txt
```

## 🛰️ Quick Start

```bash
python generate_adversarial_samples_foa_attack.py
python blackbox_text_generation.py -m blackbox.model_name=gpt4o,claude,gemini
python gpt_evaluate.py -m blackbox.model_name=gpt4o,claude,gemini
python keyword_matching_gpt.py -m blackbox.model_name=gpt4o,claude,gemini
```

## 💖 Acknowledgements
This project is built on [M-Attack](https://github.com/VILA-Lab/M-Attack). We sincerely thank them for their outstanding work.
