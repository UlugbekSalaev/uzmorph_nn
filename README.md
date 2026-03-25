# UzMorphNN: Uzbek Morphological Analyzer

UzMorphNN is a sequence-labeling morphological analyzer designed to process the highly agglutinative structure of the Uzbek language. Leveraging character-level neural network architectures trained on a massive 80K+ word standard corpus, the system achieves state-of-the-art stemming accuracy (>95%) while precisely extracting up to 16 grammatical features per token.

## 🚀 Available Deployments
This repository hosts the source inference code for the models. You can also access these pre-trained models via:

- **PyPI (Python Package Index):** Install the models directly into your environment using `pip`.
  - **BiLSTM (Baseline):** `pip install uzmorph-nn` 👉 [View on PyPI](https://pypi.org/project/uzmorph-nn/)
  - **BiGRU (Optimal):** `pip install uzmorph-bigru` 👉 [View on PyPI](https://pypi.org/project/uzmorph-bigru/)
  - **Transformer (Long Sequence):** `pip install uzmorph-transformer` 👉 [View on PyPI](https://pypi.org/project/uzmorph-transformer/)

- **Hugging Face Spaces:** Try the models directly via our interactive web interface!
  - [UzMorphNN Web Demo](https://huggingface.co/spaces/ulugbeksalaev/uzmorph_nn)

## 🧠 Core Architectures
We provide three natively trained neural architectures, optimized for different use cases. All models operate using the BIO (Beginning-Inside-Outside) character-level sequence labeling scheme:
1. **BiGRU (`uzmorph_bigru`)**: The recommended model for deployment. Achieves the highest stemming precision (95.00%) with minimal inference latency.
2. **BiLSTM (`uzmorph_bilstm`)**: An exceptionally strong recurrent baseline with 94.85% accuracy.
3. **Transformer (`uzmorph_transformer`)**: A parallel-attentional model scoring 94.76%, specifically excelling at extremely long multi-suffix word structures.

## 📦 Local GitHub Installation & Usage

Clone this repository and import the desired module class directly into your project.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/uzmorph_nn.git
cd uzmorph_nn
```

### 💻 Fast Python Inference Script:

```python
import sys
import os

# Import the favored model (e.g., BiGRU)
from uzmorph_bigru.uzmorph_bigru import uzmorph_bigru

# 1. Initialize the Analyzer (paths point to the weights inside the data folder)
analyzer = uzmorph_bigru(
    model_path="uzmorph_bigru/data/morph_bigru.pth",
    char_vocab="uzmorph_bigru/data/char_vocab.json",
    tag_vocab="uzmorph_bigru/data/tag_vocab.json"
)

# 2. Analyze a complex Uzbek word
word = "kitoblarimizdagilardanmi"
result = analyzer.analyze(word)

print(f"Word: {word}")
print(f"Stem: {result.stem}")
print(f"POS:  {result.pos}")
print(f"Grammatical Features: {result.features}")
```

## 🧩 The 16-Dimensional Feature Space
The model does not just segment morphology; it completely reconstructs the linguistic profile based on the **Complete Set of Endings (CSE)** methodology. Extracted features map up to 16 grammatical attributes:

1. `POS` (Part of Speech)
2. `Tense` (Past, Present, Future)
3. `Person` (1st, 2nd, 3rd)
4. `Possession`
5. `Cases` (Nominative, Genitive, Accusative, Dative, Locative, Ablative)
6. `Verb Voice` (Active, Passive, Reflexive, Causative, Reciprocal)
7. `Verb Function`
8. `Impulsion`
9. `Degree` (Positive, Comparative, Superlative)
10. `Number` (Singular/Plural)
11. `Question`
12. `Negative`
13. `Copula`
14. `Honorifics` (Polite forms)
15. `Lexical suffix markers`
16. `Syntactical suffix markers`

## 📄 Scientific Validation
The implementations inside this repository map directly back to verified experimental benchmarks proving deep sequence networks' capability to resolve complex Uzbek morphology beyond typical heuristic rule-based systems.
