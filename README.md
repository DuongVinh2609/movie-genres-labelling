
# 🎬 Movie Genre Labelling Based on Poster and Title

This project explores a multimodal deep learning approach for automatically classifying movie genres based on their **poster images** and **titles**. Given the subjective and multi-label nature of genre classification, our method integrates both **visual and textual features** to enhance classification accuracy.

## 📌 Overview

- 🎯 **Task**: Multi-label genre classification for movies
- 🧾 **Inputs**: Movie posters (images) and titles (text)
- 🧠 **Model**: A custom ResNet-inspired CNN for image processing and a TF-IDF-based fully connected network for titles
- 🗃️ **Dataset**: 3105 labeled movie samples from IMDB
- 📈 **Metrics**: Precision, Recall, and F1-Score

---

## 📚 Motivation

According to Statista, over **800+ movies** are released annually in North America alone. Manually labeling genres is time-consuming, subjective, and increasingly impractical at scale. This project proposes a deep learning solution to **automate** the genre labelling process using only a film’s **title and poster**, which are often available at early stages of movie production.

---

## 📦 Dataset

The dataset includes:

- **3105 movies** (1950s–2000s)
- **18 genre labels** (multi-label, e.g., Action, Drama, Romance)
- Each movie contains:
  - `title`: e.g., _"The Godfather (1972)"_
  - `poster`: RGB image (resized to 256×256)
  - `genre`: one or more genre labels

📥 [Download Dataset (Google Drive)](https://drive.usercontent.google.com/download?id=1hUqu1mbFeTEfBvl-7fc56fHFfCSzIktD)

---

## 🔍 Preprocessing

- **Titles**: Lowercased, punctuation and year removed, then encoded via **TF-IDF** (vocab size = 8192)
- **Posters**: Resized to 256×256 pixels, converted to tensors
- **Labels**: One-hot encoded genre vectors

---

## 🧠 Model Architecture

We propose a **dual-branch neural architecture**:

### 📝 Title Sub-model
- TF-IDF vector (8192-dim) input
- Two fully connected layers with ReLU, BatchNorm, Dropout
- Outputs genre probabilities

### 🖼️ Poster Sub-model
- Custom CNN inspired by **ResNet**
- Includes skip connections, BatchNorm, Dropout
- Outputs genre probabilities

### 🔗 Fusion
- Outputs from both sub-models are **summed**
- Final prediction is made from the combined logits

---

## 🏋️ Training

- **Epochs**: 50
- **Batch size**: 64
- **Loss function**: *CrossEntropyLoss* (can be improved using `BCEWithLogitsLoss` for multi-label)
- **Optimizer**: Adam (lr=1e-3, weight decay=1e-4)
- **Hardware**: Trained on NVIDIA T4 GPU (16GB), ~30 minutes

---

## 📊 Evaluation

Evaluated on a hold-out test set using:

| Metric    | Description |
|-----------|-------------|
| **Precision** | % of predicted genres that were correct |
| **Recall**    | % of actual genres that were correctly predicted |
| **F1-score**  | Harmonic mean of Precision and Recall |

Results indicate strong performance on common genres. Edge cases (nuanced or rare genres) remain challenging due to limited data size.

---

## 🧪 Real-world Inference

The model can be tested on real movie titles and poster images using the provided `inference` section.

Example:
```python
input_title = "Avengers: Endgame (2019)"
input_image_path = "./avengerEndgame.jpg"
```

Predicted genres:
```
Action: 0.92
Adventure: 0.81
Sci-Fi: 0.64
```

---

## Resources

- 📁 [Dataset Download (zip)](https://drive.usercontent.google.com/download?id=1hUqu1mbFeTEfBvl-7fc56fHFfCSzIktD)

---

## 🧠 Authors
 
- Dương Nguyễn Gia Vinh

Vietnam National University, Hanoi – University of Engineering and Technology  

---

## 📚 References

> [1] He et al., “Deep Residual Learning for Image Recognition”, CVPR 2016  
> [2] Spärck Jones, “A Statistical Interpretation of Term Specificity…”, Journal of Documentation, 1972  
> [3] Kingma & Ba, “Adam: A Method for Stochastic Optimization”, 2014  
> [4] Srivastava et al., “Dropout: A Simple Way to Prevent Neural Networks from Overfitting”, 2014  
> [5] Ioffe & Szegedy, “Batch Normalization”, 2015  
> [6] Chu & Guo, “Movie Genre Classification using Poster Images”, MUSA2 2017  
> [7] Narawade et al., “Movie Poster Classification into Multiple Genres”, ICACC 2021  
> [8] Sung & Chokshi, “Movie Genre Classification”, Stanford CS230, 2020

---

## 📌 Future Work

- Augment training data with subtitle/synopsis embeddings
- Incorporate pretrained vision transformers (ViT, CLIP)
- Explore prompt-based genre reasoning using multimodal LLMs (e.g., Flamingo, LLaVA)
