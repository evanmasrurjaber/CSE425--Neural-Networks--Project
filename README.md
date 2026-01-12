# Multimodal Music Clustering using Variational Autoencoders
#[Project Report](https://drive.google.com/file/d/1ygnVgW69BIYJYl_ilaGBRQmV3VqecKwJ/view?usp=drive_link)

An unsupervised learning research project exploring the use of Variational Autoencoders (VAEs) to cluster hybrid-language music tracks (English and Bangla). This project addresses the challenges of high-dimensional audio data and the "semantic gap" in cross-cultural music information retrieval.

## 📌 Project Overview

This project implements a deep learning pipeline to extract latent representations from heterogeneous data sources (Audio and Lyrics) for density-based clustering. The primary goal was to distinguish between musical styles and languages without using explicit labels, comparing deep generative models against traditional linear baselines.

### Key Objectives
* **Hybrid Clustering:** Grouping music tracks from a mixed dataset of Western (English) and Regional (Bangla) songs.
* **Multimodal Fusion:** Integrating audio Mel-Spectrograms with TF-IDF lyric embeddings to capture both acoustic "vibe" and semantic content.
* **Disentanglement:** Utilizing Beta-VAE architectures to learn factorized latent representations of genre and language.

## 📊 Dataset

The project utilizes a custom hybrid dataset comprising **2,617 tracks**:
* **GTZAN Genre Collection:** 1,000 tracks representing 10 Western genres (Rock, Jazz, Pop, etc.).
* **BanglaBeats:** 1,617 tracks of Bengali music to introduce linguistic diversity.
* **MERGE Dataset:** Used for the trimodal experiments (Audio + Lyrics + Genre).

## 🧠 Methodology & Architectures

The project explores three distinct levels of architectural complexity:



### 1. Basic VAE (Unimodal)
* **Input:** 57-dimensional MFCC (Mel-Frequency Cepstral Coefficients) vectors.
* **Architecture:** Dense MLP Encoder/Decoder.
* **Goal:** Baseline feature extraction on low-dimensional tabular data.

### 2. Convolutional VAE (Multimodal)
* **Input:** $128 \times 128$ Mel-Spectrograms + TF-IDF Lyric Embeddings.
* **Architecture:** 4-layer Conv2D Encoder capturing hierarchical time-frequency dependencies.
* **Clustering:** Density-based clustering (DBSCAN) on the learned latent manifold.


### 3. Beta-VAE (Disentangled)
* **Objective:** Optimized with a weighted KL-divergence term ($\beta=4.0$) to enforce independence between latent factors.
* **Result:** Produces "blurrier" reconstructions but structurally robust latent spaces that generalize better across genres.

## 📉 Results & Analysis

The deep generative approach significantly outperformed linear methods in handling complex, high-dimensional spectral data.

| Model | Clustering Method | Silhouette Score | Key Finding |
| :--- | :--- | :--- | :--- |
| **ConvVAE** | **DBSCAN** | **0.1617** | Best separation for complex spectrogram data. |
| **Basic VAE** | K-Means | 0.3351 | Effective only for simple MFCC features. |
| **PCA** | DBSCAN | -0.046 | Failed completely on high-dim spectral data. |

**Key Insights:**
* **Latent Space Topology:** t-SNE visualizations revealed that the ConvVAE successfully pulled English and Bangla tracks into distinct clusters, whereas PCA resulted in a mashed, entangled manifold.

* **Semantic Alignment:** Heatmap analysis showed high cluster purity, with specific clusters aligning strongly with language labels despite the model never seeing those labels during training.
* **Generative Trade-off:** The Beta-VAE demonstrated a trade-off where higher disentanglement (better structure) resulted in slightly lower clustering density compared to the standard ConvVAE.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch / TensorFlow (Implied by layer definitions)
* **Audio Processing:** Librosa (MFCC/Spectrogram extraction)
* **Clustering:** Scikit-learn (K-Means, DBSCAN, Agglomerative)
* **Visualization:** Matplotlib, t-SNE
