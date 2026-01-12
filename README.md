# Multimodal Music Clustering using Variational Autoencoders

An unsupervised learning research project exploring the use of Variational Autoencoders (VAEs) to cluster hybrid-language music tracks (English and Bangla). [cite_start]This project addresses the challenges of high-dimensional audio data and the "semantic gap" in cross-cultural music information retrieval[cite: 151, 155, 169].

## 📌 Project Overview

This project implements a deep learning pipeline to extract latent representations from heterogeneous data sources (Audio and Lyrics) for density-based clustering. [cite_start]The primary goal was to distinguish between musical styles and languages without using explicit labels, comparing deep generative models against traditional linear baselines [cite: 156-159].

### Key Objectives
* [cite_start]**Hybrid Clustering:** Grouping music tracks from a mixed dataset of Western (English) and Regional (Bangla) songs[cite: 166].
* [cite_start]**Multimodal Fusion:** Integrating audio Mel-Spectrograms with TF-IDF lyric embeddings to capture both acoustic "vibe" and semantic content [cite: 157, 183-187].
* [cite_start]**Disentanglement:** Utilizing Beta-VAE architectures to learn factorized latent representations of genre and language[cite: 196, 204].

## 📊 Dataset

[cite_start]The project utilizes a custom hybrid dataset comprising **2,617 tracks**[cite: 268]:
* [cite_start]**GTZAN Genre Collection:** 1,000 tracks representing 10 Western genres (Rock, Jazz, Pop, etc.)[cite: 264].
* [cite_start]**BanglaBeats:** 1,617 tracks of Bengali music to introduce linguistic diversity[cite: 267].
* [cite_start]**MERGE Dataset:** Used for the trimodal experiments (Audio + Lyrics + Genre)[cite: 286].

## 🧠 Methodology & Architectures

The project explores three distinct levels of architectural complexity:

### 1. Basic VAE (Unimodal)
* [cite_start]**Input:** 57-dimensional MFCC (Mel-Frequency Cepstral Coefficients) vectors[cite: 220].
* [cite_start]**Architecture:** Dense MLP Encoder/Decoder [cite: 238-239].
* **Goal:** Baseline feature extraction on low-dimensional tabular data.

### 2. Convolutional VAE (Multimodal)
* [cite_start]**Input:** $128 \times 128$ Mel-Spectrograms + TF-IDF Lyric Embeddings[cite: 224, 230].
* [cite_start]**Architecture:** 4-layer Conv2D Encoder capturing hierarchical time-frequency dependencies[cite: 242].
* [cite_start]**Clustering:** Density-based clustering (DBSCAN) on the learned latent manifold[cite: 160].

### 3. Beta-VAE (Disentangled)
* [cite_start]**Objective:** Optimized with a weighted KL-divergence term ($\beta=4.0$) to enforce independence between latent factors[cite: 368, 373].
* [cite_start]**Result:** Produces "blurrier" reconstructions but structurally robust latent spaces that generalize better across genres [cite: 541-543].

## 📉 Results & Analysis

The deep generative approach significantly outperformed linear methods in handling complex, high-dimensional spectral data.

| Model | Clustering Method | Silhouette Score | Key Finding |
| :--- | :--- | :--- | :--- |
| **ConvVAE** | **DBSCAN** | **0.1617** | [cite_start]Best separation for complex spectrogram data[cite: 403]. |
| **Basic VAE** | K-Means | 0.3351 | [cite_start]Effective only for simple MFCC features[cite: 403]. |
| **PCA** | DBSCAN | -0.046 | [cite_start]Failed completely on high-dim spectral data[cite: 403]. |

**Key Insights:**
* [cite_start]**Latent Space Topology:** t-SNE visualizations revealed that the ConvVAE successfully pulled English and Bangla tracks into distinct clusters, whereas PCA resulted in a mashed, entangled manifold [cite: 425-428].
* [cite_start]**Semantic Alignment:** Heatmap analysis showed high cluster purity, with specific clusters aligning strongly with language labels despite the model never seeing those labels during training [cite: 470-472].
* [cite_start]**Generative Trade-off:** The Beta-VAE demonstrated a trade-off where higher disentanglement (better structure) resulted in slightly lower clustering density compared to the standard ConvVAE[cite: 550].

## 🛠️ Tech Stack
* **Language:** Python
* [cite_start]**Deep Learning:** PyTorch / TensorFlow (Implied by layer definitions) [cite: 242]
* [cite_start]**Audio Processing:** Librosa (MFCC/Spectrogram extraction) [cite: 297, 326]
* [cite_start]**Clustering:** Scikit-learn (K-Means, DBSCAN, Agglomerative) [cite: 383, 389]
* [cite_start]**Visualization:** Matplotlib, t-SNE [cite: 407]
