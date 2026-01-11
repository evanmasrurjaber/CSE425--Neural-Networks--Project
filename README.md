Multimodal Music Clustering through Unsupervised Learning using Variational Autoencoders

Project Overview

This project implements an unsupervised learning pipeline utilizing Variational Autoencoders (VAE) to extract latent representations for clustering music tracks. The system addresses the challenge of clustering "hybrid" datasets containing tracks from multiple languages (English and Bangla) and modalities (Audio and Lyrics).

The project evaluates the trade-off between reconstruction quality and cluster separability across three architectural complexities: Standard VAE, Convolutional VAE (ConvVAE), and Beta-VAE.

Key Features & Tasks

The project is divided into three complexity levels as defined in the experimental design :

Easy Task (Baseline): Comparison of Standard VAE against linear PCA using MFCC features on a hybrid audio dataset.

Medium Task (Multimodal Fusion): Implementation of a Convolutional VAE (ConvVAE) fusing Mel-spectrograms with TF-IDF lyric embeddings. Evaluated using density-based clustering (DBSCAN).

Hard Task (Disentanglement): Implementation of a Beta-VAE to enforce disentangled latent representations on a complex dataset (Audio + Lyrics + Genre).
