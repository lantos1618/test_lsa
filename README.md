# Latent Semantic Analysis (LSA)

A Nim implementation of Latent Semantic Analysis for document similarity and information retrieval.

## Overview

This project implements key NLP techniques from scratch:

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Quantifies how important each word is to a document in a collection
- **SVD (Singular Value Decomposition)**: Decomposes the TF-IDF matrix into latent semantic dimensions
- **Cosine Similarity**: Measures how similar two documents are based on their semantic vectors
- **Power Iteration Method**: Computes eigenvalues and eigenvectors for SVD calculation

## How It Works

1. **Preprocessing**: Documents are tokenized and lowercased
2. **TF-IDF Calculation**: Creates a term-document matrix weighted by importance
3. **SVD Decomposition**: Reduces the matrix to capture latent semantic relationships
4. **Document Comparison**: Uses cosine similarity to find related documents

The implementation includes helper functions for matrix operations (transpose, multiplication, normalization) and eigenvalue computation.

## Building

```bash
nimble build
```

## Running Tests

```bash
nim r src/test_lsa.nim
```

## Features

- Pure Nim implementation with no external dependencies (except standard library)
- Demonstrates fundamental linear algebra operations
- Practical example of semantic text analysis
