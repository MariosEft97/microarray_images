## Project: Peptide Array Image Analysis

This project focuses on the quantitative analysis of peptide microarray experiments through computational image processing and segmentation techniques. The developed framework extracts spot-level signal intensities from peptide array images and transforms these measurements into peptide substitution probabilities, enabling the characterization of sequence preferences and binding specificities.

### Repository Overview

#### Core Module (`module.py`)
A collection of functions supporting the complete image analysis workflow.

**Features**
- Preprocesses peptide array images for downstream analysis.
- Performs spot detection and image segmentation to identify individual microarray features.
- Extracts and quantifies spot intensity measurements.
- Applies normalization and quality-control procedures.
- Converts signal intensity profiles into peptide substitution probability matrices.
- Provides utility functions used throughout the analysis pipeline.

#### Analysis Notebook (`notebook.py`)
An interactive workflow demonstrating the complete analysis pipeline.

**Features**
- Executes the image processing and segmentation workflow.
- Integrates the functions implemented in `module.py`.
- Generates visualizations and summary statistics of peptide array measurements.
- Computes peptide substitution probabilities from the extracted intensity data.
