# KLAGE
KLAGE

This repository contains the inference and reporting components of KLAGE (Knowledge-based Language-Aided Graph Embeddings), a framework designed to improve intrusion detection and threat analysis in complex network environments, particularly in the Internet of Things (IoT) domain.
Overview

KLAGE addresses the limitations of traditional security systems in detecting stealthy and sophisticated cyber attacks such as:

    Distributed Denial of Service (DDoS)

    ARP Poisoning

    Reconnaissance Scans

The methodology combines:

    Knowledge Graphs, to represent communication patterns and enrich log data with contextual metadata;

    Graph-BERT, to capture structural and semantic relationships through graph-based embeddings;

    Large Language Models (LLMs), for automated, natural-language threat reporting (not included in this repository);

    LIME, for model interpretability by identifying the features influencing predictions.

What This Repository Includes

This codebase focuses on the inference phase of the KLAGE pipeline and the generation of structured outputs for analysis and reporting. It includes:

    Loading pre-processed node features and adjacency matrices from .pkl files;

    Reconstructing attack connections following the pattern attack → source → destination;

    Performing inference using a fine-tuned Graph-BERT model;

    Extracting additional metadata (e.g., protocol, service, traffic statistics) from source nodes;

    Classifying each connection into one of several attack types;

    Generating a CSV report containing:

        Source and destination ports

        Protocol and service

        Predicted attack type and subclass

        Scenario label (benign/malicious)

        Attack vector (external/internal/lateral)

        Flow-related features

        Prediction status (correct/incorrect)

Use Case

KLAGE has been tested on benchmark datasets such as RT-IoT2022 and CIC-IoT2023, achieving classification accuracy above 84%. It provides not only high detection performance, but also explainable outputs that can support cybersecurity analysts in quickly understanding threats and making informed decisions.
Next Steps

The next sections of the README will provide a breakdown of the pipeline, including:

    Dependencies and setup

    Code structure

    How to run the inference

    Output format

    Notes on model training and dataset preparation (optional)
