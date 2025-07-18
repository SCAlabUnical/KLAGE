# KLAGE


**KLAGE**

This repository contains the inference and reporting components of KLAGE (Knowledge-based Language-Aided Graph Embeddings), a framework designed to improve intrusion detection and threat analysis in complex network environments, particularly in the Internet of Things (IoT) domain.

**Overview**

KLAGE addresses the limitations of traditional security systems in detecting stealthy and sophisticated cyber attacks such as:

    + Distributed Denial of Service (DDoS)
    + ARP Poisoning
    + Reconnaissance Scan
    
The methodology combines:

    + Knowledge Graphs, to represent communication patterns and enrich log data with contextual metadata;
    + Graph-BERT, to capture structural and semantic relationships through graph-based embeddings;
    + Large Language Models (LLMs), for automated, natural-language threat reporting (not included in this repository);
    + LIME, for model interpretability by identifying the features influencing predictions.

**What This Repository Includes**

This codebase focuses on the inference phase of the KLAGE pipeline and the generation of structured outputs for analysis and reporting. It includes:

    + Loading pre-processed node features and adjacency matrices from .pkl files;
    + Reconstructing attack connections following the pattern attack → source → destination;
    + Performing inference using a fine-tuned Graph-BERT model;
    + Extracting additional metadata (e.g., protocol, service, traffic statistics) from source nodes;
    + Classifying each connection into one of several attack types;
    + Generating a CSV report containing:

        - Source and destination ports
        - Protocol and service
        - Predicted attack type and subclass
        - Scenario label (benign/malicious)
        - Attack vector (external/internal/lateral)
        - Flow-related features
        - Prediction status (correct/incorrect)

**Use Case**

KLAGE has been tested on benchmark datasets such as RT-IoT2022 and CIC-IoT2023. It provides not only high detection performance, but also explainable outputs that can support cybersecurity analysts in quickly understanding threats and making informed decisions.

Next Steps

The next sections of the README will provide a breakdown of the pipeline, including:

**Phase 1 – Graph Preprocessing and Feature Extraction**

File: wl_based_graph_processing.py

This phase is responsible for preparing the graph data for model training and inference. It performs multiple preprocessing steps on a graph built from network traffic logs, where nodes represent source devices, destination devices, and attack instances.

Main operations performed:

    + Graph construction from JSON files containing relationships between attacks (at), sources (sp), and destinations (dp);
    + Node feature enrichment by embedding relationship attributes (r1, r2) into node metadata;
    + Attack type encoding using LabelEncoder to transform attack labels into numeric values;
    + Weisfeiler-Lehman (WL) node coloring to compute unique structural representations for nodes based on their neighbors;
    + Similarity matrix computation between nodes based on shared attributes like protocol and payload size;
    + Adaptive neighbor selection (k-value) using centrality metrics to define how many neighbors each node should consider;
    + Subgraph batching: each node is associated with its top-k most similar neighbors;
    + Hop distance calculation to determine the shortest path between nodes, bounded by a max threshold;
    + Caching of intermediate results (WL coloring, batches, hop distances) using Pickle to avoid recomputation.

These processed structures (node colors, neighbor batches, hop distances) are saved for reuse in later training or inference steps.

**Phase 2 – Feature and Adjacency Matrix Generation**

File: generate_graph_features_and_adjacency.py

In this phase, the JSON graph is processed to create the node feature matrix and the adjacency matrix, which are required for inference with Graph-BERT.

The script loads the graph structure where each entry includes a source node (sp), a destination node (dp), and an attack node (at). Categorical attributes such as Attack_type, Protocollo, and Servizio are encoded numerically using LabelEncoder.

Each node is transformed into a fixed-length feature vector (max_feature_length = 15).

    + Source nodes (sp) include:
        - their own traffic statistics
        - destination-related metrics (e.g. received data)
    + Destination nodes (dp) are encoded based on their attributes and edge properties
    + Attack nodes (at) store the encoded attack type and associated service
    
A NetworkX graph is then built to reflect the connections:

    + at nodes are connected to sp via the r2 relationship
    + sp nodes are connected to dp via the r1 relationship

Finally, the script:

    + Saves the node feature matrix as node_features.pkl
    + Saves the adjacency matrix as adjacency_matrix.pkl

**Phase 3 – Pre-training**

File: graphbert_pretraining.py

This phase defines the Graph-BERT architecture and prepares the input data for training.

A lightweight BERT-based model is configured with 4 hidden layers and 2 attention heads. It is designed to operate on graph-structured data and includes structural encodings such as Weisfeiler-Lehman roles, hop distances, and node position indices.

The BertLayer module performs self-attention with residual connections and intermediate transformations. These layers are stacked within the BertEncoder, which propagates node information through the network. An optional mask is applied to increase the influence of attack nodes during training.

The data loader handles the input preparation pipeline:

    + Loads node features and adjacency matrix from .pkl files
    + Applies Min-Max normalization and adjusts dimensionality (padding if needed)
    + Builds embeddings for WL roles, hop distances, and position indices
    + Averages raw features from each node and its neighbors
    + Generates a binary mask to identify attack nodes (value = 1 for attacks)

Finally, all processed components are returned as PyTorch tensors:
X, A, wl_embedding, hop_embeddings, int_embeddings, raw_embeddings, and attack_mask.

**Phase 4 – Fine-Tuning the Graph-BERT Model**

File: graphbert_finetuning.py

This phase focuses on training a custom Graph-BERT model to classify attack types based on IoT connection patterns.

    + The model takes embeddings from three node types per connection: source (sp), destination (dp), and attack (at). Each is represented by 13 features and projected to BERT's             hidden size.
    + A GraphBertForFineTuning class is defined, based on BertModel, with a classifier on top to predict the attack class.

The training data is constructed by identifying all valid triples (at → sp → dp) from the adjacency matrix. Node features are extracted and converted into PyTorch tensors. Attention masks are generated to fit the BERT input format.

    + The model is trained using cross-entropy loss and the Adam optimizer.
    + A learning rate scheduler (ReduceLROnPlateau) adjusts the learning rate when the loss plateaus.
    + Early stopping is applied: if no accuracy improvement is observed over 200 evaluations, training is stopped.

Validation accuracy is checked every 10 epochs, and the best-performing model is saved automatically.

The final model is stored as a .pth file and is ready for inference.

**Phase 5 – Inference and Evaluation**

File: inference_graphbert.py

This phase performs inference using the fine-tuned Graph-BERT model to classify attack types on new IoT network data.

    + The script loads node features and the adjacency matrix of the prediction graph and reconstructs all valid at → sp → dp connections.
    + For each connection, it extracts the first 13 features from the source (sp), destination (dp), and attack (at) nodes, formats them as input tensors, and generates predictions          using the trained model.

The predicted labels are compared to ground truth values already encoded in the dataset to evaluate accuracy. Each connection is enriched with contextual information:

    + Protocol and service type
    + Source and destination ports
    + Scenario (Benign or Malicious)
    + Attack vector (Internal, External, Lateral Movement)ì
    + Subclass (e.g., DoS, Reconnaissance)

The results are saved into a CSV file with a complete breakdown of correct and incorrect predictions, enriched with detailed features from the source node. Final accuracy is printed to the console.

This step completes the full evaluation pipeline and prepares the dataset for report generation.


    

    
