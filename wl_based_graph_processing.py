import numpy as np
import networkx as nx
import hashlib
from collections import defaultdict
import pickle
import os
import json
from sklearn.preprocessing import LabelEncoder

# This code performs a series of operations on a graph representing relationships between network attacks and nodes.
# 1. Apply WL (Weisfeiler-Lehman) node coloring.
# 2. Compute similarity matrix between nodes, based on properties such as attack type and protocols.
# 3. Use centrality to determine variable number of neighbors for each node and create subgraphs.
# 4. Compute hop distances between nodes.
# 5. Save and load intermediate results to avoid repeating expensive calculations.

# --- Utility functions for saving and loading intermediate results ---
def save_intermediate_result(data, filepath):
    # Save intermediate data to a pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_intermediate_result(filepath):
    # Load intermediate results from a file, if it exists. Otherwise return None.
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

# --- Step 1: WL Node Coloring ---
def wl_node_coloring(graph, max_iter=5):
    # Initialize dictionaries to track color and neighbors of each node
    node_color_dict = {}
    node_neighbor_dict = {}

    for node in graph.nodes():
        # Build the initial color based on node properties like type, attack and protocol
        node_type = graph.nodes[node].get('type', 'generic')
        attack_type = graph.nodes[node].get('attack_type', '0')
        protocol = graph.nodes[node].get('r1_Protocollo', 'Unknown')
        flow_duration = graph.nodes[node].get('r1_DurataFlusso', 0)

        # Define the initial node color based on its properties
        node_color_dict[node] = f"{node_type}_{attack_type}_{protocol}_{flow_duration}"
        node_neighbor_dict[node] = list(graph.neighbors(node))  # Save node neighbors

    iteration_count = 0
    while iteration_count < max_iter:
        # Create a new dictionary for updated colors
        new_color_dict = {}
        for node in graph.nodes():
            # Get neighbors' colors
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            # Build a string with the node color and neighbors', then hash to produce a new color
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing

        # Assign unique index to each color
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]

        # Check whether colors have stabilized, if so break loop
        if node_color_dict == new_color_dict:
            break
        else:
            node_color_dict = new_color_dict
        iteration_count += 1

    return node_color_dict

# --- Step 2: Intimacy Calculation and Subgraph Batching ---
class DatasetLoader:
    def __init__(self, graph):
        # Initialize the graph to extract data from
        self.graph = graph

    def load(self):
        # Extract nodes and edges from the graph
        nodes = list(self.graph.nodes)
        edges = list(self.graph.edges)
        # Compute similarity matrix between nodes
        S = self.compute_similarity_matrix(nodes, edges)
        index_id_map = {i: node for i, node in enumerate(nodes)}  # Create index-to-node map
        return {'S': S, 'index_id_map': index_id_map, 'edges': edges, 'nodes': nodes}

    def compute_similarity_matrix(self, nodes, edges):
        # Initialize similarity matrix S with zeros
        S = np.zeros((len(nodes), len(nodes)))
        for u, v in edges:
            # Add a similarity score based on attack type
            if 'attack_type' in self.graph.nodes[u] and 'attack_type' in self.graph.nodes[v]:
                if self.graph.nodes[u]['attack_type'] == self.graph.nodes[v]['attack_type']:
                    S[nodes.index(u), nodes.index(v)] = 1
                    S[nodes.index(v), nodes.index(u)] = 1

            # Add score based on protocol
            if 'r1_Protocollo' in self.graph.nodes[u] and 'r1_Protocollo' in self.graph.nodes[v]:
                if self.graph.nodes[u]['r1_Protocollo'] == self.graph.nodes[v]['r1_Protocollo']:
                    S[nodes.index(u), nodes.index(v)] += 0.5

            # Add score based on received payload size
            if 'r1_PayloadDimensionReceive' in self.graph.nodes[u] and 'r1_PayloadDimensionReceive' in self.graph.nodes[v]:
                if self.graph.nodes[u]['r1_PayloadDimensionReceive'] == self.graph.nodes[v]['r1_PayloadDimensionReceive']:
                    S[nodes.index(u), nodes.index(v)] += 0.5

            # Add score based on sent payload size
            if 'r1_PayloadDimensionSend' in self.graph.nodes[u] and 'r1_PayloadDimensionSend' in self.graph.nodes[v]:
                if self.graph.nodes[u]['r1_PayloadDimensionSend'] == self.graph.nodes[v]['r1_PayloadDimensionSend']:
                    S[nodes.index(u), nodes.index(v)] += 0.5

        return S

# --- Step 3: Adaptive k based on centrality ---
def calculate_adaptive_k(graph, centrality_measure='degree', k_min=3, k_max=10):
    # Compute node centrality using specified measure
    if centrality_measure == 'degree':
        centrality = nx.degree_centrality(graph)
    elif centrality_measure == 'betweenness':
        centrality = nx.betweenness_centrality(graph)
    else:
        raise ValueError("Unsupported centrality measure")

    # Normalize centrality values between 0 and 1
    centrality_values = np.array(list(centrality.values()))
    centrality_norm = (centrality_values - centrality_values.min()) / (centrality_values.max() - centrality_values.min())

    # Adapt k based on normalized centrality
    node_k_values = {node: int(k_min + (k_max - k_min) * centrality_norm[i]) for i, node in enumerate(graph.nodes())}

    return node_k_values

class MethodGraphBatchingAdaptive:
    def __init__(self, data, node_k_values):
        # Initialize data and k-values for each node
        self.data = data
        self.node_k_values = node_k_values

    def run(self):
        # Extract similarity matrix and index-to-node mapping
        S = self.data['S']
        index_id_map = self.data['index_id_map']
        user_top_k_neighbor_intimacy_dict = {}

        for node_index in index_id_map:
            node_id = index_id_map[node_index]
            s = S[node_index]
            s[node_index] = -1000.0  # Exclude self-similarity

            # Get the top k most similar neighbors
            k = self.node_k_values[node_id]
            top_k_neighbor_index = s.argsort()[-k:][::-1]

            # Save neighbors and similarity scores
            user_top_k_neighbor_intimacy_dict[node_id] = []
            for neighbor_index in top_k_neighbor_index:
                neighbor_id = index_id_map[neighbor_index]
                user_top_k_neighbor_intimacy_dict[node_id].append((neighbor_id, s[neighbor_index]))

        return user_top_k_neighbor_intimacy_dict

# --- Step 4: Hop Distance Calculation ---
class MethodHopDistance:
    def __init__(self, data, k, dataset_name, max_hop_distance=10):
        # Initialize data, k, dataset name, and maximum hop distance
        self.data = data
        self.k = k
        self.dataset_name = dataset_name
        self.max_hop_distance = max_hop_distance

    def run(self):
        # Create a graph with nodes and edges from dataset
        node_list = self.data['nodes']
        edge_list = self.data['edges']
        G = nx.Graph()
        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)

        # Load previously computed batches
        with open('path/to/results/batched_graph.pkl', 'rb') as f:
            batch_dict = pickle.load(f)

        hop_dict = {}
        # Compute hop distances between nodes in the batch
        for node in batch_dict:
            if node not in hop_dict:
                hop_dict[node] = {}
            for neighbor, score in batch_dict[node]:
                try:
                    hop = nx.shortest_path_length(G, source=node, target=neighbor)
                except nx.NetworkXNoPath:
                    hop = 99  # Assign a high value if no path exists
                hop = min(hop, self.max_hop_distance)
                hop_dict[node][neighbor] = hop
        return hop_dict

# --- Result Saving ---
class ResultSaving:
    def __init__(self, data, result_destination_folder_path, result_destination_file_name):
        # Initialize data and destination path for saving the result
        self.data = data
        self.result_destination_folder_path = result_destination_folder_path
        self.result_destination_file_name = result_destination_file_name

    def save(self):
        # Save data to a pickle file
        with open(self.result_destination_folder_path + self.result_destination_file_name, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self):
        # Load data from a pickle file
        with open(self.result_destination_folder_path + self.result_destination_file_name, 'rb') as f:
            result = pickle.load(f)
        return result

# --- Load graph from JSON file and apply Label Encoding ---
graph_path = 'path/to/graph.json'
with open(graph_path, 'r') as f:
    json_data = json.load(f)

G = nx.Graph()

# Apply Label Encoding for attack types
attack_types = []
for entry in json_data:
    if 'at' in entry:
        attack_types.append(entry['at']['properties'].get('Attack_type', 'Unknown'))

# Convert attack types to numeric values
label_encoder = LabelEncoder()
encoded_attack_types = label_encoder.fit_transform(attack_types)

attack_type_mapping = dict(zip(attack_types, encoded_attack_types))

for entry in json_data:
    attack_node = entry['at']['identity']
    source = entry['sp']['identity']
    target = entry['dp']['identity']

    # Add r1 relationship properties to source node
    source_properties = entry['sp']['properties']
    r1_properties = entry['r1']['properties']
    for prop, value in r1_properties.items():
        source_properties[f"r1_{prop}"] = value

    # Add r1 properties to target node
    target_properties = entry['dp']['properties']
    for prop, value in r1_properties.items():
        target_properties[f"r1_{prop}"] = value

    # Add attack_type to source and target nodes
    if 'at' in entry:
        attack_type = entry['at']['properties'].get('Attack_type', 'Unknown')
        source_properties['attack_type'] = attack_type
        target_properties['attack_type'] = attack_type

    # Add nodes and edges to the graph
    G.add_node(source, **source_properties)
    G.add_node(target, **target_properties)
    G.add_edge(source, target, **entry['r1']['properties'])

    if 'at' in entry:
        encoded_attack_type = attack_type_mapping[attack_type]

        # Add r2 relationship properties to source and attack nodes
        r2_properties = entry['r2']['properties']
        for prop, value in r2_properties.items():
            source_properties[f"r2_{prop}"] = value
            entry['at']['properties'][f"r2_{prop}"] = value

        # Add attack node with its encoded attack type
        G.add_node(attack_node, **entry['at']['properties'])
        G.nodes[attack_node]['Attack_type_encoded'] = encoded_attack_type  # Encode attack type
        G.add_edge(attack_node, source, **entry['r2']['properties'])

# Step 1: WL Node Coloring
wl_filepath = 'path/to/results/wl_coloring.pkl'
node_colors = load_intermediate_result(wl_filepath)
if node_colors is None:
    # Apply WL node coloring and save results
    node_colors = wl_node_coloring(G, max_iter=5)
    save_intermediate_result(node_colors, wl_filepath)
print("WL Coloring completed!")

# Step 2: Compute centrality and adapt k value
node_k_values = calculate_adaptive_k(G, centrality_measure='degree', k_min=3, k_max=10)

# Step 3: Load graph and compute similarity matrix
dataset_loader = DatasetLoader(G)
data = dataset_loader.load()

# Step 4: Compute intimacy between nodes and batch into subgraphs
intimacy_filepath = 'path/to/results/batched_graph.pkl'
intimacy_batches = load_intermediate_result(intimacy_filepath)
if intimacy_batches is None:
    method_graph_batching = MethodGraphBatchingAdaptive(data, node_k_values)
    intimacy_batches = method_graph_batching.run()
    save_intermediate_result(intimacy_batches, intimacy_filepath)
print("Intimacy calculation and subgraph batching with adaptive k completed and saved!")

# Step 5: Compute Hop Distance
hop_filepath = 'path/to/results/hop_distances.pkl'
hop_distances = load_intermediate_result(hop_filepath)
if hop_distances is None:
    method_hop_distance = MethodHopDistance(data, k=5, dataset_name='graph')
    hop_distances = method_hop_distance.run()
    save_intermediate_result(hop_distances, hop_filepath)
print("Hop Distance calculation completed and saved!")
