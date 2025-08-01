import json
import numpy as np
import pickle
import networkx as nx
from sklearn.preprocessing import LabelEncoder

# This code loads a graph from a JSON file, applies label encoding to properties such as attack type, protocol, and service,
# creates nodes and edges for each entry in the graph, and builds a node feature matrix and an adjacency matrix.
# Then it saves the features and the adjacency matrix in .pkl format

# File paths
graph_path = '/path/to/your/graph.json'
node_features_path = '/path/to/save/node_features.pkl'
adjacency_matrix_path = '/path/to/save/adjacency_matrix.pkl'

# Define the maximum number of features per node
max_feature_length = 15  # Modified to reflect the additional features

# Load the graph from the JSON file
with open(graph_path, 'r') as f:
    json_data = json.load(f)

# Initialize lists to store attack types, protocols, and services
attack_types = []
protocols = []
services = []

# Iterate over the entries in the JSON file to collect the Attack_type, Protocol, and Service properties
for entry in json_data:
    if 'at' in entry:
        attack_types.append(entry['at']['properties'].get('Attack_type', 'Unknown'))
    protocols.append(entry['r1']['properties'].get('Protocol', 'Unknown'))
    services.append(entry['r2']['properties'].get('Service', 'Unknown'))

# Apply Label Encoding to attack types, protocols, and services
label_encoder_attack = LabelEncoder()
encoded_attack_types = label_encoder_attack.fit_transform(attack_types)

label_encoder_protocol = LabelEncoder()
encoded_protocols = label_encoder_protocol.fit_transform(protocols)

label_encoder_service = LabelEncoder()
encoded_services = label_encoder_service.fit_transform(services)

# Create mappings from original values to encoded values
attack_type_mapping = dict(zip(attack_types, encoded_attack_types))
protocol_mapping = dict(zip(protocols, encoded_protocols))
service_mapping = dict(zip(services, encoded_services))

# Dictionary to count the number of occurrences for each attack type
attack_count = {attack: 0 for attack in attack_types}

# Create a dictionary to store the features of the nodes
node_features = {}

# Iterate again over the graph entries to create nodes and edges
for entry in json_data:
    attack_node = entry['at']['identity']  # Attack node
    source_node = entry['sp']['identity']  # Source node
    dest_node = entry['dp']['identity']    # Destination node

    # Merge the properties of edge r1 into the source (sp) and destination (dp) nodes
    source_properties = entry['sp']['properties']
    dest_properties = entry['dp']['properties']
    r1_properties = entry['r1']['properties']
    r2_properties = entry['r2']['properties']  # r2 properties for the attack node

    # Encode the protocol
    protocol = r1_properties.get('Protocol', 'Unknown')
    encoded_protocol = protocol_mapping.get(protocol, -1)

    # Encode the attack type to keep track of the count
    attack_type = entry['at']['properties'].get('Attack_type', 'Unknown')
    encoded_attack_type = attack_type_mapping.get(attack_type, -1)  # Use -1 if attack type not found

    # Increase the attack type count
    if attack_type in attack_count:
        attack_count[attack_type] += 1

    # Check if the attack type was correctly mapped
    if encoded_attack_type == -1:
        print(f"Attack '{attack_type}' not found in mapping. Fallback value used: {encoded_attack_type}")

    # Create source node (sp) if it doesn't already exist
    if source_node not in node_features:
        source_feats = [
            -1,  # Source identifier
            float(source_properties.get('Total_data_sent', 0)),
            float(source_properties.get('Source_port', 0)),
            float(source_properties.get('Packets_sent', 0)),
            float(r1_properties.get('FlowDuration', 0)),
            float(r1_properties.get('PayloadDimensionSend', 0)),
            float(r1_properties.get('PktReceiveInterval', 0)),
            float(encoded_protocol),
            float(r1_properties.get('FrequencySend', 0)),
            float(r1_properties.get('PktSendInterval', 0)),
            float(r1_properties.get('PayloadDimensionReceive', 0)),
            float(r1_properties.get('FrequencyReceive', 0))

        ]

        # Add destination (dp) features to the source node (sp) without the attack type
        source_feats.extend([
        float(dest_properties.get('Total_data_received', 0)),
        float(dest_properties.get('Destination_port', 0)),
        float(dest_properties.get('Packets_received', 0))

        ])

        # Add padding to ensure all features have the same length
        source_feats = np.pad(source_feats, (0, max_feature_length - len(source_feats)), 'constant')
        node_features[source_node] = np.array(source_feats, dtype=float)

    # Create destination node (dp) if it doesn't already exist
    if dest_node not in node_features:
        dest_feats = [
            -2,  # Destination identifier
            float(dest_properties.get('Total_data_received', 0)),
            float(dest_properties.get('Destination_port', 0)),
            float(dest_properties.get('Packets_received', 0)),
            float(r1_properties.get('Flow_duration', 0)),
            float(r1_properties.get('Payload_size_send', 0)),
            float(r1_properties.get('Pkt_receive_interval', 0)),
            float(encoded_protocol),
            float(r1_properties.get('Send_frequency', 0)),
            float(r1_properties.get('Pkt_send_interval', 0)),
            float(r1_properties.get('Payload_size_receive', 0)),
            float(r1_properties.get('Receive_frequency', 0))

        ]

        # Add padding to normalize feature length
        dest_feats = np.pad(dest_feats, (0, max_feature_length - len(dest_feats)), 'constant')
        node_features[dest_node] = np.array(dest_feats, dtype=float)

    # Create attack node (at) if it doesn't already exist
    if attack_node not in node_features:
        # Encode attack type and service
        service = r2_properties.get('Servizio', 'Unknown')
        encoded_service = service_mapping.get(service, -1)

        # Indicate it's an attack node (-3)
        attack_feats = [
            -3,  # Attack identifier
            float(encoded_attack_type),
            float(encoded_service)
        ]
        # Add padding to normalize feature length
        attack_feats = np.pad(attack_feats, (0, max_feature_length - len(attack_feats)), 'constant')
        node_features[attack_node] = np.array(attack_feats, dtype=float)

    # Add r2 properties to the source node (caused)
    node_features[source_node] = np.append(node_features[source_node], float(encoded_service))
    if len(node_features[source_node]) > max_feature_length:
        node_features[source_node] = node_features[source_node][:max_feature_length]

# Make sure all nodes have exactly max_feature_length features
for node in node_features:
    if len(node_features[node]) < max_feature_length:
        node_features[node] = np.pad(node_features[node], (0, max_feature_length - len(node_features[node])), 'constant')

# Print only the first 30 node features
print("First 30 node features:")
for idx, (node, features) in enumerate(node_features.items()):
    if idx < 30:
        print(f"Node: {node}, Features: {features}")
    else:
        break

# Print the count for each attack type
print("\nCount for each attack type:")
for attack_type, count in attack_count.items():
    print(f"{attack_type}: {count}")

# Convert node features to a matrix
node_list = list(node_features.keys())
feature_matrix = np.vstack([node_features[node] for node in node_list])

# Save node features as .pkl file
with open(node_features_path, 'wb') as f:
    pickle.dump(feature_matrix, f)

print(f"Node features saved to {node_features_path}")

# Create the NetworkX graph and adjacency matrix
G = nx.Graph()

for entry in json_data:
    attack_node = entry['at']['identity']
    source_node = entry['sp']['identity']
    dest_node = entry['dp']['identity']

    # Add r2 edge between at and sp (caused)
    G.add_edge(attack_node, source_node)

    # Add r1 edge between sp and dp (connected)
    G.add_edge(source_node, dest_node)

# Extract adjacency matrix from the graph
adj_matrix = nx.to_numpy_array(G, nodelist=node_list)

# Save adjacency matrix as .pkl file
with open(adjacency_matrix_path, 'wb') as f:
    pickle.dump(adj_matrix, f)

print(f"Adjacency matrix saved to {adjacency_matrix_path}")

# Check the generated files and print only the first 200 rows of the adjacency matrix
with open(node_features_path, 'rb') as f:
    loaded_node_features = pickle.load(f)
    print("First 200 node features:")
    print(loaded_node_features[:400])

with open(adjacency_matrix_path, 'rb') as f:
    loaded_adj_matrix = pickle.load(f)
    print("First 200 rows of the adjacency matrix:")
    print(loaded_adj_matrix[:100])
