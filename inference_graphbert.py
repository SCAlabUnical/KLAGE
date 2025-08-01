import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import csv
from transformers import BertConfig, BertModel

# Set the seed for random number generation to ensure reproducibility
seed = 42
torch.manual_seed(seed)  # Set the seed for PyTorch
torch.cuda.manual_seed_all(seed)  # Set the seed for all CUDA devices
np.random.seed(seed)  # Set the seed for numpy

# File paths for loading node features, adjacency matrix, and trained model
nodes_features_pred_path = '/your/path/to/nodes_features_prediction.pkl'
adjacency_matrix_pred_path = '/your/path/to/adjacency_matrix_prediction.pkl'
fine_tuned_model_path = '/your/path/to/best_fine_tuned_model.pth'
csv_output_path = '/your/path/to/connections_predicted.csv'

# Mapping of protocols and services based on numerical codes
protocol_mapping = {1: 'udp', 0: 'tcp'}
service_mapping = {1: 'dns', 0: '-', 2: 'http', 3: 'mqtt', 5: 'ssl', 4: 'ntp'}

# Mapping of attack types to their corresponding numeric labels
attack_type_mapping = {
    'ARP_poisioning': 0,
    'DDOS_Slowloris': 1,
    'DOS_SYN_Hping': 2,
    'MQTT_Publish': 3,
    'NMAP_OS_DETECTION': 4,
    'NMAP_TCP_scan': 5,
    'NMAP_XMAS_TREE_SCAN': 6,
    'Thing_Speak': 7,
    'Wipro_bulb': 8
}

# Reverse mapping to decode predicted labels into attack types
reverse_attack_type_mapping = {v: k for k, v in attack_type_mapping.items()}

# Mapping of subclasses to identify the category of each attack
subclass_mapping = {
    'ARP_poisioning': 'MITM',
    'DDOS_Slowloris': 'DoS',
    'DOS_SYN_Hping': 'DoS',
    'NMAP_OS_DETECTION': 'Reconnaissance',
    'NMAP_TCP_scan': 'Reconnaissance',
    'NMAP_XMAS_TREE_SCAN': 'Reconnaissance'
}

# Function to determine if the scenario is benign or malicious
def determine_scenario(attack_type_str):
    if attack_type_str in ['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb']:
        return 'Benign'
    else:
        return 'Malicious'

# Function to determine the attack vector based on protocol and service
def determine_attack_vector(proto, service):
    if proto == 'udp' and service == 'dns':
        return 'External'  # Example of external attack
    elif proto == 'tcp' and service in ['http', 'ssh']:
        return 'Lateral Movement'  # Example of lateral movement
    else:
        return 'Internal'  # Otherwise, assume it's an internal attack

# Load node features for inference
with open(nodes_features_pred_path, 'rb') as f:
    nodes_features_pred = pickle.load(f)

# Load the adjacency matrix to reconstruct the connections
with open(adjacency_matrix_pred_path, 'rb') as f:
    adjacency_matrix_pred = pickle.load(f)

# GraphBert model for inference, identical to the one used for fine-tuning
class GraphBertForFineTuning(nn.Module):
    def __init__(self, config):
        super(GraphBertForFineTuning, self).__init__()
        self.bert_encoder = BertModel(config)
        self.projection = nn.Linear(13, config.hidden_size)  # Layer to project embeddings to hidden_size
        self.dropout = nn.Dropout(p=0.2)  # Dropout for regularization
        self.classifier = nn.Linear(config.hidden_size, 9)  # Final classifier with 9 classes

    def forward(self, sp_embeds, dp_embeds, at_embeds, attention_mask_sp=None, attention_mask_dp=None, attention_mask_at=None):
        # Project embeddings into the required dimension for BERT
        sp_embeds = self.projection(sp_embeds)
        dp_embeds = self.projection(dp_embeds)
        at_embeds = self.projection(at_embeds)

        # Get BERT output for each type of embedding
        sp_output = self.bert_encoder(inputs_embeds=sp_embeds, attention_mask=attention_mask_sp).last_hidden_state[:, 0, :]
        dp_output = self.bert_encoder(inputs_embeds=dp_embeds, attention_mask=attention_mask_dp).last_hidden_state[:, 0, :]
        at_output = self.bert_encoder(inputs_embeds=at_embeds, attention_mask=attention_mask_at).last_hidden_state[:, 0, :]

        # Combine embeddings via summation
        combined_output = sp_output + dp_output + at_output
        combined_output = self.dropout(combined_output)  # Apply dropout

        # Final classification to obtain logits
        logits = self.classifier(combined_output)

        return logits

# Function to reconstruct at --> sp --> dp connections from loaded data
def reconstruct_connections(nodes_features, adjacency_matrix):
    connections = []
    for at_id, at_features in enumerate(nodes_features):
        if at_features[0] == -3:  # Attack node (at)
            for sp_id, is_attached in enumerate(adjacency_matrix[at_id]):
                if is_attached and nodes_features[sp_id][0] == -1:  # Source node (sp)
                    for dp_id, is_connected in enumerate(adjacency_matrix[sp_id]):
                        if is_connected and nodes_features[dp_id][0] == -2:  # Destination node (dp)
                            # Extract relevant information for the connection
                            attack_type_pred = int(nodes_features[at_id][1])  # Predicted attack type
                            sp_port = int(nodes_features[sp_id][2])  # Source port
                            dp_port = int(nodes_features[dp_id][2])  # Destination port
                            # Add the connection to the list
                            connections.append((at_id, sp_id, dp_id, sp_port, dp_port, attack_type_pred))
    return connections

# Function to decode protocol and service from feature vector
def decode_protocol_service(node_features, attack_node_features):
    protocol_code = int(node_features[7])  # Protocol is at position 7 in the vector
    service_code = int(attack_node_features[2])  # Service is at position 2 in the attack node
    protocol = protocol_mapping.get(protocol_code, 'Unknown')  # Decode protocol
    service = service_mapping.get(service_code, 'Unknown')  # Decode service
    return protocol, service

# Function to extract all relevant features from the source node (sp)
def extract_sp_features(node_features):
    # Create a dictionary with relevant features from sp node
    features = {
        'Total_data_sent': node_features[1],
        'Packets_sent': node_features[3],
        'Flow_duration': node_features[4],
        'Payload_size_send': node_features[5],
        'Pkt_receive_interval': node_features[6],
        'Send_frequency': node_features[8],
        'Pkt_send_interval': node_features[9],
        'Payload_size_receive': node_features[10],
        'Receive_frequency': node_features[11],
        'Total_data_received': node_features[12],
        'Packets_received': node_features[14]

    }
    return features

# Reconstruct predicted connections from loaded data using at --> sp --> dp structure
connections_pred_filtered = reconstruct_connections(nodes_features_pred, adjacency_matrix_pred)

# Prepare input data for inference only for filtered attacks
X_sp_inference_filtered = []
X_dp_inference_filtered = []
X_at_inference_filtered = []
source_ports_filtered = []  # Source ports for each connection
destination_ports_filtered = []  # Destination ports for each connection

# Extract embeddings and ports for each connection
for at_id, sp_id, dp_id, sp_port, dp_port, attack_type_pred in connections_pred_filtered:
    X_sp_inference_filtered.append(nodes_features_pred[sp_id][:13])
    X_dp_inference_filtered.append(nodes_features_pred[dp_id][:13])
    X_at_inference_filtered.append(nodes_features_pred[at_id][:13])

    source_ports_filtered.append(sp_port)
    destination_ports_filtered.append(dp_port)

# Convert data to PyTorch tensors
X_sp_inference_filtered = torch.FloatTensor(X_sp_inference_filtered).unsqueeze(1)
X_dp_inference_filtered = torch.FloatTensor(X_dp_inference_filtered).unsqueeze(1)
X_at_inference_filtered = torch.FloatTensor(X_at_inference_filtered).unsqueeze(1)

# Create attention masks required for BERT model
attention_mask_sp_filtered = torch.ones(X_sp_inference_filtered.shape[:-1])
attention_mask_dp_filtered = torch.ones(X_dp_inference_filtered.shape[:-1])
attention_mask_at_filtered = torch.ones(X_at_inference_filtered.shape[:-1])

# Configure the BERT model with hidden_size = 13 for inference
config = BertConfig(hidden_size=13, num_attention_heads=1)
model = GraphBertForFineTuning(config)
# Load the fine-tuned model from saved weights
model.load_state_dict(torch.load(fine_tuned_model_path, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Perform inference without gradient computation to save memory
with torch.no_grad():
    predictions_filtered = model(X_sp_inference_filtered, X_dp_inference_filtered, X_at_inference_filtered,
                                 attention_mask_sp_filtered, attention_mask_dp_filtered, attention_mask_at_filtered)
    # Get predicted classes as final output from model
    predicted_classes_filtered = torch.argmax(predictions_filtered, dim=1)

# Calculate accuracy and identify discrepancies between predictions and actual values
correct_predictions = []
discrepancies = []
correct = 0

# Check each prediction and compare with actual value
for i, pred_class in enumerate(predicted_classes_filtered):
    real_attack_type = int(nodes_features_pred[connections_pred_filtered[i][0]][1])
    pred_attack_type = pred_class.item()

    # Decode protocol and service for source node (sp)
    sp_protocol, sp_service = decode_protocol_service(nodes_features_pred[connections_pred_filtered[i][1]],
                                                      nodes_features_pred[connections_pred_filtered[i][0]])

    # Extract source node (sp) features
    sp_features = extract_sp_features(nodes_features_pred[connections_pred_filtered[i][1]])

    # Determine scenario and attack vector
    attack_type_str = reverse_attack_type_mapping[pred_attack_type]
    scenario = determine_scenario(attack_type_str)
    attack_vector = determine_attack_vector(sp_protocol, sp_service)
    subclass = subclass_mapping.get(attack_type_str, "Unknown")

    # Check if prediction is correct
    if real_attack_type == pred_attack_type:
        status = "correct"
        correct += 1
        # Add correct prediction to the list
        correct_predictions.append({

            'Source_port': source_ports_filtered[i],
            'Destination_port': destination_ports_filtered[i],
            'Protocol': sp_protocol,
            'Service': sp_service,
            'Attack_type': attack_type_str,
            'Subclass': subclass,
            'Scenario': scenario,
            'Attack_vector': attack_vector,
            'Total_data_sent': sp_features['Total_data_sent'],
            'Packets_sent': sp_features['Packets_sent'],
            'Flow_duration': sp_features['Flow_duration'],
            'Payload_size_send': sp_features['Payload_size_send'],
            'Pkt_receive_interval': sp_features['Pkt_receive_interval'],
            'Send_frequency': sp_features['Send_frequency'],
            'Pkt_send_interval': sp_features['Pkt_send_interval'],
            'Payload_size_receive': sp_features['Payload_size_receive'],
            'Receive_frequency': sp_features['Receive_frequency'],
            'Total_data_received': sp_features['Total_data_received'],
            'Packets_received': sp_features['Packets_received'],
            'Status': status


        })
    else:
        status = "incorrect"
        # Add discrepancy to the list
        discrepancies.append({
            'Source_port': source_ports_filtered[i],
            'Destination_port': destination_ports_filtered[i],
            'Protocol': sp_protocol,
            'Service': sp_service,
            'Attack_type': attack_type_str,
            'Subclass': subclass,
            'Scenario': scenario,
            'Attack_vector': attack_vector,
            'Total_data_sent': sp_features['Total_data_sent'],
            'Packets_sent': sp_features['Packets_sent'],
            'Flow_duration': sp_features['Flow_duration'],
            'Payload_size_send': sp_features['Payload_size_send'],
            'Pkt_receive_interval': sp_features['Pkt_receive_interval'],
            'Send_frequency': sp_features['Send_frequency'],
            'Pkt_send_interval': sp_features['Pkt_send_interval'],
            'Payload_size_receive': sp_features['Payload_size_receive'],
            'Receive_frequency': sp_features['Receive_frequency'],
            'Total_data_received': sp_features['Total_data_received'],
            'Packets_received': sp_features['Packets_received'],
            'Status': status

        })

# Calculate total prediction accuracy
total_connections = len(connections_pred_filtered)
accuracy = (correct / total_connections) * 100 if total_connections > 0 else 0
print(f"Total accuracy compared to actual data: {accuracy:.2f}%")

# Save all connections to a CSV file for evaluation
with open(csv_output_path, mode='w', newline='') as file:
fieldnames = ['Source_port', 'Destination_port', 'Protocol', 'Service', 'Attack_type', 'Subclass', 'Scenario',
              'Attack_vector', 'Total_data_sent', 'Packets_sent', 'Flow_duration', 'Payload_size_send',
              'Pkt_receive_interval', 'Send_frequency', 'Pkt_send_interval', 'Payload_size_receive', 'Receive_frequency',
              'Total_data_received', 'Packets_received', 'Status']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write header to CSV file

    # Write correct predictions to CSV
    for row in correct_predictions:
        writer.writerow(row)

    # Write discrepancies to CSV
    for row in discrepancies:
        writer.writerow(row)

# Final message to confirm that the CSV file was saved successfully
print(f"CSV with predicted connections saved to {csv_output_path}")
