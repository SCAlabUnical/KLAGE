import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from transformers import BertConfig, BertModel
import random

# Set seed to generate random numbers for reproducibility
seed = 42
torch.manual_seed(seed)  # Set seed for PyTorch
torch.cuda.manual_seed_all(seed)  # Set seed for all CUDA devices
np.random.seed(seed)  # Set seed for numpy
random.seed(seed)  # Set seed for the random module

# Specify paths to pickle files containing node features and adjacency matrix
nodes_features_path = '/path/to/your/nodes_features.pkl'
adjacency_matrix_path = '/path/to/your/adjacency_matrix.pkl'
best_model_path = '/path/to/save/best_fine_tuned_model.pth'

# Load node feature data from pickle file
with open(nodes_features_path, 'rb') as f:
    nodes_features = pickle.load(f)

# Load adjacency matrix from pickle file
with open(adjacency_matrix_path, 'rb') as f:
    adjacency_matrix = pickle.load(f)

# Mapping of attack type to respective numeric labels
attack_type_mapping = {
    0: "ARP_poisioning", 1: "DDOS_Slowloris", 2: "DOS_SYN_Hping", 3: "MQTT_Publish",
    4: "NMAP_OS_DETECTION", 5: "NMAP_TCP_scan", 6: "NMAP_XMAS_TREE_SCAN", 7: "Thing_Speak", 8: "Wipro_bulb"
}

# Define the GraphBert model for fine-tuning
class GraphBertForFineTuning(nn.Module):
    def __init__(self, config):
        super(GraphBertForFineTuning, self).__init__()
        # Initialize the BERT model with the specified configuration
        self.bert_encoder = BertModel(config)
        # Linear layer to project node embeddings from 13 dimensions to BERT's hidden size
        self.projection = nn.Linear(13, config.hidden_size)
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=0.1)
        # Final classifier to predict the type of attack
        self.classifier = nn.Linear(config.hidden_size, len(attack_type_mapping))
        # self.batch_norm = nn.BatchNorm1d(config.hidden_size)  # Optional: batch normalization (commented out)

    def forward(self, sp_embeds, dp_embeds, at_embeds, attention_mask_sp, attention_mask_dp, attention_mask_at):
        # Project the input embeddings to the required BERT dimension
        sp_embeds = self.projection(sp_embeds)
        dp_embeds = self.projection(dp_embeds)
        at_embeds = self.projection(at_embeds)

        # Get the output from the BERT model for each input embedding
        sp_output = self.bert_encoder(inputs_embeds=sp_embeds, attention_mask=attention_mask_sp).last_hidden_state[:, 0, :]
        dp_output = self.bert_encoder(inputs_embeds=dp_embeds, attention_mask=attention_mask_dp).last_hidden_state[:, 0, :]
        at_output = self.bert_encoder(inputs_embeds=at_embeds, attention_mask=attention_mask_at).last_hidden_state[:, 0, :]

        # Sum the three outputs to combine the information
        combined_output = sp_output + dp_output + at_output
        # combined_output = self.batch_norm(combined_output)  # Optional batch normalization, commented out
        combined_output = self.dropout(combined_output)  # Apply dropout for regularization

        # Pass the combined output through the classifier to get final predictions
        logits = self.classifier(combined_output)

        return logits

# Create a list to store connections: at --> sp --> dp
connections = []
# For each node identified as an attack (at), check its connection with source (sp) and destination (dp) nodes
for at_id, at_features in enumerate(nodes_features):
    if at_features[0] == -3:  # Check if it is an attack node
        for sp_id, is_attached in enumerate(adjacency_matrix[at_id]):
            if is_attached and nodes_features[sp_id][0] == -1:  # Check connection with source node
                for dp_id, is_connected in enumerate(adjacency_matrix[sp_id]):
                    if is_connected and nodes_features[dp_id][0] == -2:  # Check connection with destination node
                        # Add the found connection to the list
                        connections.append((at_id, sp_id, dp_id))

# Lists to store embeddings and labels for training
X_sp_train = []
X_dp_train = []
X_at_train = []
attack_labels = []

# Process each connection and create the training dataset
for at_id, sp_id, dp_id in connections:
    # Extract the first 13 features of each node
    X_sp_train.append(nodes_features[sp_id][:13])
    X_dp_train.append(nodes_features[dp_id][:13])
    X_at_train.append(nodes_features[at_id][:13])
    # Add the label for the attack type
    attack_labels.append(int(nodes_features[at_id][1]))  # Attack type is in the second feature

# Convert data to PyTorch tensors
X_sp_train = torch.FloatTensor(X_sp_train).unsqueeze(1)  # Add one dimension for batch
X_dp_train = torch.FloatTensor(X_dp_train).unsqueeze(1)
X_at_train = torch.FloatTensor(X_at_train).unsqueeze(1)
attack_labels = torch.LongTensor(attack_labels)  # Convert labels to LongTensor for classification

# Create attention masks required by BERT to handle variable input lengths
attention_mask_sp = torch.ones(X_sp_train.shape[:2])
attention_mask_dp = torch.ones(X_dp_train.shape[:2])
attention_mask_at = torch.ones(X_at_train.shape[:2])

# Check tensor dimensions to ensure they are correct
print("X_sp_train shape:", X_sp_train.shape)
print("X_dp_train shape:", X_dp_train.shape)
print("X_at_train shape:", X_at_train.shape)
print("attack_labels shape:", attack_labels.shape)

# Configure BERT with reduced hidden size (13) and only 1 attention head for simplicity
config = BertConfig(hidden_size=13, num_attention_heads=1)
model = GraphBertForFineTuning(config)

# Set training parameters
epochs = 3000  # Total number of training epochs
optimizer = optim.Adam(model.parameters(), lr=5e-5)  # Adam optimizer with initial learning rate of 5e-5
criterion = nn.CrossEntropyLoss()  # Loss function for classification

# Define a scheduler to reduce learning rate on loss plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)

# Set early stopping parameters to stop training when no improvement is observed
best_accuracy = 0  # Store best validation accuracy
patience = 200  # Max epochs without improvement before stopping
stopping_counter = 0  # Counter for epochs without improvement

# Training loop for the specified number of epochs
for epoch in range(epochs):
    model.train()  # Set model to training mode
    # Compute model predictions on training data
    outputs = model(X_sp_train, X_dp_train, X_at_train, attention_mask_sp, attention_mask_dp, attention_mask_at)
    loss = criterion(outputs, attack_labels)  # Compute loss against ground truth labels
    optimizer.zero_grad()  # Reset accumulated gradients
    loss.backward()  # Backpropagation to compute gradients
    optimizer.step()  # Update model weights

    # Evaluate the model every 10 epochs
    if epoch % 10 == 0:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation to save memory
            predicted_classes = torch.argmax(outputs, dim=1)  # Get predicted class with highest score
            correct_predictions = (predicted_classes == attack_labels).sum().item()  # Count correct predictions
            total_predictions = len(attack_labels)  # Total number of predictions
            accuracy = correct_predictions / total_predictions * 100  # Compute accuracy in percent

        # Print partial results
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Accuracy: {accuracy:.2f}%')

        # Check for early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy  # Update best accuracy
            stopping_counter = 0  # Reset early stopping counter
            # Save model state if it reached a new best accuracy
            torch.save(model.state_dict(), best_model_path)
        else:
            stopping_counter += 1  # Increase counter if no improvement

        # Update learning rate using scheduler if loss doesn't improve
        scheduler.step(loss)

        # If early stopping counter reaches patience value, stop training
        if stopping_counter >= patience:
            print(f"Early stopping: No improvement after {patience} epochs, stopping training.")
            break

# Save the fine-tuned model after training
torch.save(model.state_dict(), best_model_path)

print(f"Fine-tuning completed. Best Val Accuracy: {best_accuracy:.2f}%")
