import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput, BertPreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# GraphBert model configuration
# I chose a lightweight configuration with only 4 hidden layers and 2 attention heads to limit RAM usage and ensure good performance
# Parameters max_wl_role_index, max_hop_dis_index, and max_inti_pos_index are set to handle specific node features in the graph
class GraphBertConfig(PretrainedConfig):
    def __init__(
        self,
        residual_type='none',  # Residual connection to facilitate gradient flow
        x_size=7,  # Number of node features
        y_size=12,  # Number of output classes or categories
        k=5,  # Number of neighbors per node used in attention
        max_wl_role_index=100,  # Maximum index for WL (Weisfeiler-Lehman) role embeddings
        max_hop_dis_index=100,  # Maximum index for hop distances
        max_inti_pos_index=100,  # Maximum index for intimate positions
        hidden_size=16,  # Latent space dimensionality
        num_hidden_layers=4,  # 4 hidden layers to balance depth and memory usage
        num_attention_heads=2,  # 2 attention heads to keep the model lightweight
        intermediate_size=16,  # Size of intermediate layers
        hidden_act="gelu",  # GELU activation function, suitable for BERT-like models
        hidden_dropout_prob=0.5,  # Dropout to control overfitting
        attention_probs_dropout_prob=0.3,  # Dropout in attention to improve generalization
        initializer_range=0.02,  # Weight initialization range
        layer_norm_eps=1e-12,  # Epsilon for layer normalization
        is_decoder=False,  # This model is not a decoder
        **kwargs
    ):
        super(GraphBertConfig, self).__init__(**kwargs)
        self.max_wl_role_index = max_wl_role_index
        self.max_hop_dis_index = max_hop_dis_index
        self.max_inti_pos_index = max_inti_pos_index
        self.residual_type = residual_type
        self.x_size = x_size
        self.y_size = y_size
        self.k = k
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder

# Definition of a single BertLayer
# Each layer applies attention on nodes with residual connections, allowing the model to retain information from previous layers
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)  # Attention on nodes
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        # Apply attention on nodes
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # If it's a decoder, add cross-attention
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        # Apply intermediate layer and output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

# Definition of a Bert encoder for feature propagation across model layers
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        # Create a list of layers as specified by the number of hidden layers
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, residual_h=None, attack_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        # For each layer in the model, pass hidden states through the layer and apply attention
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask)
            hidden_states = layer_outputs[0]

            # Increase attention on attack nodes, if present
            if attack_mask is not None:
                attack_weights = attack_mask.unsqueeze(-1)  # Extend the mask
                hidden_states = hidden_states * (1 + attack_weights * 0.1)

            # Add any residual information
            if residual_h is not None:
                for index in range(hidden_states.size()[1]):
                    hidden_states[:, index, :] += residual_h

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs

# Function to load and reduce node features
def load_and_reduce_features():
    # Load node features and adjacency matrix
    with open('/path/to/your/node_features.pkl', 'rb') as f:
        nodes_features = pickle.load(f)

    with open('/path/to/your/adjacency_matrix.pkl', 'rb') as f:
        adjacency_matrix = pickle.load(f)

    # Normalize features
    scaler = MinMaxScaler()
    nodes_features = scaler.fit_transform(nodes_features)

    hidden_size = 16  # Latent dimensionality, consistent with model config
    if nodes_features.shape[1] < hidden_size:
        # Add padding if necessary
        padding = np.zeros((nodes_features.shape[0], hidden_size - nodes_features.shape[1]))
        reduced_features = np.hstack((nodes_features, padding))
    else:
        reduced_features = nodes_features[:, :hidden_size]  # Resize features

    print(f"Reduced feature matrix shape: {reduced_features.shape}")

    return reduced_features, adjacency_matrix

# DatasetLoader class to load batches, hop values, and WL embeddings
class DatasetLoader:
    def load_hop_wl_batch(self):
        # Load saved WL embeddings, hop distances, and batches
        with open('/path/to/your/wl_embeddings.pkl', 'rb') as f:
            wl_dict = pickle.load(f)

        with open('/path/to/your/hop_distances.pkl', 'rb') as f:
            hop_dict = pickle.load(f)

        with open('/path/to/your/batch_data.pkl', 'rb') as f:
            batch_dict = pickle.load(f)

        return hop_dict, wl_dict, batch_dict

    def load(self):
        # Load hop, WL, and batch data
        hop_dict, wl_dict, batch_dict = self.load_hop_wl_batch()
        nodes_features, adjacency_matrix = load_and_reduce_features()

        raw_feature_list = []
        role_ids_list = []
        position_ids_list = []
        hop_ids_list = []
        attack_mask_list = []

        # For each node, build embeddings based on WL, hop, and neighbors
        for idx, node_features in enumerate(nodes_features):
            node_key = str(idx)
            neighbors_list = batch_dict.get(node_key, [])

            raw_feature = [node_features[:16]]
            role_ids = [wl_dict.get(node_key, 0)]
            position_ids = list(range(len(neighbors_list) + 1))  # Intimate positions
            hop_ids = [0]  # Hop distances

            for neighbor_key, _ in neighbors_list:
                neighbor_idx = int(neighbor_key)
                raw_feature.append(nodes_features[neighbor_idx][:16])
                role_ids.append(wl_dict.get(neighbor_key, 0))
                hop_ids.append(hop_dict.get(node_key, {}).get(neighbor_key, 99))  # Hop distance

            raw_feature = np.mean(raw_feature, axis=0)
            raw_feature_list.append(raw_feature)
            role_ids_list.append(role_ids)
            position_ids_list.append(position_ids)
            hop_ids_list.append(hop_ids)

            # Build mask to identify attack nodes
            if node_features[0] == -3:  # Attack node
                attack_mask_list.append(1)
            else:
                attack_mask_list.append(0)

        # Convert lists to numpy arrays and then to PyTorch tensors
        raw_feature_list = np.array(raw_feature_list, dtype=np.float32)
        role_ids_list = np.array(role_ids_list, dtype=np.float32)
        position_ids_list = np.array(position_ids_list, dtype=np.float32)
        hop_ids_list = np.array(hop_ids_list, dtype=np.float32)
        attack_mask_list = np.array(attack_mask_list, dtype=np.float32)

        raw_feature_list = np.expand_dims(raw_feature_list, axis=1)

        return {
            'X': torch.FloatTensor(nodes_features[:, :16]),
            'A': torch.FloatTensor(adjacency_matrix),
            'wl_embedding': torch.LongTensor(role_ids_list),
            'hop_embeddings': torch.LongTensor(hop_ids_list),
            'int_embeddings': torch.LongTensor(position_ids_list),
            'raw_embeddings': torch.FloatTensor(raw_feature_list),
            'attack_mask': torch.FloatTensor(attack_mask_list)  # Mask for attack nodes
        }
