import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any


class LCMEmbedding(nn.Module):
    """
    Neural embedding layer for LCM topics
    """
    def __init__(self, num_topics: int, dimension: int = 512):
        """
        Initialize embedding layer for topics
        
        Args:
            num_topics: Number of topics to embed
            dimension: Dimension of embedding vectors (default: 512)
        """
        super(LCMEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_topics, dimension)
        # Initialize with values between -1 and 1 to mimic bipolar vectors
        nn.init.uniform_(self.embedding.weight, -1.0, 1.0)
    
    def forward(self, indices):
        """
        Get embeddings for topic indices
        """
        return self.embedding(indices)


class DistanceLayer(nn.Module):
    """
    Layer for computing distances between topic embeddings
    """
    def __init__(self, distance_type: str = "cosine"):
        """
        Initialize distance layer
        
        Args:
            distance_type: Type of distance to compute ("cosine", "hamming", "euclidean")
        """
        super(DistanceLayer, self).__init__()
        self.distance_type = distance_type
    
    def forward(self, x1, x2):
        """
        Compute distance between two embeddings
        """
        if self.distance_type == "cosine":
            # Cosine similarity (negative distance)
            return 1.0 - F.cosine_similarity(x1, x2, dim=-1)
        elif self.distance_type == "hamming":
            # Approximate Hamming distance for continuous values
            binary_x1 = torch.sign(x1)
            binary_x2 = torch.sign(x2)
            return torch.mean(torch.abs(binary_x1 - binary_x2), dim=-1) / 2.0
        else:  # euclidean
            return torch.sqrt(torch.sum((x1 - x2)**2, dim=-1) + 1e-8)


class LCMNN(nn.Module):
    """
    Neural Network implementation of Language-Concept Mapping (LCM)
    """
    def __init__(self, topics: List[Any], dimension: int = 512, distance_type: str = "cosine"):
        """
        Initialize the LCM Neural Network
        
        Args:
            topics: List of topics
            dimension: Dimension of embedding vectors (default: 512)
            distance_type: Type of distance to use (default: "cosine")
        """
        super(LCMNN, self).__init__()
        
        self.topics = topics
        self.topic_to_idx = {topic: idx for idx, topic in enumerate(topics)}
        self.dimension = dimension
        self.distance_type = distance_type
        
        # Create embedding layer for topics
        self.embedding = LCMEmbedding(len(topics), dimension)
        
        # Distance computation layer
        self.distance_layer = DistanceLayer(distance_type)
        
        # Mapping layer to convert embeddings to binary/bipolar representations
        self.to_binary = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Sigmoid()
        )
        
        self.to_bipolar = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh()
        )
        
        # Transformation layer for distance vectors
        self.distance_transform = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, dimension)
        )
        
        # Store computed distances and vectors
        self.distances = {}
        self.hypervectors = {}
        self.node_network = None
    
    def get_topics(self) -> List[Any]:
        """
        Return the list of topics
        """
        return self.topics
    
    def add_topic(self, topic: Any) -> None:
        """
        Add a new topic to the list
        """
        if topic not in self.topics:
            self.topics.append(topic)
            self.topic_to_idx[topic] = len(self.topics) - 1
            
            # Create new embedding for the added topic
            with torch.no_grad():
                old_weight = self.embedding.embedding.weight
                new_weight = torch.zeros(len(self.topics), self.dimension)
                new_weight[:old_weight.shape[0], :] = old_weight
                # Initialize the new embedding
                nn.init.uniform_(new_weight[old_weight.shape[0]:], -1.0, 1.0)
                
                # Create new embedding layer
                new_embedding = LCMEmbedding(len(self.topics), self.dimension)
                new_embedding.embedding.weight.data = new_weight
                self.embedding = new_embedding
    
    def remove_topic(self, topic: Any) -> None:
        """
        Remove a topic from the list if it exists
        """
        if topic in self.topics:
            idx = self.topics.index(topic)
            self.topics.remove(topic)
            
            # Update topic_to_idx
            self.topic_to_idx = {t: i for i, t in enumerate(self.topics)}
            
            # Create new embedding without the removed topic
            with torch.no_grad():
                old_weight = self.embedding.embedding.weight
                new_weight = torch.zeros(len(self.topics), self.dimension)
                
                # Copy all weights except the removed one
                j = 0
                for i in range(old_weight.shape[0]):
                    if i != idx:
                        new_weight[j] = old_weight[i]
                        j += 1
                
                # Create new embedding layer
                new_embedding = LCMEmbedding(len(self.topics), self.dimension)
                new_embedding.embedding.weight.data = new_weight
                self.embedding = new_embedding
            
            # Clear cached distances and vectors
            self.distances = {}
            self.hypervectors = {}
            self.node_network = None
    
    def has_topic(self, topic: Any) -> bool:
        """
        Check if a topic exists in the list
        """
        return topic in self.topics
    
    def get_topic_embedding(self, topic_idx: int) -> torch.Tensor:
        """
        Get the embedding for a topic by index
        """
        indices = torch.tensor([topic_idx])
        return self.embedding(indices)[0]
    
    def get_binary_vector(self, topic_idx: int) -> torch.Tensor:
        """
        Get binary representation for a topic
        """
        embedding = self.get_topic_embedding(topic_idx)
        return self.to_binary(embedding)
    
    def get_bipolar_vector(self, topic_idx: int) -> torch.Tensor:
        """
        Get bipolar representation for a topic
        """
        embedding = self.get_topic_embedding(topic_idx)
        return self.to_bipolar(embedding)
    
    def calculate_distances(self, vector_type: str = "bipolar") -> Tuple[Dict, Dict]:
        """
        Calculate distances between all pairs of topics
        
        Args:
            vector_type: Type of vector to use ("binary" or "bipolar")
        
        Returns:
            Tuple containing distances dictionary and hypervectors dictionary
        """
        with torch.no_grad():
            distances = {}
            hypervectors = {}
            
            # Get all topic indices
            indices = torch.arange(len(self.topics))
            
            # Get embeddings for all topics
            embeddings = self.embedding(indices)
            
            if vector_type == "binary":
                vectors = self.to_binary(embeddings)
            else:  # bipolar
                vectors = self.to_bipolar(embeddings)
            
            # Calculate distances between all pairs
            for i in range(len(self.topics)):
                for j in range(i + 1, len(self.topics)):
                    # Get vectors
                    vec_i = vectors[i]
                    vec_j = vectors[j]
                    
                    # Calculate distance
                    distance = self.distance_layer(vec_i.unsqueeze(0), vec_j.unsqueeze(0)).item()
                    distances[(i, j)] = distance
                    distances[(j, i)] = distance  # Symmetric
                    
                    # Create hypervector representing the distance
                    if vector_type == "binary":
                        # XOR operation for binary vectors
                        distance_vector = torch.logical_xor(
                            vec_i > 0.5, 
                            vec_j > 0.5
                        ).float()
                    else:  # bipolar
                        # Element-wise multiplication for bipolar vectors
                        distance_vector = vec_i * vec_j
                    
                    # Apply transformation
                    distance_vector = self.distance_transform(distance_vector)
                    
                    # Scale by distance (optional)
                    distance_vector = distance_vector * distance
                    
                    hypervectors[(i, j)] = distance_vector
                    hypervectors[(j, i)] = distance_vector  # Symmetric
            
            self.distances = distances
            self.hypervectors = hypervectors
            
            return distances, hypervectors
    
    def simulate_node_network(self) -> Dict:
        """
        Simulate the network of nodes with their connections
        
        Returns:
            Dictionary representing the network
        """
        # Calculate distances if not already done
        if not self.distances:
            self.calculate_distances()
        
        network = {
            'nodes': [],
            'edges': []
        }
        
        with torch.no_grad():
            # Get all topic indices
            indices = torch.arange(len(self.topics))
            
            # Get embeddings for all topics
            embeddings = self.embedding(indices)
            binary_vectors = self.to_binary(embeddings)
            bipolar_vectors = self.to_bipolar(embeddings)
            
            # Add nodes to the network
            for idx in range(len(self.topics)):
                network['nodes'].append({
                    'id': idx,
                    'topic': self.topics[idx],
                    'binary_vector_sample': binary_vectors[idx][:10].tolist(),
                    'bipolar_vector_sample': bipolar_vectors[idx][:10].tolist()
                })
            
            # Add edges to the network
            for (i, j), distance in self.distances.items():
                if i < j:  # Avoid duplicates
                    network['edges'].append({
                        'source': i,
                        'target': j,
                        'distance': distance,
                        'vector_sample': self.hypervectors[(i, j)][:10].tolist()
                    })
        
        self.node_network = network
        return network
    
    def hamming_distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Calculate Hamming distance between two vectors
        """
        with torch.no_grad():
            # Convert to binary
            binary_vec1 = (vec1 > 0).float()
            binary_vec2 = (vec2 > 0).float()
            return torch.sum(binary_vec1 != binary_vec2).item()
    
    def cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        with torch.no_grad():
            return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    
    def get_node_arguments(self, topic_idx: int) -> Optional[Dict]:
        """
        Get all arguments (values) attached to a node
        """
        if topic_idx >= len(self.topics):
            return None
        
        # Calculate distances if not already done
        if not self.distances:
            self.calculate_distances()
        
        with torch.no_grad():
            binary_vector = self.get_binary_vector(topic_idx)
            bipolar_vector = self.get_bipolar_vector(topic_idx)
            
            connections = {}
            for j in range(len(self.topics)):
                if j != topic_idx and (topic_idx, j) in self.distances:
                    connections[j] = {
                        'target_topic': self.topics[j],
                        'distance': self.distances[(topic_idx, j)],
                        'vector': self.hypervectors[(topic_idx, j)].tolist()
                    }
            
            return {
                'topic': self.topics[topic_idx],
                'index': topic_idx,
                'binary_vector': binary_vector.tolist(),
                'bipolar_vector': bipolar_vector.tolist(),
                'connections': connections
            }
    
    def forward(self, topic_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the network
        
        Args:
            topic_indices: Tensor of topic indices
        
        Returns:
            Dictionary containing embeddings and representations
        """
        embeddings = self.embedding(topic_indices)
        binary_vectors = self.to_binary(embeddings)
        bipolar_vectors = self.to_bipolar(embeddings)
        
        return {
            'embeddings': embeddings,
            'binary_vectors': binary_vectors,
            'bipolar_vectors': bipolar_vectors
        }
    
    def train_embeddings(self, 
                        target_distances: Dict[Tuple[int, int], float], 
                        num_epochs: int = 100, 
                        learning_rate: float = 0.001) -> List[float]:
        """
        Train the embeddings to match target distances
        
        Args:
            target_distances: Dictionary mapping topic pairs to target distances
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        
        Returns:
            List of losses during training
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_history = []
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            optimizer.zero_grad()
            
            for (i, j), target in target_distances.items():
                # Get embeddings
                i_embed = self.get_topic_embedding(i)
                j_embed = self.get_topic_embedding(j)
                
                # Calculate current distance
                current = self.distance_layer(i_embed.unsqueeze(0), j_embed.unsqueeze(0))
                
                # Loss is MSE between current and target distance
                target_tensor = torch.tensor([target], dtype=torch.float32)
                loss = F.mse_loss(current, target_tensor)
                
                total_loss += loss.item()
                # Accumulate gradients for this pair
                loss.backward(retain_graph=True)
            
            optimizer.step()
            loss_history.append(total_loss / len(target_distances))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_history[-1]:.6f}")
        
        # Recalculate distances after training
        self.calculate_distances()
        return loss_history


# Example usage
if __name__ == "__main__":
    # Create sample topics
    topics = ["apple", "banana", "orange", "computer", "keyboard", "mouse"]
    
    # Initialize the neural LCM
    lcm_nn = LCMNN(topics, dimension=128)
    
    # Calculate distances
    distances, hypervectors = lcm_nn.calculate_distances()
    
    # Get network representation
    network = lcm_nn.simulate_node_network()
    
    # Print some statistics
    print(f"Topics: {lcm_nn.get_topics()}")
    print(f"Number of nodes: {len(network['nodes'])}")
    print(f"Number of edges: {len(network['edges'])}")
    
    # Get node arguments for a topic
    node_args = lcm_nn.get_node_arguments(0)
    print(f"Node arguments for topic '{node_args['topic']}':")
    print(f"  Binary vector (sample): {node_args['binary_vector'][:5]}...")
    print(f"  Bipolar vector (sample): {node_args['bipolar_vector'][:5]}...")
    print(f"  Connections: {len(node_args['connections'])}")
    
    # Add a new topic
    lcm_nn.add_topic("strawberry")
    print(f"Topics after adding 'strawberry': {lcm_nn.get_topics()}")
    
    # Train embeddings with custom distances
    target_distances = {
        (0, 1): 0.2,  # apple and banana should be close
        (0, 3): 0.8,  # apple and computer should be far
        (3, 4): 0.1,  # computer and keyboard should be very close
    }
    loss_history = lcm_nn.train_embeddings(target_distances, num_epochs=50)
    
    # Recalculate distances after training
    new_distances, _ = lcm_nn.calculate_distances()
    
    # Check if the training made a difference
    for (i, j), target in target_distances.items():
        topic_i = lcm_nn.topics[i]
        topic_j = lcm_nn.topics[j]
        new_dist = new_distances[(i, j)]
        print(f"Distance between '{topic_i}' and '{topic_j}': {new_dist:.4f} (target: {target:.4f})")
