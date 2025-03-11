# LCM Neural Network

This repository contains a neural network implementation of the Language-Concept Mapping (LCM) system. It extends the original LCM functionality with neural embeddings, trainable representations, and semantic operations.

## Overview

The LCM Neural Network (LCMNN) represents topics using trainable neural embeddings instead of random hypervectors. This enables:

1. **Trainable semantic relationships** - Topics can be trained to have specific distances between them
2. **Rich vector representations** - Neural embeddings capture complex semantic relationships
3. **Semantic operations** - Supports operations like analogical reasoning and concept combination
4. **Visualization capabilities** - Network visualization, embedding analysis, and distance distribution

## Files in this Repository

- `lcm_neural_network.py` - Core implementation of the LCMNN model
- `lcm_nn_example.py` - Example usage demonstrating key features
- `lcm_nn_visualizer.py` - Visualization utilities for the LCMNN
- `README.md` - This documentation file

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- NetworkX (for visualization)
- scikit-learn (for t-SNE visualization)
- pyvis (optional, for interactive visualization)

### Setup

```bash
pip install torch numpy matplotlib networkx scikit-learn pyvis
```

## Usage

### Basic Usage

```python
from lcm_neural_network import LCMNN

# Create sample topics
topics = ["apple", "banana", "orange", "computer", "keyboard", "mouse"]

# Initialize the neural LCM
lcm_nn = LCMNN(topics, dimension=128)

# Calculate distances between topics
distances, hypervectors = lcm_nn.calculate_distances()

# Get network representation
network = lcm_nn.simulate_node_network()

# Get information about a specific node
node_args = lcm_nn.get_node_arguments(0)  # Get info for the first topic
```

### Training Semantic Relationships

```python
# Define target distances between topic pairs
target_distances = {
    (0, 1): 0.2,  # apple and banana should be close
    (0, 3): 0.8,  # apple and computer should be far
    (3, 4): 0.1,  # computer and keyboard should be very close
}

# Train the embeddings
loss_history = lcm_nn.train_embeddings(target_distances, num_epochs=100)

# Recalculate distances after training
new_distances, _ = lcm_nn.calculate_distances()
```

### Visualization

```python
from lcm_nn_visualizer import visualize_network, visualize_embedding_heatmap

# Visualize the network
visualize_network(lcm_nn, threshold=0.5, layout='spring')

# Visualize embedding heatmap
visualize_embedding_heatmap(lcm_nn)

# Create interactive visualization (requires pyvis)
from lcm_nn_visualizer import create_interactive_plot
create_interactive_plot(lcm_nn, output_file="lcm_network.html")
```

## Key Features

### 1. Neural Embeddings

The LCMNN replaces random hypervectors with trainable neural embeddings. These embeddings start randomly initialized but can be trained to represent specific semantic relationships.

### 2. Binary and Bipolar Representations

The model supports both binary (0/1) and bipolar (-1/1) representations for topics, similar to the original LCM implementation:

```python
# Get binary representation for a topic
binary_vec = lcm_nn.get_binary_vector(topic_idx)

# Get bipolar representation for a topic
bipolar_vec = lcm_nn.get_bipolar_vector(topic_idx)
```

### 3. Distance Metrics

The model supports multiple distance metrics:

- Cosine distance (default)
- Hamming distance
- Euclidean distance

```python
# Initialize with custom distance type
lcm_nn = LCMNN(topics, dimension=128, distance_type="hamming")
```

### 4. Semantic Operations

The model supports semantic operations like:

- Analogical reasoning (A is to B as C is to ?)
- Concept combination (averaging embeddings)
- Semantic similarity measurement

See the `demonstrate_semantic_operations` function in `lcm_nn_example.py` for examples.

### 5. Network Simulation

```python
# Simulate the topic network
network = lcm_nn.simulate_node_network()

# Network contains:
# - 'nodes': List of nodes with topic info and vector samples
# - 'edges': List of edges with distance info and vector samples
```

## Extending the Model

### Custom Distance Functions

You can extend the `DistanceLayer` class to implement custom distance metrics:

```python
class CustomDistanceLayer(DistanceLayer):
    def forward(self, x1, x2):
        # Implement your custom distance metric
        return my_custom_distance(x1, x2)
```

### Custom Vector Representations

You can extend the transformation layers for custom vector representations:

```python
lcm_nn.to_custom = nn.Sequential(
    nn.Linear(dimension, dimension),
    nn.ReLU(),
    nn.Linear(dimension, dimension),
    YourCustomActivation()
)
```

## Differences from Original LCM

1. **Trainable Representations**: Unlike the original LCM which uses fixed random vectors, LCMNN uses trainable neural embeddings.

2. **Embedding Dimension**: LCMNN uses a more reasonable default embedding dimension (512 vs 10000).

3. **Distance Calculation**: LCMNN provides multiple distance metrics and can learn specific distance relationships.

4. **Semantic Operations**: Enhanced support for semantic operations like analogical reasoning.

5. **Integration with PyTorch**: Uses PyTorch for efficient computation and gradient-based optimization.

## Contributing

Contributions are welcome! Areas for improvement include:

- Additional distance metrics
- More sophisticated semantic operations
- Integration with external knowledge sources
- Optimization for large-scale topic networks

## License

MIT

## Acknowledgments

This implementation was inspired by the original LCM class and extends it with neural network capabilities.
