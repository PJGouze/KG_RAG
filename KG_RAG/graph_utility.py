###--- Importing libraries ---####

import os
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from sentence_transformers import SentenceTransformer


##################################


###- Defining the functions -###

def create_graph():
    pass

def save_graph(graph, folder_name, file_name="graphe.pkl"):
    """
    Saves a NetworkX graph as a .pkl file in the specified folder.

    Args:
        graph (nx.Graph): The graph to save.
        folder_name (str): Name of the folder (e.g., "Data").
        file_name (str, optional): Name of the file. Defaults to "graphe.pkl".

    Raises:
        ValueError: If the graph is empty.
        OSError: If the folder cannot be created or accessed.
    """
    try:
        # Check if the graph is valid
        if not graph.nodes():
            raise ValueError("The graph is empty. No data to save.")

        # Create the folder if it doesn't exist
        os.makedirs(folder_name, exist_ok=True)

        # Save the graph
        path = os.path.join(folder_name, file_name)
        nx.write_gpickle(graph, path)

        print(f"Graph saved in: {path}")
        return True

    except Exception as e:
        print(f"Error while saving: {e}")
        return False



def load_graph(file_name, folder_name="Data"):
    """
    Loads a NetworkX graph from a .pkl file in the specified folder.

    Args:
        file_name (str): Name of the file (e.g., "medical_graph.pkl").
        folder_name (str, optional): Name of the folder. Defaults to "Data".

    Returns:
        nx.Graph: The loaded graph if successful, None otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the folder cannot be accessed.
        ValueError: If the loaded data is not a valid graph.
    """
    try:
        # Create the full path safely
        path = Path(folder_name) / file_name

        # Check if the file exists
        if not path.exists():
            raise FileNotFoundError(f"The file '{path}' was not found in the '{folder_name}' folder.")

        # Load the graph
        G = nx.read_gpickle(str(path))

        # Validate the loaded graph
        if not isinstance(G, nx.Graph):
            raise ValueError(f"The loaded data is not a valid NetworkX graph. Got: {type(G)}")

        print(f"Graph loaded successfully from: {path}")
        return G

    except FileNotFoundError as fnf:
        print(f"File not found: {fnf}")
        return None

    except ValueError as ve:
        print(f"Invalid graph data: {ve}")
        return None

    except Exception as e:
        print(f"Unexpected error while loading: {e}")
        return None


def view_graph(graph, title="Graph Visualization", figsize=(14, 10), seed=42):
    """
    Visualizes a NetworkX graph with customizable settings.

    Args:
        graph (nx.Graph): The graph to visualize.
        title (str, optional): Title of the figure. Defaults to "Graph Visualization".
        figsize (tuple, optional): Figure size. Defaults to (14, 10).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        matplotlib.figure.Figure: The figure with the graph visualization.

    Notes:
        - Uses 'salmon' color for nodes and 'gray' for edges.
        - Node size is set to 3000.
        - Edge labels are extracted from the 'relation' attribute.
        - For undirected graphs, set `nx.Graph()` in the function call.
    """
    try:
        # Set up the figure
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=14)

        # Use a random seed for reproducibility
        pos = nx.spring_layout(graph, k=0.5, seed=seed)

        # Draw nodes and edges
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=3000,
            node_color="salmon",
            font_size=10,
            font_weight="bold",
            edge_color="gray",
            connectionstyle="arc3,rad=0.1",
            arrowsize=20
        )

        # Extract and draw edge labels
        edge_labels = nx.get_edge_attributes(graph, 'relation')
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5
        )

        print(f"Graph visualized successfully: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        return plt.gcf()

    except Exception as e:
        print(f"Error while visualizing: {e}")
        return None


def generate_graph_embeddings(
    graph,
    folder="Data/Embeddings",
    file_prefix="nodes",
    node_attributes=["type", "severity", "prevalence", "description"],
    model_name="all-MiniLM-L6-v2"
):
    """
    Generates sentence embeddings for each node in a NetworkX graph and saves the results.

    Args:
        graph (nx.Graph): The networkX graph to process.
        folder (str, optional): Output directory. Defaults to "Data/Embeddings".
        file_prefix (str, optional): Prefix for output files. Defaults to "nodes".
        node_attributes (list, optional): List of node attributes to include in the description.
                                         Defaults to ["type", "severity", "prevalence", "description"].
        model_name (str, optional): Name of the SentenceTransformer model. Defaults to "all-MiniLM-L6-v2".

    Returns:
        tuple: (embeddings, node_ids) if successful, (None, None) otherwise.

    Raises:
        ValueError: If the graph is empty or node_attributes list is invalid.
        OSError: If the folder cannot be created or accessed.
    """
    try:
        # Validate inputs
        if not graph.nodes():
            raise ValueError("Graph is empty. No data to process.")
        if not isinstance(node_attributes, list):
            raise ValueError("node_attributes must be a list of strings.")

        # Create output directory
        os.makedirs(folder, exist_ok=True)

        # Extract descriptions for each node
        descriptions = []
        for node, attr in graph.nodes(data=True):
            description_parts = []
            for attr_key in node_attributes:
                if attr_key in attr:
                    description_parts.append(f"{attr_key.replace('_', ' ').title()}: {attr[attr_key]}")
                elif attr_key == "node":
                    description_parts.append(f"Node: {node}")

            description = " ".join(description_parts)
            descriptions.append(description)

        # Load the embedding model
        print(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)

        # Generate embeddings
        print(f"Encoding {len(descriptions)} node descriptions...")
        embeddings = model.encode(descriptions, convert_to_tensor=True)

        # Display success info
        print(f"Embeddings generated for {len(descriptions)} nodes.")
        print(f"Embedding dimension: {embeddings.shape[1]}")

        # Convert to numpy and save
        embeddings_np = embeddings.cpu().numpy()
        embedding_path = os.path.join(folder, f"{file_prefix}_embeddings.npy")
        np.save(embedding_path, embeddings_np)

        # Save node IDs
        node_ids = list(graph.nodes())
        ids_path = os.path.join(folder, f"{file_prefix}_node_ids.txt")

        with open(ids_path, "w", encoding="utf-8") as f:
            f.write("\n".join(node_ids))

        print(f"Files saved in: {folder}")
        return embeddings_np, node_ids

    except ValueError as ve:
        print(f"Input validation error: {ve}")
        return None, None

    except OSError as oserr:
        print(f"OS-level error (e.g., folder access): {oserr}")
        return None, None

    except Exception as e:
        print(f"Unexpected error during embeddings generation: {e}")
        return None, None
