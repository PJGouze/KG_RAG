###--- Importing libraries ---####

import os
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

##################################

### --- Variables --- ############
SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

##################################

####- Defining the functions -####
def load_llm(model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'):
    """
    Loads a pretrained llm

    Args:
        model_name : input specified by the user (config file).
        If not specified : TinyLlama/TinyLlama-1.1B-Chat-v1.0
    Raises:
        ValueError: if the model is not found.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")


def vectorize_query(query, SentenceTransformer):
    """
    Creates an embedding of the query in order to find 
    the right entities in the knowledge graph

    Args:
        query : the prompt input by the user
    Raises:
        ValueError: If the graph is empty.
        OSError: If the folder cannot be created or accessed.
    """

    query_embedding = SentenceTransformer.encode(query)
    
    return query_embedding

def retrieve_context():
    pass