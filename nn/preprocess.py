# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    positives = [seqs[i] for i in range(len(seqs)) if labels[i]]
    negatives = [seqs[i] for i in range(len(seqs)) if not labels[i]]
    num_samples_per_class = min(len(positives), len(negatives))
    positives = list(np.random.choice(positives, num_samples_per_class, replace = False))
    negatives = list(np.random.choice(negatives, num_samples_per_class, replace = False))
    sampled_labels = [True for i in range(num_samples_per_class)]
    sampled_labels.extend([False for i in range(num_samples_per_class)])
    positives.extend(negatives)
    min_len_sequence = min([len(i) for i in positives])
    fixed_len_indeces = [random.randint(0, len(i) - min_len_sequence) for i in positives]
    positives = [positives[i][fixed_len_indeces[i]:fixed_len_indeces[i] + min_len_sequence] for i in range(len(positives))]
    return (positives, sampled_labels)

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    conversion = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "C": [0, 0, 1, 0], "G": [0, 0, 0, 1]}
    ohe_seqs = []
    for i in range(len(seq_arr)):
        ohe_seq_arr = []
        for j in seq_arr[i]:
            ohe_seq_arr.extend(conversion[j.upper()])
        ohe_seqs.append(ohe_seq_arr)
    return ohe_seqs