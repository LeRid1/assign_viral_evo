import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO, Entrez

def jukes_cantor(reference_sequence: str, distant_sequence: str) -> float:
    """The Jukes-Cantor correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.

    Parameters
    ----------
    reference_sequence: str
        A string of nucleotides in a sequence used as a reference
        in an alignment with other (e.g. AGGT-GA)
    distant_sequence: str
        A string of nucleotides in a sequence after the alignment
        with a reference (e.g. AGC-AGA)

    Returns
    -------
    float
        The Jukes-Cantor corrected genetic distance using Hamming distance.
        For example 1.163.

    """
    # Remove indels (positions where either sequence has a '-')
    # We only consider positions where both have a nucleotide
    ref_clean = []
    dist_clean = []
    for r, d in zip(reference_sequence, distant_sequence):
        if r != '-' and d != '-':
            ref_clean.append(r)
            dist_clean.append(d)
            
    L = len(ref_clean)
    if L == 0:
        return 0.0
    
    # Calculate Hamming distance (number of mismatches)
    diffs = sum(1 for r, d in zip(ref_clean, dist_clean) if r != d)
    
    # Fraction of different sites
    p = diffs / L
    
    # JC formula: d = -3/4 * ln(1 - 4/3 * p)
    # The example implies this is the distance per site.
    # To get total distance, we multiply by L.
    
    if p >= 0.75:
        # Distance is undefined or infinite in JC if p >= 0.75
        return float('inf')
        
    # Standard per-site JC distance
    d_per_site = -0.75 * np.log(1 - (4/3) * p)
    
    # Return total distance
    return d_per_site * L


def kimura_two_parameter(reference_sequence: str, distant_sequence: str) -> float:
    """The Kimura Two Parameter correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.

    Parameters
    ----------
    reference_sequence: str
        A string of nucleotides in a sequence used as a reference
        in an alignment with other (e.g. AGGT-GA)
    distant_sequence: str
        A string of nucleotides in a sequence after the alignment
        with a reference (e.g. AGC-AGA)

    Returns
    -------
    float
        The Kimura corrected genetic distance using Hamming distance.
        For example 1.196.

    """
    raise NotImplementedError()
