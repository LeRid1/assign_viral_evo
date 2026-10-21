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

def get_mutation_rates(fasta_file: str, reference: str):
   
    """
    Input: fasta file containing aligned samples, and reference sequence 
    from database. 
    """

    # Extract dates and IDs into a list
    records = list(SeqIO.parse(fasta_file, "fasta"))
        # get date and IDs
    ids, dates = [], []
    for rec in records:
        # rec.description is e.g. "EPI_ISL_3834282|Australia|2021-08-21"
        epi_id, country, date = rec.description.split("|")
        ids.append(epi_id)
        dates.append(date)
    entry_df = pd.DataFrame({"id": ids, "date": pd.to_datetime(dates, format='mixed'), 
                            "seq": [str(r.seq).upper() for r in records]})

    # Earliest collection date
    start_date = entry_df["date"].min()
    entry_df["days"] = (entry_df["date"] - start_date).dt.days

    # Distance from the reference to each virus
    entry_df["dist_genome"] = [jukes_cantor(reference, s) for s in entry_df["seq"]]

    # Shift so the earliest sequence is at the origin (0, 0)
    first = entry_df.loc[entry_df["days"].idxmin()]
    entry_df["dist_genome"] = entry_df["dist_genome"] - entry_df.loc[entry_df["days"].idxmin(), "dist_genome"]
    entry_df = entry_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["dist_genome"])

    # Linear regression using np.polyfit (all points, including the earliest one)
    slope, intercept = np.polyfit(entry_df["days"], entry_df["dist_genome"], 1)

    per_genome_per_day = slope
    per_nt_per_day = slope / len(reference)

    return per_genome_per_day, per_nt_per_day, intercept, entry_df


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
