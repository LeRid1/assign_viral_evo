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
    earliest_idx = entry_df["days"].idxmin()
    entry_df["dist_genome"] = entry_df["dist_genome"] - entry_df.loc[earliest_idx, "dist_genome"]
    entry_df = entry_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["dist_genome"])

    # First regression: all points, including the earliest one
    slope, intercept = np.polyfit(entry_df["days"], entry_df["dist_genome"], 1)

    # If the intercept is too large, drop the earliest (origin) sample and refit
    if intercept > 5.0:
        fit_df = entry_df.drop(index=earliest_idx, errors="ignore")
        slope, intercept = np.polyfit(fit_df["days"], fit_df["dist_genome"], 1)


    per_genome_per_day = slope
    per_nt_per_day = slope / len(reference)

    return per_genome_per_day, per_nt_per_day, intercept, entry_df

def compute_mutation_frequencies(fasta_file, alpha_ids, delta_ids, ref_id):
    """
    Compute per-position mutation frequencies (0 to 1) for the Alpha and Delta
    variants relative to the NC_045512.2 reference.

    Returns
    -------
    alpha_freq, delta_freq : np.ndarray
        Arrays with the fraction of sequences differing from the reference
        at each alignment position.
    """
    # Load all sequences keyed by record ID (uppercase)
    records = {rec.id: str(rec.seq).upper()
               for rec in SeqIO.parse(fasta_file, "fasta")}

    def find_key(seq_id):
        # Exact match first, then fall back to headers that contain the ID
        if seq_id in records:
            return seq_id
        for k in records:
            if seq_id in k:
                return k
        raise KeyError(f"Sequence {seq_id} not found in {fasta_path}")

    ref = np.array(list(records[find_key(ref_id)]))

    def variant_frequency(ids):
        mismatches = []
        for sid in ids:
            seq = np.array(list(records[find_key(sid)]))
            if len(seq) != len(ref):
                raise ValueError(f"{sid} length differs from reference (not aligned?)")
            mismatches.append(seq != ref)
        return np.mean(mismatches, axis=0)

    return variant_frequency(alpha_ids), variant_frequency(delta_ids)

def plot_mutation_frequencies(alpha_freq, delta_freq, gene_locations,
                              start=20000):
    """
    Plot mutation frequencies for Alpha and Delta (two stacked subplots),
    showing only the genome region after `start` and marking gene locations.
    """
    gene_colors = plt.cm.tab10(np.linspace(0, 1, len(gene_locations)))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

    for ax, freq, name in zip(axes, (alpha_freq, delta_freq), ("Alpha", "Delta")):
        freq = np.asarray(freq)
        x = np.arange(start, len(freq))
        ax.plot(x, freq[start:], color="black", linewidth=0.8)

        # Shade and label each gene
        for (gene, (g_start, g_end)), color in zip(gene_locations.items(), gene_colors):
            ax.axvspan(g_start, g_end, color=color, alpha=0.25, label=gene)
            ax.text((g_start + g_end) / 2, 1.03, gene, ha="center", va="bottom",
                    color=color, fontweight="bold",
                    transform=ax.get_xaxis_transform())

        ax.set_xlim(start, len(freq))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Average mutation occurrence")
        ax.set_title(f"{name} variant", pad=16)
        ax.legend(title="Genes", loc="upper left", ncol=len(gene_locations),
                  fontsize=8)

    axes[-1].set_xlabel("Sequence index (nucleotide position)")
    fig.tight_layout()

    fig.savefig("problem3.svg", dpi=300)
    plt.show()

from collections import Counter

def find_shared_s_mutations(alpha_freq, delta_freq, fasta_path,
                            alpha_variants, delta_variants,
                            reference_id,
                            gene_locations, threshold=0.5):
    """
    Find sites in the S gene mutated (frequency > threshold) in BOTH variants.

    Returns
    -------
    mutations_in_both_variants : list[int]
        1-based reference positions where both variants have frequency > threshold.
    same_mutation : list[str]
        Subset of those sites where both variants carry the SAME new nucleotide,
        formatted like "G123A" (reference base, 1-based site, variant base).
    """
    alpha_freq, delta_freq = np.asarray(alpha_freq), np.asarray(delta_freq)
    records = list(SeqIO.parse(fasta_path, "fasta"))

    def match(ids):
        # exact ID match first, then fallback to substring match in longer headers
        return [str(r.seq).upper() for r in records
                if r.id in ids or any(v in r.description for v in ids)]

    ref_seq = next(str(r.seq).upper() for r in records
                   if r.id == reference_id or reference_id in r.description)
    alpha_seqs, delta_seqs = match(alpha_variants), match(delta_variants)

    def consensus(seqs, i):
        return Counter(s[i] for s in seqs).most_common(1)[0][0]

    # S gene coordinates are treated as 1-based and inclusive
    s_start, s_end = gene_locations['S']
    s_idx = np.arange(s_start - 1, s_end)

    both = s_idx[(alpha_freq[s_idx] > threshold) & (delta_freq[s_idx] > threshold)]

    mutations_in_both_variants = [int(i) + 1 for i in both]   # 1-based sites

    same_mutation = []
    for i in both:
        ref_nt = ref_seq[i]
        a_nt, d_nt = consensus(alpha_seqs, i), consensus(delta_seqs, i)
        if a_nt == d_nt and a_nt != ref_nt and a_nt in "ACGT":
            same_mutation.append(f"{ref_nt}{int(i) + 1}{a_nt}")

    return mutations_in_both_variants, same_mutation

import csv
import json

def get_reference_spike(entrez_email = "z540752@ad.unsw.edu.au"):
    """Download NC_045512.2 and return the translated spike (S) protein."""
    Entrez.email = entrez_email
    handle = Entrez.efetch(db="nucleotide", id="NC_045512.2",
                           rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()
    for feat in record.features:
        if feat.type == "CDS" and feat.qualifiers.get("gene", [""])[0] == "S":
            return feat.qualifiers["translation"][0]
    raise ValueError("Spike CDS not found in reference record")


def classify_spikes(fasta_path="data/p5-aus-spike-proteins.fasta",
                    variants_path="data/variants.json",
                    out_path="problem5-classification.csv",
                    threshold=0.51):
    ref = get_reference_spike()

    with open(variants_path) as f:
        variants = json.load(f)

    # Keep only spike nonsynonymous mutations for each variant
    variant_muts = {}
    for name, info in variants.items():
        muts = [(m["pos"], m["right"])
                for m in info.get("nonsynonymous", [])
                if m.get("gene") == "S"]
        if muts:
            variant_muts[name] = muts

    rows = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        accession = rec.description.split("|")[-1].strip()
        seq = str(rec.seq).upper()

        # Any mutation at all vs. the reference? (ignore X; '-' = deletion; ignore insertions).
        has_mutation = any(
            a != "X" and a != r
            for a, r in zip(seq, ref)
        )
        if not has_mutation:
            rows.append((accession, "NO_MUTATIONS"))
            continue

        # Percentage of each variant's mutations present in the sequence
        best_name, best_frac = "UNKNOWN", 0.0
        for name, muts in variant_muts.items():
            matches = sum(
                1 for pos, right in muts
                if pos <= len(seq) and seq[pos - 1] == right  # 1-based -> 0-based
            )
            frac = matches / len(muts)
            if frac > best_frac:
                best_name, best_frac = name, frac

        label = best_name if best_frac >= threshold else "UNKNOWN"
        rows.append((accession, label))

    # Write CSV with no header
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    return rows


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
