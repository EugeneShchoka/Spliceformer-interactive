import argparse
import copy
from pysam import VariantFile
from pyfastx import Fasta

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from model import SpliceFormer


def get_options():
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to input VCF file"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output VCF file"
    )
    parser.add_argument(
        "-r", "--reference", required=True, help="Path to reference FASTA file (hg19 or hg38)"
    )

    args = parser.parse_args()

    return args


def load_model(CL_max, device):
    NUM_ACCUMULATION_STEPS = 1

    n_models = 10
    model_m = SpliceFormer(CL_max, bn_momentum=0.01 / NUM_ACCUMULATION_STEPS, depth=4, heads=4, n_transformer_blocks=2,
                           determenistic=True, crop=False)
    model_m = model_m.to(device)

    models = [copy.deepcopy(model_m) for i in range(n_models)]

    # This for loop is necessary when loading the weights to a single GPU
    for i, model in enumerate(models):
        state_dict = torch.load(
            './PyTorch_Models/transformer_encoder_40k_finetune_rnasplice-blood_all_050623_{}'.format(i),
            map_location=device
        )
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    for model in models:
        model.eval()

    return models


def one_hot_encode(sequence):
    # Define the encoding map
    encoding_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }

    # Initialize an array to hold the one-hot encoded sequence
    one_hot_array = np.zeros((len(sequence), 4), dtype=int)

    # Populate the one-hot encoded array
    for i, nucleotide in enumerate(sequence):
        one_hot_array[i] = encoding_map.get(nucleotide, [0, 0, 0, 0])

    return one_hot_array


def get_deltas(ref_prediction, alt_prediction, pos_s, crop, ref_len, alt_len, ref_seq_len, alt_seq_len):
    """

    Args:
      ref_prediction: Splice site scores for all nucleotides in the reference sequence
      alt_prediction: Splice site scores for all nucleotides in the alternative sequence
      pos_s: Variant position minus sequence start position
      crop: Region to crop from both sides of the delta tracks

    Returns: Donor and acceptor delta tracks (difference between alt_prediction and ref_prediction)

    """
    ref_acceptor = ref_prediction[1, :]
    alt_acceptor = alt_prediction[1, :]
    ref_donor = ref_prediction[2, :]
    alt_donor = alt_prediction[2, :]

    delta_1_a = alt_acceptor[:pos_s] - ref_acceptor[:pos_s]
    delta_1_d = alt_donor[:pos_s] - ref_donor[:pos_s]
    delta_3_a = alt_acceptor[pos_s + alt_len:] - ref_acceptor[pos_s + ref_len:]
    delta_3_d = alt_donor[pos_s + alt_len:] - ref_donor[pos_s + ref_len:]
    if ref_seq_len == alt_seq_len:
        delta_2_a = alt_acceptor[pos_s:pos_s + ref_len] - ref_acceptor[pos_s:pos_s + ref_len]
        delta_2_d = alt_donor[pos_s:pos_s + ref_len] - ref_donor[pos_s:pos_s + ref_len]
    elif ref_seq_len > alt_seq_len:
        a_pad = np.pad(alt_acceptor[pos_s:pos_s + alt_len], (0, ref_len - alt_len), 'constant', constant_values=0)
        d_pad = np.pad(alt_donor[pos_s:pos_s + alt_len], (0, ref_len - alt_len), 'constant', constant_values=0)
        delta_2_a = a_pad - ref_acceptor[pos_s:pos_s + ref_len]
        delta_2_d = d_pad - ref_donor[pos_s:pos_s + ref_len]

    elif ref_seq_len < alt_seq_len:
        a_pad = np.pad(ref_acceptor[pos_s:pos_s + ref_len], (0, alt_len - ref_len), 'constant', constant_values=0)
        d_pad = np.pad(ref_donor[pos_s:pos_s + ref_len], (0, alt_len - ref_len), 'constant', constant_values=0)
        delta_2_a = alt_acceptor[pos_s:pos_s + alt_len] - a_pad
        delta_2_d = alt_donor[pos_s:pos_s + alt_len] - d_pad

        delta_2_a = np.append(delta_2_a[:ref_len - 1],
                              delta_2_a[np.argmax(np.absolute(delta_2_a[ref_len - 1:alt_len]))])
        delta_2_d = np.append(delta_2_d[:ref_len - 1],
                              delta_2_d[np.argmax(np.absolute(delta_2_d[ref_len - 1:alt_len]))])

    acceptorDelta = np.concatenate([delta_1_a, delta_2_a, delta_3_a])
    donorDelta = np.concatenate([delta_1_d, delta_2_d, delta_3_d])
    return acceptorDelta[crop:-crop], donorDelta[crop:-crop]


def main():
    args = get_options()

    SL = 5000
    CL_max = 40000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = load_model(CL_max, device)

    with VariantFile(args.input, "r") as vcf:
        fasta = Fasta(args.reference)
        with VariantFile(args.output, "w", header=vcf.header) as out:
            for record in vcf:
                chrom = record.chrom if record.chrom.startswith("chr") else f"chr{record.chrom}"
                pos = record.pos
                ref = record.ref
                alt = [alt for alt in record.alts]

                ref_fa = fasta[chrom][pos - 1:pos - 1 + len(ref)].seq

                try:
                    assert ref == ref_fa
                except ValueError:
                    raise ValueError(
                        f"Reference base mismatch at {chrom}:{pos} (expected {ref}, got {fasta.fetch(chrom, pos - 1, pos)})"
                    )

                start, end = pos - SL // 2 - CL_max // 2, pos + SL // 2 + CL_max // 2
                pos_start = pos - start
                ref_seq = fasta[chrom][start - 1:end - 1].seq.upper()
                ref_len = len(ref)

                ref_seq_len = len(ref_seq)

                ref_seq_encoded = one_hot_encode(ref_seq)

                ref_seq_tensor = torch.tensor(ref_seq_encoded, dtype=torch.float32).T.unsqueeze(0).to(device)

                ref_prediction = torch.stack([model(ref_seq_tensor)[0].detach() for model in models]).mean(
                    dim=0).cpu().numpy()[0, :, :]

                alt_number = len(alt)
                for i in range(alt_number):
                    alt_seq = ref_seq[:pos_start] + alt[i] + ref_seq[(pos_start + ref_len):]

                    alt_len = len(alt[i])

                    alt_seq_len = len(alt_seq)

                    alt_seq_encoded = one_hot_encode(alt_seq)

                    alt_seq_tensor = torch.tensor(alt_seq_encoded, dtype=torch.float32).T.unsqueeze(0).to(device)

                    alt_prediction = torch.stack([model(alt_seq_tensor)[0].detach() for model in models]).mean(
                        dim=0).cpu().numpy()[0, :, :]

                    #
                    acceptor_delta, donor_delta = get_deltas(ref_prediction, alt_prediction, pos_start, CL_max // 2,
                                                             ref_len, alt_len, ref_seq_len, alt_seq_len)

                    delta_score = np.max(np.concatenate([acceptor_delta, donor_delta], axis=0))


if __name__ == '__main__':
    main()
