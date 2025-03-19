import argparse
import copy
import pysam

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
        "-r", "--reference", required=True, help="Path to reference FASTA file"
    )

    args = parser.parse_args()

    return args


def load_model(CL_max):
    NUM_ACCUMULATION_STEPS = 1

    n_models = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_m = SpliceFormer(CL_max, bn_momentum=0.01 / NUM_ACCUMULATION_STEPS, depth=4, heads=4, n_transformer_blocks=2,
                           determenistic=True, crop=False)
    model_m = model_m.to(device)

    models = [copy.deepcopy(model_m) for i in range(n_models)]

    # This for loop is necessary when loading the weights to a single GPU
    for i, model in enumerate(models):
        state_dict = torch.load(
            '../PyTorch_Models/transformer_encoder_40k_finetune_rnasplice-blood_all_050623_{}'.format(i))
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


def main():
    args = get_options()

    SL = 5000
    CL_max = 40000

    models = load_model(CL_max)

    with pysam.VariantFile(args.input, "r") as vcf:
        with pysam.FastaFile(args.reference) as fasta:
            with pysam.VariantFile(args.output, "w", header=vcf.header) as out:
                for record in vcf:

                    chrom = record.chrom
                    pos = record.pos
                    ref = record.ref
                    alt = record.alts # ???

                    try:
                        assert ref == fasta.fetch(chrom, pos - 1, pos)
                    except ValueError:
                        raise ValueError(
                            f"Reference base mismatch at {chrom}:{pos} (expected {ref}, got {fasta.fetch(chrom, pos - 1, pos)})"
                        )

                    start, end = pos - SL // 2 - CL_max // 2, pos + SL // 2 + CL_max // 2
                    pos_start = pos - start
                    ref_seq = fasta[chrom][start - 1:end - 1].seq.upper()
                    ref_len = len(ref)
                    alt_len = len(alt)
                    alt_seq = ref_seq[:pos_start] + alt + ref_seq[(pos_start + ref_len):]

                    ref_seq_len = len(ref_seq)
                    alt_seq_len = len(alt_seq)

                    ref_seq = one_hot_encode(ref_seq)
                    alt_seq = one_hot_encode(alt_seq)

                    ref_seq = torch.tensor(ref_seq, dtype=torch.float32).T.unsqueeze(0).to('cuda')
                    alt_seq = torch.tensor(alt_seq, dtype=torch.float32).T.unsqueeze(0).to('cuda')

                    ref_prediction = torch.stack([model(ref_seq)[0].detach() for model in models]).mean(
                        dim=0).cpu().numpy()[0, :, :]
                    alt_prediction = torch.stack([model(alt_seq)[0].detach() for model in models]).mean(
                        dim=0).cpu().numpy()[0, :, :]

if __name__ == '__main__':
    main()