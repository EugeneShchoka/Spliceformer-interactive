import argparse
import copy
from pysam import VariantRecord, VariantFile
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


def get_name_and_strand(self, chrom, pos):
    chrom = normalise_chrom(chrom, list(self.chroms)[0])
    idxs = np.intersect1d(np.nonzero(self.chroms == chrom)[0],
                          np.intersect1d(np.nonzero(self.tx_starts <= pos)[0],
                                         np.nonzero(pos <= self.tx_ends)[0]))

    if len(idxs) >= 1:
        return self.genes[idxs], self.strands[idxs], idxs
    else:
        return [], [], []


def normalise_chrom(source, target):
    def has_prefix(x):
        return x.startswith('chr')

    if has_prefix(source) and not has_prefix(target):
        return source.strip('chr')
    elif not has_prefix(source) and has_prefix(target):
        return 'chr' + source

    return source


def get_pos_data(self, idx, pos):
    dist_tx_start = self.tx_starts[idx] - pos
    dist_tx_end = self.tx_ends[idx] - pos
    dist_exon_bdry = min(np.union1d(self.exon_starts[idx], self.exon_ends[idx]) - pos, key=abs)
    dist_ann = (dist_tx_start, dist_tx_end, dist_exon_bdry)

    return dist_ann


def get_delta_scores(record, ann, dist_var, mask):
    cov = 2 * dist_var + 1
    wid = 10000 + cov
    delta_scores = []

    (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)

    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])

    seq = ann.ref_fasta[chrom][record.pos - wid // 2 - 1:record.pos + wid // 2].seq

    if len(idxs) == 0:
        return delta_scores

    for j in range(len(record.alts)):
        for i in range(len(idxs)):

            dist_ann = get_pos_data(idxs[i], record.pos)

            if '.' in record.alts[j] or '-' in record.alts[j] or '*' in record.alts[j]:
                continue

            if '<' in record.alts[j] or '>' in record.alts[j]:
                continue

            if len(record.ref) > 1 and len(record.alts[j]) > 1:
                delta_scores.append("{}|{}|.|.|.|.|.|.|.|.".format(record.alts[j], genes[i]))
                continue

            dist_ann = ann.get_pos_data(idxs[i], record.pos)
            pad_size = [max(wid // 2 + dist_ann[0], 0), max(wid // 2 - dist_ann[1], 0)]
            ref_len = len(record.ref)
            alt_len = len(record.alts[j])
            del_len = max(ref_len - alt_len, 0)

            x_ref = 'N' * pad_size[0] + seq[pad_size[0]:wid - pad_size[1]] + 'N' * pad_size[1]
            x_alt = x_ref[:wid // 2] + str(record.alts[j]) + x_ref[wid // 2 + ref_len:]

            x_ref = one_hot_encode(x_ref)[None, :]
            x_alt = one_hot_encode(x_alt)[None, :]

            if strands[i] == '-':
                x_ref = x_ref[:, ::-1, ::-1]
                x_alt = x_alt[:, ::-1, ::-1]

            y_ref = np.mean([ann.models[m].predict(x_ref) for m in range(5)], axis=0)
            y_alt = np.mean([ann.models[m].predict(x_alt) for m in range(5)], axis=0)

            if strands[i] == '-':
                y_ref = y_ref[:, ::-1]
                y_alt = y_alt[:, ::-1]

            if ref_len > 1 and alt_len == 1:
                y_alt = np.concatenate([
                    y_alt[:, :cov // 2 + alt_len],
                    np.zeros((1, del_len, 3)),
                    y_alt[:, cov // 2 + alt_len:]],
                    axis=1)
            elif ref_len == 1 and alt_len > 1:
                y_alt = np.concatenate([
                    y_alt[:, :cov // 2],
                    np.max(y_alt[:, cov // 2:cov // 2 + alt_len], axis=1)[:, None, :],
                    y_alt[:, cov // 2 + alt_len:]],
                    axis=1)

            y = np.concatenate([y_ref, y_alt])

            idx_pa = (y[1, :, 1] - y[0, :, 1]).argmax()
            idx_na = (y[0, :, 1] - y[1, :, 1]).argmax()
            idx_pd = (y[1, :, 2] - y[0, :, 2]).argmax()
            idx_nd = (y[0, :, 2] - y[1, :, 2]).argmax()

            mask_pa = np.logical_and((idx_pa - cov // 2 == dist_ann[2]), mask)
            mask_na = np.logical_and((idx_na - cov // 2 != dist_ann[2]), mask)
            mask_pd = np.logical_and((idx_pd - cov // 2 == dist_ann[2]), mask)
            mask_nd = np.logical_and((idx_nd - cov // 2 != dist_ann[2]), mask)

            delta_scores.append("{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}".format(
                record.alts[j],
                genes[i],
                (y[1, idx_pa, 1] - y[0, idx_pa, 1]) * (1 - mask_pa),
                (y[0, idx_na, 1] - y[1, idx_na, 1]) * (1 - mask_na),
                (y[1, idx_pd, 2] - y[0, idx_pd, 2]) * (1 - mask_pd),
                (y[0, idx_nd, 2] - y[1, idx_nd, 2]) * (1 - mask_nd),
                idx_pa - cov // 2,
                idx_na - cov // 2,
                idx_pd - cov // 2,
                idx_nd - cov // 2))

    return delta_scores


def main():
    args = get_options()

    SL = 5000
    CL_max = 40000

    # ???
    dist_var = 50
    mask = 0
    cov = 2 * dist_var + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = load_model(CL_max, device)

    with VariantFile(args.input, "r") as vcf:
        fasta = Fasta(args.reference)

        with VariantFile(args.output, "w", header=vcf.header) as out:
            for record in vcf:
                delta_scores = []

                chrom = record.chrom if record.chrom.startswith("chr") else f"chr{record.chrom}"
                pos = record.pos
                ref = record.ref
                alt = [alt for alt in record.alts]

                ref_fa = fasta[chrom][pos - 1:pos - 1 + len(ref)].seq

                assert ref == ref_fa, f"Reference base mismatch at {chrom}:{pos} (expected {ref}, got {ref_fa})"

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
                    alt_seq = ref_seq[:pos_start] + alt[i] + ref_seq[(pos_start + len(alt[i]) - 1 + ref_len):]

                    alt_len = len(alt[i])

                    alt_seq_len = len(alt_seq)

                    alt_seq_encoded = one_hot_encode(alt_seq)

                    alt_seq_tensor = torch.tensor(alt_seq_encoded, dtype=torch.float32).T.unsqueeze(0).to(device)

                    alt_prediction = torch.stack([model(alt_seq_tensor)[0].detach() for model in models]).mean(
                        dim=0).cpu().numpy()[0, :, :]

                    y = np.stack([ref_prediction, alt_prediction], axis=0)

                    idx_pa = (y[1, 1, :] - y[0, 1, :]).argmax()
                    idx_na = (y[0, :, 1] - y[1, :, 1]).argmax()
                    idx_pd = (y[1, :, 2] - y[0, :, 2]).argmax()
                    idx_nd = (y[0, :, 2] - y[1, :, 2]).argmax()

                    DS_AG = (y[1, 1, idx_pa] - y[0, 1, idx_pa])
                    DS_AL = (y[0, 1, idx_na] - y[1, 1, idx_na])
                    DS_DG = (y[1, 2, idx_pd] - y[0, 2, idx_pd])
                    DS_DL = (y[0, 2, idx_nd] - y[1, 2, idx_nd])

                    DP_AG = idx_pa - cov // 2
                    DP_AL = idx_na - cov // 2
                    DP_DG = idx_pd - cov // 2
                    DP_DL = idx_nd - cov // 2


if __name__ == '__main__':
    main()
