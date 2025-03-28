from pkg_resources import resource_filename

import pandas as pd

import argparse
import copy
from pysam import VariantRecord, VariantFile
from pyfastx import Fasta

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from model import SpliceFormer

import logging


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
    parser.add_argument(
        "-a", "--annotations", required=True,
        help='"grch37" (GENCODE V24lift37 canonical annotation file in '
             'package), "grch38" (GENCODE V24 canonical annotation file in '
             'package), or path to a similar custom gene annotation file'
    )
    parser.add_argument(
        "-d", "--distance", nargs='?', default=50,
        type=int, choices=(0, 5000),
        help="distance between the variant and gained/lost splice"
             "site, default is 50"
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


# def get_deltas(ref_prediction, alt_prediction, pos_s, crop, ref_len, alt_len, ref_seq_len, alt_seq_len):
#     """
#
#     Args:
#       ref_prediction: Splice site scores for all nucleotides in the reference sequence
#       alt_prediction: Splice site scores for all nucleotides in the alternative sequence
#       pos_s: Variant position minus sequence start position
#       crop: Region to crop from both sides of the delta tracks
#
#     Returns: Donor and acceptor delta tracks (difference between alt_prediction and ref_prediction)
#
#     """
#     ref_acceptor = ref_prediction[1, :]
#     alt_acceptor = alt_prediction[1, :]
#     ref_donor = ref_prediction[2, :]
#     alt_donor = alt_prediction[2, :]
#
#     delta_1_a = alt_acceptor[:pos_s] - ref_acceptor[:pos_s]
#     delta_1_d = alt_donor[:pos_s] - ref_donor[:pos_s]
#     delta_3_a = alt_acceptor[pos_s + alt_len:] - ref_acceptor[pos_s + ref_len:]
#     delta_3_d = alt_donor[pos_s + alt_len:] - ref_donor[pos_s + ref_len:]
#     if ref_seq_len == alt_seq_len:
#         delta_2_a = alt_acceptor[pos_s:pos_s + ref_len] - ref_acceptor[pos_s:pos_s + ref_len]
#         delta_2_d = alt_donor[pos_s:pos_s + ref_len] - ref_donor[pos_s:pos_s + ref_len]
#     elif ref_seq_len > alt_seq_len:
#         a_pad = np.pad(alt_acceptor[pos_s:pos_s + alt_len], (0, ref_len - alt_len), 'constant', constant_values=0)
#         d_pad = np.pad(alt_donor[pos_s:pos_s + alt_len], (0, ref_len - alt_len), 'constant', constant_values=0)
#         delta_2_a = a_pad - ref_acceptor[pos_s:pos_s + ref_len]
#         delta_2_d = d_pad - ref_donor[pos_s:pos_s + ref_len]
#
#     elif ref_seq_len < alt_seq_len:
#         a_pad = np.pad(ref_acceptor[pos_s:pos_s + ref_len], (0, alt_len - ref_len), 'constant', constant_values=0)
#         d_pad = np.pad(ref_donor[pos_s:pos_s + ref_len], (0, alt_len - ref_len), 'constant', constant_values=0)
#         delta_2_a = alt_acceptor[pos_s:pos_s + alt_len] - a_pad
#         delta_2_d = alt_donor[pos_s:pos_s + alt_len] - d_pad
#
#         delta_2_a = np.append(delta_2_a[:ref_len - 1],
#                               delta_2_a[np.argmax(np.absolute(delta_2_a[ref_len - 1:alt_len]))])
#         delta_2_d = np.append(delta_2_d[:ref_len - 1],
#                               delta_2_d[np.argmax(np.absolute(delta_2_d[ref_len - 1:alt_len]))])
#
#     acceptorDelta = np.concatenate([delta_1_a, delta_2_a, delta_3_a])
#     donorDelta = np.concatenate([delta_1_d, delta_2_d, delta_3_d])
#     return acceptorDelta[crop:-crop], donorDelta[crop:-crop]

class Annotator:

    def __init__(self, ref_fasta, annotations, device):

        if annotations == 'grch37':
            annotations = resource_filename(__name__, 'annotations/grch37.txt')
        elif annotations == 'grch38':
            annotations = resource_filename(__name__, 'annotations/grch38.txt')

        try:
            df = pd.read_csv(annotations, sep='\t', dtype={'CHROM': object})
            self.genes = df['#NAME'].to_numpy()
            self.chroms = df['CHROM'].to_numpy()
            self.strands = df['STRAND'].to_numpy()
            self.tx_starts = df['TX_START'].to_numpy() + 1
            self.tx_ends = df['TX_END'].to_numpy()
            self.exon_starts = [np.asarray([int(i) for i in c.split(',') if i]) + 1
                                for c in df['EXON_START'].to_numpy()]
            self.exon_ends = [np.asarray([int(i) for i in c.split(',') if i])
                              for c in df['EXON_END'].to_numpy()]
        except IOError as e:
            logging.error('{}'.format(e))
            exit()
        except (KeyError, pd.errors.ParserError) as e:
            logging.error('Gene annotation file {} not formatted properly: {}'.format(annotations, e))
            exit()

        try:
            self.ref_fasta = Fasta(ref_fasta)
        except IOError as e:
            logging.error('{}'.format(e))
            exit()

        paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
        self.models = [load_model(resource_filename(__name__, x), device) for x in paths]

    def get_name_and_strand(self, chrom, pos):

        chrom = normalise_chrom(chrom, list(self.chroms)[0])
        idxs = np.intersect1d(np.nonzero(self.chroms == chrom)[0],
                              np.intersect1d(np.nonzero(self.tx_starts <= pos)[0],
                                             np.nonzero(pos <= self.tx_ends)[0]))

        if len(idxs) >= 1:
            return self.genes[idxs], self.strands[idxs], idxs
        else:
            return [], [], []

    def get_pos_data(self, idx, pos):

        dist_tx_start = self.tx_starts[idx] - pos
        dist_tx_end = self.tx_ends[idx] - pos
        dist_exon_bdry = min(np.union1d(self.exon_starts[idx], self.exon_ends[idx]) - pos, key=abs)
        dist_ann = (dist_tx_start, dist_tx_end, dist_exon_bdry)

        return dist_ann


def get_delta_scores(ann, record, models, SL, CL_max, device, fasta):
    delta_score = []  # ???

    chrom = record.chrom if record.chrom.startswith("chr") else f"chr{record.chrom}"
    pos = record.pos
    ref = record.ref
    alts = [alt for alt in record.alts]

    ref_fa = fasta[chrom][pos - 1:pos - 1 + len(ref)].seq

    assert ref == ref_fa, f"Reference base mismatch at {chrom}:{pos} (expected {ref}, got {ref_fa})"

    (genes, strands, idxs) = ann.get_name_and_strand(chrom, pos)

    if len(idxs) == 0:
        return delta_score  # add to sep function else closes a script

    start, end = pos - SL // 2 - CL_max // 2, pos + SL // 2 + CL_max // 2  # mb + len-1
    pos_start = pos - start
    try:
        ref_seq = fasta[chrom][start - 1:end - 1].seq.upper()
    except (IndexError, ValueError):
        logging.warning('Skipping record (fasta issue): {}'.format(record))
        return delta_score
    ref_len = len(ref)

    ref_seq_len = len(ref_seq)

    alt_number = len(alts)
    for i in range(alt_number):
        for j in range(len(idxs)):

            if '.' in alts[i] or '-' in alts[i] or '*' in alts[i]:
                continue

            if '<' in alts[i] or '>' in alts[i]:
                continue

            if len(record.ref) > 1 and len(alts[i]) > 1:
                delta_score.append("{}|{}|.|.|.|.|.|.|.|.".format(alts[i], genes[j]))
                continue

            alt_seq = ref_seq[:pos_start] + alts[i] + ref_seq[(pos_start + ref_len):]

            alt_len = len(alts[i])

            alt_seq_len = len(alt_seq)
            if alt_seq_len > ref_seq_len:
                alt_seq = alt_seq[:ref_seq_len]
            elif alt_seq_len < ref_seq_len:
                alt_seq = alt_seq + fasta[chrom][end - 1:end - 1 + ref_seq_len - alt_seq_len].seq.upper()

            ref_seq_encoded = one_hot_encode(ref_seq)
            alt_seq_encoded = one_hot_encode(alt_seq)

            if strands[j] == '-':
                ref_seq_encoded = ref_seq_encoded[::-1, ::-1]
                alt_seq_encoded = alt_seq_encoded[::-1, ::-1]

            # get predictions
            ref_seq_tensor = torch.tensor(ref_seq_encoded.copy(), dtype=torch.float32).T.unsqueeze(0).to(
                device)
            alt_seq_tensor = torch.tensor(alt_seq_encoded.copy(), dtype=torch.float32).T.unsqueeze(0).to(
                device)

            ref_prediction = torch.stack([model(ref_seq_tensor)[0].detach() for model in models]).mean(
                dim=0).cpu().numpy()[0, :, :]
            alt_prediction = torch.stack([model(alt_seq_tensor)[0].detach() for model in models]).mean(
                dim=0).cpu().numpy()[0, :, :]

            if strands[j] == '-':
                ref_prediction = ref_prediction[:, ::-1]
                alt_prediction = alt_prediction[:, ::-1]

            prediction = np.stack((ref_prediction, alt_prediction), axis=0)

            prediction_cropped = prediction[:, :, CL_max // 2:-CL_max // 2]

            # acceptor
            idx_pa = (prediction_cropped[1, 1, :] - prediction_cropped[0, 1, :]).argmax()
            idx_na = (prediction_cropped[0, 1, :] - prediction_cropped[1, 1, :]).argmax()
            # donor
            idx_pd = (prediction_cropped[1, 2, :] - prediction_cropped[0, 2, :]).argmax()
            idx_nd = (prediction_cropped[0, 2, :] - prediction_cropped[1, 2, :]).argmax()

            Gene_name = genes[j]
            Alt_base = alts[i]

            # delta score
            DS_AG = (prediction_cropped[1, 1, idx_pa] - prediction_cropped[0, 1, idx_pa])
            DS_AL = (prediction_cropped[0, 1, idx_na] - prediction_cropped[1, 1, idx_na])
            DS_DG = (prediction_cropped[1, 2, idx_pd] - prediction_cropped[0, 2, idx_pd])
            DS_DL = (prediction_cropped[0, 2, idx_nd] - prediction_cropped[1, 2, idx_nd])

            # delta position
            # DP_AG = idx_pa - SL // 2 - CL_max // 2
            # DP_AL = idx_na - SL // 2 - CL_max // 2
            # DP_DG = idx_pd - SL // 2 - CL_max // 2
            # DP_DL = idx_nd - SL // 2 - CL_max // 2

            DP_AG = idx_pa - SL // 2
            DP_AL = idx_na - SL // 2
            DP_DG = idx_pd - SL // 2
            DP_DL = idx_nd - SL // 2

            delta_score.append("{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}".format(
                Alt_base,
                Gene_name,
                DS_AG, DS_AL, DS_DG, DS_DL,
                DP_AG, DP_AL, DP_DG, DP_DL
            ))

    return delta_score


def main():
    args = get_options()

    variant_dist = args.distance

    SL = variant_dist * 2
    CL_max = 50000 - SL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = load_model(CL_max, device)

    ann = Annotator(args.reference, args.annotations, device)

    fasta = ann.ref_fasta

    try:
        vcf = VariantFile(args.input)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    header = vcf.header

    header.add_line('##INFO=<ID=Spliceformer,Number=.,Type=String,Description="Spliceformer v1.0.0_custom '
                    'variant annotation. These include delta scores (DS) and delta positions (DP) for '
                    'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
                    'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')

    try:
        output = VariantFile(args.output, mode='w', header=header)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    for record in vcf:

        info = get_delta_scores(ann, record, models, SL, CL_max, device, fasta)

        if len(info) > 0:
            record.info['Spliceformer'] = info

        output.write(record)

    vcf.close()
    output.close()


if __name__ == '__main__':
    main()
