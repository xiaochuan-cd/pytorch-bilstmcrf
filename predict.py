import os
import pickle
import shutil
import logging
import argparse
import tempfile
import subprocess
import collections

import numpy as np
import yaap
import tqdm
import torch
import torch.nn as nn
import torch.optim as O
import torch.autograd as A

import utils
import data as D
import model as M
import evaluate as E


parser = yaap.ArgParser(
    allow_config=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

group = parser.add_group("Basic Options")
group.add("--input-path", type=yaap.path, action="append", required=True,
          help="Path to input file that contains sequences of tokens "
               "separated by spaces.")
group.add("--label-path", type=yaap.path, required=True,
          help="Path to label file that contains sequences of token "
               "labels separated by spaces. Note that the number of "
               "tokens in each sequence must be equal to that of the "
               "corresponding input sequence.")
group.add("--save-dir", type=yaap.path, required=True,
          help="Directory to save outputs (checkpoints, vocabs, etc.)")
group.add("--gpu", type=int, action="append",
          help="Device id of gpu to use. Could supply multiple gpu ids "
               "to denote multi-gpu utilization. If no gpus are "
               "specified, cpu is used as default.")
group.add("--tensorboard", action="store_true", default=False,
          help="Whether to enable tensorboard visualization. Requires "
               "standalone tensorboard, which can be installed via "
               "'https://github.com/dmlc/tensorboard'.")

group = parser.add_group("Word Embedding Options")
group.add("--wordembed-type", type=str, action="append",
          choices=["glove", "fasttext", "none"],
          help="Type of pretrained word embeddings to use for each input. "
               "If multiple input paths are supplied, the same number of "
               "this option must be specified as well. If no option is "
               "supplied, no word embeddings will be used.")
group.add("--wordembed-path", type=yaap.path, action="append",
          help="Path to pre-trained word embeddings. "
               "If embedding type is 'glove', glove-style embedding "
               "file is expected. If embedding type is 'fasttext', "
               "fasttext model file is expected. The number of "
               "specifications must match the number of inputs.")
group.add("--fasttext_path", type=yaap.path, default=None,
          help="If embedding type is 'fasttext', path to fasttext "
               "binaries must be specified. Otherwise, this option is "
               "ignored.")
group.add("--wordembed-freeze", type=bool, action="append",
          help="Whether to freeze embeddings matrix during training. "
               "The number of specifications must match the number of "
               "inputs. If none is specified, word embeddings will not be "
               "frozen by default.")

group = parser.add_group("Model Parameters")
group.add("--word-dim", type=int, action="append",
          help="Dimensions of word embeddings. Must be specified for each "
               "input. Defaults to 300 if none is specified.")
group.add("--lstm-dim", type=int, default=300,
          help="Dimensions of lstm cells. This determines the hidden "
               "state and cell state sizes.")
group.add("--lstm-layers", type=int, default=1,
          help="Layers of lstm cells.")
group.add("--bidirectional", action="store_true", default=False,
          help="Whether lstm cells are bidirectional.")


def check_arguments(args):
    num_inputs = len(args.input_path)

    assert num_inputs > 0, \
        "At least one input file must be specified."

    defaults = {
        "wordembed-type": "none",
        "wordembed-path": None,
        "wordembed-freeze": False,
        "word-dim": 300,
    }

    # default values for list type arguments
    for attr, default in defaults.items():
        attr_name = attr.replace("-", "_")
        if getattr(args, attr_name) is None:
            setattr(args, attr_name, [default] * num_inputs)

    # check if input counts are correct
    for attr in defaults:
        attr_name = attr.replace("-", "_")
        assert len(getattr(args, attr_name)) == num_inputs, \
            f"--{attr} must be specified as many as inputs. " \
            f"specified: {len(getattr(args, attr_name))}, required: {num_inputs}"

    assert 0.0 < args.val_ratio < 1.0, \
        "Specify a valid validation ratio."

    # ensure that the save-dir exists
    os.makedirs(args.save_dir, exist_ok=True)


def main(args):
    logging.basicConfig(level=logging.INFO)
    check_arguments(args)

    label_vocab = pickle.load(os.path.join(args.save_dir, f"vocab-input.pkl"))
    input_vocabs = pickle.load(os.path.join(args.save_dir, f"vocab-label.pkl"))

    logging.info("Initializing model...")
    crf = M.CRF(len(label_vocab))
    model = M.LSTMCRF(
        crf=crf,
        vocab_sizes=[len(v) for v in input_vocabs],
        word_dims=args.word_dim,
        hidden_dim=args.lstm_dim,
        layers=args.lstm_layers,
        dropout_prob=args.dropout_prob,
        bidirectional=args.bidirectional
    )
    model.reset_parameters()
    if args.gpu:
        gpu_main = args.gpu[0]
        model = model.cuda(gpu_main)

    checkpoint = torch.load(os.path.join(args.save_dir, f"model"))
    model.load_state_dict(checkpoint['model'])

    print('finish!')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.tensorboard:
        import tensorboard as T
    main(args)