#!/usr/bin/env python
"""
Multivariate Deep Neural Network Autoencoder network anomaly detection. Anomaly detection is performed on model output
by ranking of loss scores within a time window from output of model. An output file is created with a timestamp prepended,
and values of hyper-parameters in name.

**Abbreviation of hyper-parameter descriptions in name are as follows:**

- tg -> target: {auto, next}
- lr -> learnrate
- nl -> number of hidden layers
- hs -> size of hidden layers
- mb -> minibatch size
- bn -> Whether of not to use batch normalization
- kp -> The keep probability for drop out
- ds -> The family of multivariate distribution to use {ident, diag, full}
- bc -> bad count for early stopping
- em -> The ratio of embedding sizes to number of classes for categorical inputs
"""

import os
import sys
# So we can run this code on arbitrary environment which has tensorflow but not safekit installed

# TODO: Test this on DS data
# TODO: skipheader error message that is informative
# TODO: Fix replay learning
# TODO: Comment for usage.
cyberpath = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, cyberpath)

import tensorflow as tf
import numpy as np
import json
from safekit.batch import OnlineBatcher, NormalizingReplayOnlineBatcher, split_batch
from safekit.graph_training_utils import ModelRunner, EarlyStop
from safekit.tf_ops import join_multivariate_inputs, dnn, \
diag_mvn_loss, multivariate_loss, eyed_mvn_loss, \
full_mvn_loss, layer_norm, batch_normalize
from safekit.util import get_multivariate_loss_names, make_feature_spec, make_loss_spec, Parser
import time
import random


def return_parser():
    """
    Defines and returns argparse ArgumentParser object.

    :return: ArgumentParser
    """
    parser = Parser("Dnn auto-encoder for online unsupervised training.")
    parser.add_argument('datafile',
                        type=str,
                        help='The csv data file for our unsupervised training.'+\
                             'fields: day, user, redcount, [count1, count2, ...., count408]')
    parser.add_argument('results_folder', type=str,
                        help='The folder to print results to.')
    parser.add_argument('dataspecs', type=str,
                        help='Filename of json file with specification of feature indices.')
    parser.add_argument('-learnrate', type=float, default=0.001,
                        help='Step size for gradient descent.')
    parser.add_argument('-numlayers', type=int, default=3,
                        help='Number of hidden layers.')
    parser.add_argument('-hiddensize', type=int, default=20,
                        help='Number of hidden units in hidden layers.')
    parser.add_argument('-mb', type=int, default=256,
                        help='The mini batch size for stochastic gradient descent.')
    parser.add_argument('-act', type=str, default='tanh',
                        help='May be "tanh" or "relu"')
    parser.add_argument('-norm', type=str, default="none",
                        help='Can be "layer", "batch", or "none"')
    parser.add_argument('-keep_prob', type=float, default=None,
                        help='Percent of nodes to keep for dropout layers.')
    parser.add_argument('-debug', action='store_true',
                        help='Use this flag to print feed dictionary contents and dimensions.')
    parser.add_argument('-dist', type=str, default='diag',
                        help='"diag" or "ident". Describes whether to model multivariate guassian with identity, '
                             'or arbitrary diagonal covariance matrix.')
    parser.add_argument('-maxbadcount', type=str, default=20,
                        help='Threshold for early stopping.')
    parser.add_argument('-embedding_ratio', type=float, default=.75,
                        help='For determining size of embeddings for categorical features.')
    parser.add_argument('-min_embed', type=int, default=2,
                        help='Minimum size for embeddings of categorical features.')
    parser.add_argument('-max_embed', type=int, default=1000,
                        help='Maximum size for embeddings of categorical features.')
    parser.add_argument('-verbose', type=int, default=0, help='1 to print full loss contributors.')
    parser.add_argument('-variance_floor', type=float, default=0.01,
                        help='Parameter for diagonal MVN learning.')
    parser.add_argument('-initrange', type=float, default=1.0,
                        help='For weight initialization.')
    parser.add_argument('-decay_rate', type=float, default=1.0,
                        help='Exponential learn rate decay for gradient descent.')
    parser.add_argument('-decay_steps', type=int, default=20,
                        help='Number of updates to perform learn rate decay')
    parser.add_argument('-alpha', type=float, default=0.99,
                        help='Parameter for exponential moving average and variance')
    parser.add_argument('-input_norm', action='store_true',
                        help='Use this flag for online normalization')
    parser.add_argument('-refresh_ratio', type=float, default=0.5,
                        help='The proportion of the new mini-batch to use in refreshing the pool.')
    parser.add_argument('-ratio', nargs='+', type=int, default=[1, 1],
                        help='(tuple) (x, y): Number of new batches of data points x and number of old data points y.')
    parser.add_argument('-pool_size', type=int, default=9000,
                        help='The scale of the pool.')
    parser.add_argument('-random_seed', type=int, default=None,
                        help='For reproducible results')
    parser.add_argument('-replay', action='store_true',
                        help='Use this flag for replay learning')
    parser.add_argument('-delimiter', type=str, default=' ',
                        help="Delimiter for input text file. You should be using ' ' for the dayshuffled cert.")
    parser.add_argument('-skipheader', action='store_true',
                        help="Whether or not to skip first line of input file.")

    return parser


def write_results(datadict, pointloss, outfile):
    """
    Writes loss for each datapoint, along with meta-data to file.

    :param datadict: Dictionary of data names (str) keys to numpy matrix values for this mini-batch.
    :param pointloss: MB X 1 numpy array
    :param outfile: Where to write results.
    """
    for d, u, t, l, in zip(datadict['time'].flatten().tolist(), datadict['user'].tolist(),
                           datadict['redteam'].flatten().tolist(), pointloss.flatten().tolist()):
        outfile.write('%s %s %s %s\n' % (d, u, t, l))


def write_all_contrib(datadict, pointloss, contrib, outfile):
    """
    Writes loss, broken down loss from all contributors, and metadata for each datapoint to file.

    :param datadict: Dictionary of data names (str) keys to numpy matrix values for this mini-batch.
    :param pointloss: MB X 1 numpy array.
    :param contrib: MB X total_num_loss_contributors nompy array.
    :param outfile: Where to write results.
    """
    for time, user, red, loss, contributor in zip(datadict['time'].tolist(),
                                                  datadict['user'].tolist(),
                                                  datadict['redteam'].tolist(),
                                                  pointloss.flatten().tolist(),
                                                  contrib.tolist()):
        outfile.write('%s %s %s %s ' % (time, user, red, loss))
        outfile.write(str(contributor).strip('[').strip(']').replace(',', ''))
        outfile.write('\n')

if __name__ == '__main__':
    np.seterr(all='raise')
    args = return_parser().parse_args()
    if args.random_seed is None:
        args.random_seed = random.randint(0,1000)

    normalizers = {'none': None,
                   'layer': layer_norm,
                   'batch': batch_normalize}
 
    outfile_name = "lanlAgg__lr_%.2f__tg_auto__rs_%s__ir_%.2f__nl_%s__hs_%s__mb_%s__nm_%s__kp_%s__ds_%s__bc_%s__em_%s__dr_%.2f__ds_%s" % (
                    args.learnrate,
                    args.random_seed,
                    args.initrange,
                    args.numlayers,
                    args.hiddensize,
                    args.mb,
                    args.norm,
                    args.keep_prob,
                    args.dist,
                    args.maxbadcount,
                    args.embedding_ratio,
                    args.decay_rate,
                    args.decay_steps)
    tf.set_random_seed(args.random_seed)
    
    start_time = time.time()
    outfile_name = str(start_time) + '__' + outfile_name
    if not args.results_folder.endswith('/'):
        args.results_folder += '/'
    os.system('mkdir /tmp/dnn_agg/')
    outfile = open('/tmp/dnn_agg/' + outfile_name, 'w')

    if args.act == 'tanh':
        activation = tf.tanh
    elif args.act == 'relu':
        activation = tf.nn.relu
    else:
        raise ValueError('Activation must be "relu", or "tanh"')

    if args.dist == "ident":
        mvn = eyed_mvn_loss
    elif args.dist == "diag":
        mvn = diag_mvn_loss
    elif args.dist == "full":
        mvn = full_mvn_loss
    dataspecs = json.load(open(args.dataspecs, 'r'))
    datastart_index = dataspecs['counts']['index'][0]
    if not args.replay:
        data = OnlineBatcher(args.datafile, args.mb, skipheader=args.skipheader,
                             alpha=args.alpha, norm=args.input_norm,
                             delimiter=args.delimiter,
                             datastart_index=datastart_index)
    else:
        data = NormalizingReplayOnlineBatcher(args.datafile, args.mb, skipheader=args.skipheader,
                                              refresh_ratio=args.refresh_ratio, ratio=tuple(args.ratio),
                                              pool_size=args.pool_size, delimiter=args.delimiter,
                                              alpha=args.alpha,datastart_index=datastart_index)
    
    feature_spec = make_feature_spec(dataspecs)
    x, ph_dict = join_multivariate_inputs(feature_spec, dataspecs,
                                          args.embedding_ratio, args.max_embed, args.min_embed)

    h = dnn(x, layers=[args.hiddensize for i in range(args.numlayers)],
            act=activation, keep_prob=args.keep_prob, norm=normalizers[args.norm],
            scale_range=args.initrange)

    loss_spec = make_loss_spec(dataspecs, mvn)
    loss_matrix = multivariate_loss(h, loss_spec, ph_dict, variance_floor=args.variance_floor)
    loss_vector = tf.reduce_sum(loss_matrix, reduction_indices=1)  # is MB x 1
    loss = tf.reduce_mean(loss_vector)  # is scalar
    loss_names = get_multivariate_loss_names(loss_spec)
    eval_tensors = [loss, loss_vector, loss_matrix]
    model = ModelRunner(loss, ph_dict, learnrate=args.learnrate, opt='adam', debug=args.debug, decay_rate=args.decay_rate, decay_steps=args.decay_steps)
    raw_batch = data.next_batch()
    current_loss = sys.float_info.max
    not_early_stop = EarlyStop(args.maxbadcount)

    loss_feats = [triple[0] for triple in loss_spec]
    # training loop
    continue_training = not_early_stop(raw_batch, current_loss)
    while continue_training:  # mat is not None and self.badcount < self.badlimit and loss != inf, nan:
        datadict = split_batch(raw_batch, dataspecs)
        targets = {'target_' + name: datadict[name] for name in loss_feats}
        datadict.update(targets)
        current_loss, pointloss, contrib = model.eval(datadict, eval_tensors)
        model.train_step(datadict)
        if args.verbose == 1 and not data.replay:
            write_all_contrib(datadict, pointloss, contrib, outfile)
        elif args.verbose == 0 and not data.replay:
            write_results(datadict, pointloss, outfile)
        print('index: %s loss: %.4f' % (data.index, current_loss))
        raw_batch = data.next_batch()
        continue_training = not_early_stop(raw_batch, current_loss)
        if continue_training < 0:
            exit(1)
    outfile.close()
    os.system('mv /tmp/dnn_agg/' + outfile_name + ' ' + args.results_folder + outfile_name)
