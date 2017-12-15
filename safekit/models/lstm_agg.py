"""

"""
import json
import os
import sys
# TODO: Comment for usage.
# TODO: Test this on  DS data
# TODO: skipheader error message that is informative
# TODO: Make consistent output printing and writing to file for aggregate and baseline models.
# TODO: Comment crazy transform functions

# So we can run this code on arbitrary environment which has tensorflow but not safekit installed
cyberpath = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, cyberpath)

import itertools
import numpy as np
import tensorflow as tf
import time
import math
from safekit.batch import StateTrackingBatcher
from safekit.tf_ops import diag_mvn_loss, multivariate_loss, eyed_mvn_loss, full_mvn_loss
from safekit.util import get_multivariate_loss_names, make_feature_spec, make_loss_spec, Parser
from safekit.tf_ops import weights, swapping_rnn, layer_norm_rnn
from safekit.graph_training_utils import ModelRunner, EarlyStop


def lstm_parser():

    parser = Parser(description="Cert Aggregate Feature LSTM.")
    parser.add_argument('datafile', type=str,
                        help='Path to data file.')
    parser.add_argument('results_folder', type=str,
                        help='Folder where to write losses.')
    parser.add_argument('dataspecs', type=str,
                        help='Name of json file with specs for splitting data.')
    parser.add_argument('-num_steps', type=int,
                        help='Number of time steps for truncated backpropagation.', default=5)
    parser.add_argument('-learnrate', type=float, default=0.01,
                        help='Step size for gradient descent.')
    parser.add_argument('-initrange', type=float, default=0.0001,
                        help='For initialization of weights.')
    parser.add_argument('-numlayers', type=int, default=3,
                        help='Number of hidden layers')
    parser.add_argument('-hiddensize', type=int, default=3,
                        help='Number of hidden nodes per layer')
    parser.add_argument('-verbose', type=int, default=1,
                        help='Level to print training progress and/or other details.')
    parser.add_argument('-mb', type=int, default=21,
                        help='The max number of events in the structured mini_batch.')
    parser.add_argument('-embedding_ratio', type=float, default=0.5,
                        help='Embedding_ratio * num_classes = embedding size.')
    parser.add_argument('-min_embedding', type=int, default=5,
                        help='Minimum embedding size.')
    parser.add_argument('-max_embedding', type=int, default=500,
                        help='Maximum embedding size.')
    parser.add_argument('-use_next_time_step', default=0, type=int,
                        help='Whether to predict next time step or autoencode.')
    parser.add_argument('-act', default='relu', type=str,
                        help='A string denoting the activation function.')
    parser.add_argument('-dist', default='diag', type=str,
                        help='A string denoting the multivariate normal type for prediction.')
    parser.add_argument('-variance_floor', default='0.1', type=float,
                        help='Float to derive variance floor.')
    parser.add_argument('-norm', type=str,
                        help='"layer" for layer normalization. Default is None.')
    parser.add_argument('-keep_prob', type=float, default=None,
                        help='Percent of nodes to keep for dropout layers.')
    parser.add_argument('-debug', action='store_true',
                        help='Use this flag to print feed dictionary contents and dimensions.')
    parser.add_argument('-random_seed', type=int, default=5,
                        help='Random seed for reproducible experiments.')
    parser.add_argument('-replay_ratio', type=int, nargs='+', default=(1, 0)),
    parser.add_argument('-delimiter', type=str, default=' ',
                        help="Delimiter for input text file. You should be using ' ' for the dayshuffled cert.")
    parser.add_argument('-maxbadcount', type=int, default=100,
                        help="For stopping training when loss does not improve.")
    parser.add_argument('-residual', action='store_true',
                        help="Flag for calculating residual (difference between sequential actions) instead of next action")
    parser.add_argument('-skipheader', action='store_true',
                        help="Whether or not to skip first line of input file.")
    parser.add_argument('-alpha', type=float, default=0.99,
                        help='Parameter for exponential moving average and variance')
    parser.add_argument('-input_norm', action='store_true',
                        help='Use this flag for online normalization')
    return parser


def regression_transform(matrix, eval_indices):
    """

    :param matrix:
    :param eval_indices:
    :return:
    """
    return np.concatenate([matrix[timestep][indices.astype(int)]
                           for timestep, indices
                           in enumerate(eval_indices)], axis=0)


def evaluation_transform(matrix, eval_indices):
    """

    :param matrix:
    :param eval_indices:
    :return:
    """
    num_steps = matrix.shape[0]
    return list(itertools.chain.from_iterable([matrix[tstep][indices.astype(int)]
                                               for tstep, indices
                                               in zip(range(num_steps), eval_indices)]))


def index_classification_transform(matrix, eval_indices):
    """

    :param matrix:
    :param eval_indices:
    :return:
    """
    return np.concatenate([matrix[timestep][indices.astype(int)]
                                         for timestep, indices
                                         in enumerate(eval_indices)], axis=0)


def redteam_transform(matrix, eval_indices, dataspec=None):
    """

    :param matrix:
    :param eval_indices:
    :param dataspec:
    :return:
    """
    matrix = np.sum(matrix, axis=2)
    return evaluation_transform(matrix, eval_indices)


def augment_datadict(datadict, next_time_step, feature_spec, residual=False):
    """

    :param datadict:
    :param next_time_step:
    :param feature_spec:
    :param residual:
    :return:
    """
    startIdx = next_time_step

    for cat_feat in feature_spec['categorical']:
        datadict["target_%s" % cat_feat] = index_classification_transform(datadict[cat_feat][startIdx:],
                                                                          datadict["eval_indices"])
        endIdx = (datadict[cat_feat].shape[0], -1)[next_time_step]
        datadict[cat_feat] = datadict[cat_feat][:endIdx]
    for cont_feat in feature_spec['continuous']:
        if residual:
             endIdx = (datadict[cont_feat].shape[0], -1)[next_time_step]
             pre = datadict[cont_feat][:endIdx]
             post_residual = datadict[cont_feat][startIdx:] - pre 
             datadict["target_%s" % cont_feat] = regression_transform(post_residual,
                                                                 datadict["eval_indices"])
             datadict[cont_feat] = pre
        else:
             datadict["target_%s" % cont_feat] = regression_transform(datadict[cont_feat][startIdx:],
                                                                 datadict["eval_indices"])
             endIdx = (datadict[cont_feat].shape[0], -1)[next_time_step]
             datadict[cont_feat] = datadict[cont_feat][:endIdx]

    for feat in ['user']:
        endIdx = (datadict[feat].shape[0], -1)[next_time_step]
        datadict[feat + '_eval'] = evaluation_transform(datadict[feat][:endIdx],
                                                        datadict["eval_indices"])
    for feat in ['time', 'redteam']:
        endIdx = (datadict[feat].shape[0], -1)[next_time_step]
        datadict[feat + '_eval'] = redteam_transform(datadict[feat][:endIdx],
                                                        datadict["eval_indices"])


if __name__ == '__main__':
    #========================== SETUP ==========================================
    args = lstm_parser().parse_args()
    args.layers = [args.hiddensize for x in range(args.numlayers)]

    outfile_name = "cert_mv_auto_act_%s_lr_%s_nl_%s_hs_%s_mb_%s_nm_%s_kp_%s_ds_%s_em_%s_rs_%s" % (args.act,
                                                                                                  args.learnrate,
                                                                                          len(args.layers),
                                                                                          args.layers[0],
                                                                                          args.mb,
                                                                                          args.norm,
                                                                                          args.keep_prob,
                                                                                          args.dist,
                                                                                          args.embedding_ratio,
                                                                                          args.random_seed)
    outfile_name = str(time.time()) + outfile_name
    if not args.results_folder.endswith('/'):
        args.results_folder += '/'

    if args.dist == "ident":
        mvn = eyed_mvn_loss
    elif args.dist == "diag":
        mvn = diag_mvn_loss
    elif args.dist == "full":
        mvn = full_mvn_loss
    else:
        mvn = diag_mvn_loss

    tf.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # ======================== BUILD GRAPH ===================================================


    dataspecs = json.load(open(args.dataspecs, 'r'))
    ph_dict = {'training': tf.placeholder(tf.bool)}
    embeddings, continuous_features, targets = {}, {}, {}
    feature_spec = make_feature_spec(dataspecs)
    # Make placeholders for all input data and select embeddings for categorical data
    for dataname in feature_spec['categorical']:
        embedding_size = math.ceil(args.embedding_ratio * dataspecs[dataname]['num_classes'])
        embedding_size = int(max(min(args.max_embedding, embedding_size), args.min_embedding))
        with tf.variable_scope(dataname):
            ph_dict[dataname] = tf.placeholder(tf.int32, [args.num_steps, None], name=dataname)
            embeddings[dataname] = tf.nn.embedding_lookup(weights('tnorm',
                                                                  (dataspecs[dataname]['num_classes'],
                                                                   embedding_size)),
                                                          ph_dict[dataname])

    for dataname in feature_spec['continuous']:
        ph_dict[dataname] = tf.placeholder(tf.float32,
                                           [args.num_steps, None, len(dataspecs[dataname]['index'])],
                                           name=dataname)
        continuous_features[dataname] = ph_dict[dataname]

    # concatenate all features
    features = tf.concat(continuous_features.values() + embeddings.values(), 2, name='features')

    # split into a list along time dimension
    input_features = tf.unstack(features)

    initial_state = [[tf.placeholder(tf.float32, (None, units), name='cell_'),
                      tf.placeholder(tf.float32, (None, units), name='hidden')]
                      for units in args.layers]

    # Run prepared list of matrices through rnn
    if args.norm == 'layer':
        hidden_states, states = layer_norm_rnn(input_features,
                                               initial_state=initial_state,
                                               layers=args.layers,
                                               sequence_lengths=None,
                                               state_index=None)
    else:
        hidden_states, states = swapping_rnn(input_features,
                                             initial_state=initial_state,
                                             layers=args.layers,
                                             sequence_lengths=None,
                                             state_index=None)


    # evaluate on selected outputs from rnn
    eval_vecs = [tf.placeholder(tf.int32, [None], name='eval_%s' % i)
                 for i in range(args.num_steps - args.use_next_time_step)]
    hidden_states = [tf.nn.embedding_lookup(hidden_state, state_indices)
                     for hidden_state, state_indices in zip(hidden_states, eval_vecs)]
    hidden_matrix = tf.concat(hidden_states, 0)

    loss_spec = make_loss_spec(dataspecs, mvn)

    loss_matrix = multivariate_loss(hidden_matrix, loss_spec, ph_dict)
    loss_vector = tf.reduce_sum(loss_matrix, reduction_indices=1)  # is MB x 1
    loss = tf.reduce_mean(loss_vector)  # is scalar
    loss_names = get_multivariate_loss_names(loss_spec)
    eval_tensors = [loss, loss_vector, loss_matrix] + np.array(states).flatten().tolist()

    # Create an object to train on this graph
    initial_state = np.array(initial_state).flatten().tolist()
    ph_dict.update({'initial_state': initial_state,
                    'eval_indices': eval_vecs})

    trainer = ModelRunner(loss, ph_dict, args.learnrate, debug=args.debug)
    data = StateTrackingBatcher(args.datafile, dataspecs,
                                batchsize=args.mb,
                                num_steps=args.num_steps,
                                layers=args.layers,
                                next_step=args.use_next_time_step,
                                replay_ratio=args.replay_ratio,
                                delimiter=args.delimiter,
                                skipheader=args.skipheader,
                                standardize=args.input_norm,
                                alpha=args.alpha,
                                datastart_index=dataspecs['counts']['index'][0])

    # ========================== TRAIN ===============================================================
    os.system('mkdir /tmp/lstm_agg/')
    with open('/tmp/lstm_agg/' + outfile_name, 'w') as loss_file:
        header = 'time user red loss '
        if args.verbose == 3:
            header += ' '.join(loss_names)
        loss_file.write(header + '\n')
        datadict = data.next_batch()
        current_loss = sys.float_info.max
        not_early_stop = EarlyStop(args.maxbadcount)
        continue_training = not_early_stop(datadict, current_loss)
        while continue_training:  # mat is not None and self.badcount < self.badlimit and loss != inf, nan:
            augment_datadict(datadict, args.use_next_time_step, feature_spec, args.residual)
            datadict['training'] = True
            val = trainer.train_step(datadict, eval_tensors=eval_tensors, update=True)
            current_loss, current_loss_vector, current_loss_matrix, current_states = val[1], val[2], val[3], val[4:]
            data.update_states(current_states)
            for item in ['redteam_eval', 'time_eval', 'user_eval']:
                assert len(current_loss_vector) == len(datadict[item]), (
                    'Mismatched lengths for evaluation lists\n'
                    '%s: %s\n'
                    'loss_vector %s\n' % (item, len(datadict[item]),
                                          len(current_loss_vector)))
            loss_values = zip(datadict['time_eval'],
                              datadict['user_eval'],
                              datadict['redteam_eval'],
                              current_loss_vector.tolist())
            sorted_day_values = sorted(loss_values, key=lambda row: row[0], reverse=False)
            if data.event_number % 100 == 0:
                print("%s loss: %s" % (data.event_number, current_loss))
            if not data.replay and not (math.isnan(current_loss) or math.isinf(current_loss)):
                for time, user, red, loss in sorted_day_values:
                    loss_file.write('%s %s %s %s ' % (time, user, red, loss))
                    loss_file.write('\n')
            datadict = data.next_batch()
            continue_training = not_early_stop(datadict, current_loss)
            if continue_training < 0:
                exit(1)
    os.system('mv /tmp/lstm_agg/' + outfile_name + ' ' + args.results_folder + outfile_name)


