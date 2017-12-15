#!/usr/bin/env python

"""
This is a two tiered language model for anomaly detection, where the second tier LSTM (log line level)
takes the concatenation of the average sentence vector and final hidden state
from the lower tier (token level) LSTM as input, creating a new context vector and hidden state
for the given user.

Example Command for running a model configuration
-------------------------------------------------

**Raw (character token) tiered model** (The jagged parameter lets the model know there are variable length sequences) ::

python safekit/models/tiered_lm.py results/ safekit/features/specs/lm/lanl_char_config.json data_examples/lanl/lm_feats/raw_day_split/ -test -skipsos -jagged

.. Note ::
    The output results will be printed to /tmp/lanl_result/ and then moved to results/ upon completion
    to avoid experiment slowdown of constant network traffic.

File name convention:
---------------------

- em: embedding size for token embedding
- ns: number of loglines per user per mini-batch for trunctated back propagation through time
- mb: Minibatch size (mini-batch over users)
- lr: learnrate (step size for gradient descent)
- cl: context layers (number of hidden layers for top level (log line level) context rnn)
- lml: language model layers (number of hidden layers for the bottom level, token level, rnn)
- rs: random seed for reproducible results

stdout
------

For each mini-batch the following is printed to standard output ::

    batchsize line_number second status filename index current_loss

Where:

- batchsize: The size of the mini-batch
- line_number: Line number from original auth.txt file (may be off by 1)
- second: The second of the first event in the mini-batch
- status: Whether the model is updating or merely forward propagating
- filename: The current file being processed
- index: The number of samples processed to date
- current_loss: The average loss over the mini-batch

File output
-----------
::

    batch_num line second day user red loss

Where:

- batch_num: The mini-batch this event was a part of
- line: Line number from original auth.txt file (may be off by 1)
- second: The second which the event occurred on
- day: The day the event occurred on
- user: The user who performed the event
- red: Whether this event was a labeled red team activity (1 for red team activity 0 otherwise)
- loss: The anomaly score for this event

.. Note ::

    The runtime of the experiment is also printed to a file called runtimes.txt at the end of training

Input Data
----------

The format of the input makes the following assumptions:

- Input files are together in datafolder, one file for each day.
- Input files are plain text files with one line of integers per log line representing meta data and the tokens from log text.
- Input format for fixed length sequences ::

    line_nums second day user red logtokenid1 .... logtokenid_SentenceLen
- Zero paded Input format for jagged sequences ::

    line_nums second day user red SentenceLen logtokenid1 .... logtokenid_SentenceLen 0 0 .... 0
"""

import os
import sys
# So we can run this code on arbitrary environment which has tensorflow but not safekit installed
cyberpath = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, cyberpath)

import tensorflow as tf
import numpy as np
import time
from safekit.batch import OnlineLMBatcher
from simple_lm import write_results, CELL
from safekit.tf_ops import lm_rnn, bidir_lm_rnn
from safekit.graph_training_utils import ModelRunner
from safekit.util import get_mask, Parser
import json
import math


def return_parser():
    parser = Parser()
    parser.add_argument('results_folder', type=str,
                        help='The folder to print results to.')
    parser.add_argument('config', type=str,
                        help='The data spec.')
    parser.add_argument("datafolder", type=str,
                        help="File with token features")
    parser.add_argument('-encoding', type=str, default=None,
                        help='Can be "oct", "raw" or "word"')
    parser.add_argument("-em", type=int, default=5,
                        help="Dimension of token embeddings")
    parser.add_argument("-numsteps", type=int, default=3,
                        help="length of unrolled context_rnn, number of log lines per user per train step")
    parser.add_argument('-mb', type=int, default=64,
                        help='Number of users in mini-batch.')
    parser.add_argument('-learnrate', type=float, default=0.001,
                        help='Step size for gradient descent.')
    parser.add_argument("-context_layers", type=int, nargs='+', default=[10],
                        help='List of hidden layer sizes for context lstm.')
    parser.add_argument('-lm_layers', type=int, nargs='+', default=[5],
                        help='List of hidden layer sizes for token lstm.')
    parser.add_argument('-debug', action='store_true',
                        help='Use this flag to print feed dictionary contents and dimensions.')
    parser.add_argument('-random_seed', type=int, default=5,
                        help='Random seed for reproducible experiments.')
    parser.add_argument('-jagged', action='store_true',
                        help='Whether using sequences of variable length (Input should'
                             'be zero-padded to max_sequence_length.')
    parser.add_argument('-skipsos', action='store_true',
                        help='Whether to skip a start of sentence token.') 
    parser.add_argument('-bidir', action='store_true',
                        help='Whether to use bidirectional lstm for lower tier.')
    parser.add_argument('-test', action='store_true',
                        help='Whether to run on a subset of the data (5000 lines from days 1,2,3) or the entire set.')
    parser.add_argument('-verbose', type=int, default=1,
                        help='Whether to print loss during training.')
    parser.add_argument('-delimiter', type=str, default=',',
                        help="Delimiter for input text file")
    parser.add_argument('-cell_type', type=str, default='lstm',
                        help='Can be either "lstm", "ident_ran", or "ran"')
    parser.add_argument('-upper_cell_type', type=str, default='lstm',
                        help='Can be either "lstm", "ident_ran", or "ran"')
    return parser


class ContextRNN:
    """
    Log line level LSTM cell that keeps track of it's last lstm state tuple
    """
    def __init__(self, layers, initial_state,
                 cell=tf.nn.rnn_cell.LSTMCell):
        """

        :param layers: List of hidden layer sizes.
        :param initial_state: List of numlayers lists of tensors (cell_state, hidden_state),
                              or List of lstm state tuples (which are named tuples of tensors (c=cell_state, h=hidden_state)
        :param cell: Type of rnn cell to use.
        """

        self.cell_type = cell
        self.cell_stack = tf.nn.rnn_cell.MultiRNNCell([self.cell_type(cell_size) for cell_size in layers])
        self.layers = layers
        self.state = initial_state

    def __call__(self, lower_outputs, final_hidden, seq_len):
        """

        :param line_input: The input for current time step.
        :param state: The cell state output by ContextRnn from previous time step.
        :param seq_len: A 1D tensor of of size mb giving lengths of sequences in mb for this time step
        :return: (tensor, LSTMStateTuple) output, state
        """
        ctxt_input = ContextRNN._create_input(lower_outputs, final_hidden, seq_len)
        output, self.state = self.cell_stack(ctxt_input, self.state)
        return output, self.state

    @staticmethod
    def _create_input(lower_outputs, final_hidden, seq_len):
        """

        :param lower_outputs:  The list of output Tensors from the token level rnn
        :param final_hidden: The final hidden state from the token level rnn
        :param seq_len: A 1D tensor of of size mb giving lengths of token level sequences in mb for this time step
        :return: A tensor which is the concatenation of the hidden state averages and final hidden state from lower
                 tier model. Used as input to context rnn
        """
        if seq_len is not None:
            mean_hidden = tf.reduce_sum(tf.stack(lower_outputs, axis=0), axis=0)/seq_len
        else:
            mean_hidden = tf.reduce_mean(tf.stack(lower_outputs, axis=0), axis=0)
        return tf.concat([mean_hidden, final_hidden], 1)


def tiered_lm(token_set_size, embedding_size, ph_dict, context_layers, lm_layers,
              numsteps, bidir=False, jagged=False):
    """
    :param token_set_size: (int) Number of unique tokens in token set
    :param embedding_size: (int) Dimensionality of token embeddings
    :param ph_dict: dictionary of tensorflow placeholders and lists of tensorflow placeholders
    :param context_layers: List of hidden layer sizes for stacked context LSTM
    :param lm_layers: list of hidden layer sizes for stacked sentence LSTM
    :param numsteps: How many steps (log lines) to unroll the upper tier RNN
    :param bidir: Whether to use bidirectional LSTM for lower tier model
    :param jagged: Whether or not variable length sequences are used
    :return: total_loss (scalar tensor),
             context_vector (tensor),
             line_loss_matrix (tensor), Losses for each line in mini-batch
             context_state (LSTMStateTuple) Final state of upper tier model
    """
    if bidir:
        language_model = bidir_lm_rnn
    else:
        language_model = lm_rnn
    # =========================================================
    # ========== initialize token level lstm variables ========
    # =========================================================
    if jagged:
        ph_dict['lens'] = []
        ph_dict['masks'] = []
    context_vector = tf.placeholder(tf.float32, [None, ctxt_size], name="context_vector")
    ph_dict['context_vector'] = context_vector
    tf.add_to_collection('context_vector', ph_dict['context_vector'])
    token_embed = tf.Variable(tf.truncated_normal([token_set_size, embedding_size]))  # Initial embeddings vocab X embedding size
    total_loss = 0.0
    # =========================================================
    # ======= initialize log line level (context) lstm ========
    # =========================================================
    ph_dict['c_state_init'] = [tf.placeholder(tf.float32, [None, c_size]) for c_size in context_layers]
    ph_dict['h_state_init'] = [tf.placeholder(tf.float32, [None, h_size]) for h_size in context_layers]
    context_init = [tf.nn.rnn_cell.LSTMStateTuple(ph_dict['c_state_init'][i],
                                                  ph_dict['h_state_init'][i])
                    for i in range(len(context_layers))]
    ctxt_rnn = ContextRNN(context_layers, context_init, cell=CELL[args.upper_cell_type])

    # =========================================================
    # ======= initiate loop that ties together tiered lstm ====
    # =========================================================
    with tf.variable_scope("reuse_scope") as vscope:
        for i in range(numsteps):
            x = tf.placeholder(tf.int64, [None, sentence_length])
            t = tf.placeholder(tf.int64, [None, sentence_length-2*bidir])
            ph_dict['x'].append(x)
            ph_dict['t'].append(t)
            if jagged:
                seq_len = tf.placeholder(tf.int32, [None])
                ph_dict['lens'].append(seq_len)
            else:
                seq_len = None
            token_losses, hidden_states, final_hidden = language_model(x, t, token_embed, lm_layers,
                                                                       seq_len=seq_len,
                                                                       context_vector=context_vector,
                                                                       cell=CELL[args.cell_type])

            if jagged:
                ph_dict['masks'].append(tf.placeholder(tf.float32, [None, sentence_length-2*bidir]))
                token_losses *= ph_dict['masks'][-1]
                line_losses = tf.reduce_sum(token_losses, axis=1)  # batch_size X 1
                sequence_lengths = tf.reshape(tf.cast(ph_dict['lens'][-1], tf.float32), (-1, 1))
            else:
                line_losses = tf.reduce_mean(token_losses, axis=1)  # batch_size X 1
                sequence_lengths = None
            avgloss = tf.reduce_mean(line_losses)  # scalar
            total_loss += avgloss

            if i == 0:
                line_loss_matrix = tf.reshape(line_losses, [1, -1])
                tf.add_to_collection('first_line_loss_matrix', line_loss_matrix)
            else:
                line_loss_matrix = tf.concat((line_loss_matrix, tf.reshape(line_losses, [1, -1])), 0)

            context_vector, context_state = ctxt_rnn(hidden_states,
                                                     final_hidden,
                                                     sequence_lengths)
            tf.add_to_collection('context_vector', context_vector)
            tf.add_to_collection('context_state', context_state)
            tf.get_variable_scope().reuse_variables()
    total_loss /= float(numsteps)
    return total_loss, context_vector, line_loss_matrix, context_state


if __name__ == "__main__":

    #  ===========================================================================
    #  =========================PARSE ARGUMENTS===================================
    #  ===========================================================================
    args = return_parser().parse_args()
    conf = json.load(open(args.config, 'r'))

    assert all(x == args.context_layers[0] for x in args.context_layers), 'Different sized context layers not supported.'
    assert args.numsteps > 1, 'Must have at least two upper tier time steps to build graph for tiered lstm.'

    if not args.results_folder.endswith('/'):
        args.results_folder += '/'
    tf.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)
    sentence_length = (conf['sentence_length'] - 1) - int(args.skipsos) + int(args.bidir)
    token_set_size = conf['token_set_size']

    ctxt_size = args.context_layers[0]
    direction = ('fwd', 'bidir')[args.bidir]
    results_file = 'tier_%s_%s_%s_%s__em_%s__ns_%s__mb_%s__lr_%s__cl_%s__lml_%s__rs_%s' % (direction,
                                                                                        args.encoding,
                                                                                        args.cell_type,
                                                                                        time.ctime(time.time()).replace(' ', '-'),
                                                                                        args.em,
                                                                                        args.numsteps,
                                                                                        args.mb,
                                                                                        args.learnrate,
                                                                                        args.context_layers[0],
                                                                                        args.lm_layers[0],
                                                                                        args.random_seed)
    # if the -test flag passed, store predictions in a temporary file
    if "lanl_results" not in os.listdir("/tmp"):                               
        os.system("mkdir /tmp/lanl_results; chmod g+rwx /tmp/lanl_results")
    outfile = open("/tmp/lanl_results/" + results_file, 'w')
    outfile.write("batch line second day user red loss\n")
    mode = ('fixed', 'update')
    jag = int(args.jagged)
    skipsos = int(args.skipsos)

    #  ===========================================================================
    #  =========================BUILD GRAPH=======================================
    #  ===========================================================================
    ph_dict = {'x': [], 't': []}
    dummy_loss = tf.constant(1)
    total_loss, context_vector, line_loss_matrix, context_state = tiered_lm(token_set_size, args.em,
                                                                            ph_dict,
                                                                            args.context_layers,
                                                                            args.lm_layers,
                                                                            args.numsteps,
                                                                            bidir=args.bidir,
                                                                            jagged=args.jagged)
    tiered_network_model = ModelRunner(total_loss, ph_dict, learnrate=args.learnrate,
                                       debug=args.debug, decay=True,
                                       decay_rate=0.99, decay_steps=20)

    #  ===========================================================================
    #  =========================TRAINING LOOP=====================================
    #  ===========================================================================
    init_triple = (np.zeros([1, ctxt_size], np.float32), # context
                   [np.zeros([1, c_size], np.float32) for c_size in args.context_layers], # state
                   [np.zeros([1, h_size], np.float32) for h_size in args.context_layers]) # hidden

    start_time = time.time()

    def trainday(is_training, f, states, logs):
        num_processed = 0
        data = OnlineLMBatcher(args.datafolder + f, init_triple,
                               batch_size=args.mb, num_steps=args.numsteps, skiprows=0)
        do_update = is_training
        if states is not None:
            data.state_triples = states
        batch, state_triple = data.next_batch()
        batch_num = 0
        stragglers = False
        while batch is not None:
            if data.flush:
                do_update = False
            if len(batch.shape) == 2:  # Straggler log lines that don't fit into num_steps by end of day are run in large batches one step at a time
                stragglers = True
                batch = batch.reshape((1, batch.shape[0], batch.shape[1]))
                endx = batch.shape[2] - int(not args.bidir)
                endt = batch.shape[2] - int(args.bidir)
                datadict = {'line': batch[:, :, 0],
                            'second': batch[:, :, 1],
                            'day': batch[:, :, 2],
                            'user': batch[:, :, 3],
                            'red': batch[:, :, 4],
                            'x': [batch[0, :, 5 + jag + skipsos:endx]] * args.numsteps,
                            't': [batch[0, :, 6 + jag + skipsos:endt]] * args.numsteps,
                            'context_vector': state_triple['context_vector'],
                            'c_state_init': state_triple['c_state_init'],
                            'h_state_init': state_triple['h_state_init']}

                if args.jagged:
                    datadict['lens'] = [batch[0, :, 5] - skipsos] * args.numsteps
                    datadict['masks'] = [get_mask(seq_length - 2 * args.bidir, sentence_length - 2 * args.bidir) for
                                         seq_length in datadict['lens']]

                    for i in range(len(datadict['x'])):
                        assert np.all(datadict['lens'][i] <= datadict['x'][i].shape[1]), \
                            'Sequence found greater than num_tokens_predicted'
                        assert np.nonzero(datadict['lens'][i])[0].shape[0] == datadict['lens'][i].shape[0], \
                            'Sequence lengths must be greater than zero.' \
                            'Found zero length sequence in datadict["lengths"]: %s' % datadict['lens']
                first_output_context_state = tf.get_collection('context_state')[0]
                eval_tensors = ([total_loss,
                                 tf.get_collection('context_vector')[1],
                                 tf.get_collection('first_line_loss_matrix')[0]] +
                                 [state_tuple.c for state_tuple in first_output_context_state] +
                                 [state_tuple.h for state_tuple in first_output_context_state])
            else:  # Ordinary batching and matrix flush batching
                batch = np.transpose(batch, axes=(1, 0, 2))
                endx = batch.shape[2] - int(not args.bidir)
                endt = batch.shape[2] - int(args.bidir)
                datadict = {'line': batch[:, :, 0],
                            'second': batch[:, :, 1],
                            'day': batch[:, :, 2],
                            'user': batch[:, :, 3],
                            'red': batch[:, :, 4],
                            'x': [batch[i, :, 5 + jag + skipsos:endx] for i in range(args.numsteps)],
                            't': [batch[i, :, 6 + jag + skipsos:endt] for i in range(args.numsteps)],
                            'context_vector': state_triple['context_vector'],
                            'c_state_init': state_triple['c_state_init'],
                            'h_state_init': state_triple['h_state_init']}

                if args.jagged:
                    datadict['lens'] = [batch[i, :, 5] - skipsos for i in range(args.numsteps)]
                    datadict['masks'] = [get_mask(seq_length-args.bidir-args.skipsos,
                                                  sentence_length-2*args.bidir) for seq_length in datadict['lens']]
                    for i in range(len(datadict['x'])):
                        assert np.all(datadict['lens'][i] <= datadict['x'][i].shape[1]), \
                            'Sequence found greater than num_tokens_predicted'
                        assert np.nonzero(datadict['lens'][i])[0].shape[0] == datadict['lens'][i].shape[0], \
                            'Sequence lengths must be greater than zero.' \
                            'Found zero length sequence in datadict["lengths"]: %s' % datadict['lens']

                eval_tensors = ([total_loss, context_vector, line_loss_matrix] +
                                [state_tuple.c for state_tuple in context_state] +
                                [state_tuple.h for state_tuple in context_state])

            # output dims: 0: Nothing, 1 (total_loss): scalar, 2 (context_vector): num_users X hidden_size,
            # 3 (line_loss_matrix): num_users X num_steps
            output = tiered_network_model.train_step(datadict, eval_tensors=eval_tensors,
                                                     update=do_update)
            loss, context, loss_matrix = output[1], output[2], output[3]
            current_context_state = output[4:4 + len(args.context_layers)]
            current_context_hidden = output[4 + len(args.context_layers):4 + 2*len(args.context_layers)]
            data.update_state_triples([context, current_context_state, current_context_hidden])
            if args.verbose:
                print('%s %s %s %s %s %s %r' % (datadict['day'].shape[1],
                                                datadict['line'][0][0],
                                                datadict['second'][0][0],
                                                mode[do_update],
                                                f,
                                                data.line_num, loss))
            if math.isnan(loss) or math.isinf(loss):
                print('Exiting due to divergence!')
                exit(1)
            if not is_training:
                num_processed += batch.shape[0] * batch.shape[1]
                if not stragglers:
                    assert loss_matrix.shape[0] * loss_matrix.shape[1] == batch.shape[0] * batch.shape[1], 'Batch size %s is different from output size %s. May be losing datapoints.' % (batch.shape, loss_matrix.shape)
                    write_results(datadict, loss_matrix, outfile, batch_num)
                else:
                    assert loss_matrix[0, :].shape[0] == batch.shape[0] * batch.shape[1], 'Batch size is different from output size. May be losing datapoints.'
                    write_results(datadict, loss_matrix[0, :], outfile, batch_num)
            batch, state_triple = data.next_batch()
            batch_num += 1

        return data.state_triples, data.user_logs, num_processed

    weekend_days = conf["weekend_days"]
    if args.test:
        files = conf["test_files"]  # 5000 lines from each of day 0, day 1 and day 2
    else:
        files = [str(i) + '.txt' for i in range(conf["num_days"]) if i not in weekend_days]
    states1 = None
    logs1 = None
    number_processed = 0
    for idx, f in enumerate(files[:-1]):
        states1, logs1, num_processed = trainday(True, f, states1, logs1)
        states2, logs2, num_processed = trainday(False, files[idx + 1], states1, logs1)
        number_processed += num_processed
    outfile.close()
    total_time = time.time() - start_time
    print('elapsed time: %s' % total_time)
    os.system("mv /tmp/lanl_results/%s %s" % (results_file, args.results_folder + results_file))
    print('number processed', number_processed)
