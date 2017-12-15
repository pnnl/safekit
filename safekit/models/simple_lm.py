"""
Simple language model code for network modeling.

File name convention:
---------------------

- lr: learnrate for gradient descent
- nl: number of stacked lstm layers
- hs: size of hidden layers (presumes all layers have same number of hidden units)
- mb: Size of mini-batch
- bc: max bad count for early stopping
- em: Size of token embeddings
- rs: Random seed for reproducible results

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

File output:
------------
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


Example calls
-------------

**Simple character based language model.** ::

    python safekit/models/simple_lm.py results/ safekit/features/specs/lm/lanl_char_config.json data_examples/lanl/lm_feats/raw_day_split/ -test -skipsos -jagged

.. Note :: The output results will be printed to /tmp/lanl_result/ and then moved to results/ upon completion
      to avoid experiment slowdown of constant network traffic when using a distributed file system.

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
import json
from safekit.batch import OnlineBatcher
from safekit.graph_training_utils import EarlyStop, ModelRunner
from safekit.tf_ops import lm_rnn, bidir_lm_rnn
from safekit.util import get_mask, Parser
import time
import numpy as np


def return_parser():
    """
    Defines and returns argparse ArgumentParser object.

    :return: ArgumentParser
    """
    parser = Parser("Simple token based rnn for network modeling.")
    parser.add_argument('results_folder', type=str,
                        help='The folder to print results to.')
    parser.add_argument('config', type=str,
                        help='The data spec.')
    parser.add_argument("datafolder", type=str,
                        help='The folder where the data is stored.')
    parser.add_argument('-learnrate', type=float, default=0.001,
                        help='Step size for gradient descent.')
    parser.add_argument("-lm_layers", nargs='+', type=int, default=[10],
                        help="A list of hidden layer sizes.")
    parser.add_argument("-context_layers", nargs='+', type=int, default=[10],
                        help="decoy arg.")
    parser.add_argument("-numsteps", type=int, default=10,
                        help="decoy arg.")
    parser.add_argument('-mb', type=int, default=128,
                        help='The mini batch size for stochastic gradient descent.')
    parser.add_argument('-debug', action='store_true',
                        help='Use this flag to print feed dictionary contents and dimensions.')
    parser.add_argument('-maxbadcount', type=str, default=20,
                        help='Threshold for early stopping.')
    parser.add_argument('-em', type=int, default=20,
                        help='Size of embeddings for categorical features.')
    parser.add_argument('-encoding', type=str, default=None,
                        help='Can be "oct", "raw" or "word"')
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
    parser.add_argument('-verbose', type=int, default=1, help='Whether to print loss during training.')
    parser.add_argument('-delimiter', type=str, default=',',
                        help="Delimiter for input text file. You should be using ' ' for the dayshuffled cert.")
    parser.add_argument('-cell_type', type=str, default='lstm',
                        help='Can be either "lstm", "ident_ran", or "ran"')

    return parser


def write_results(datadict, pointloss, outfile, batch):
    """
    Writes loss for each datapoint, along with meta-data to file.

    :param datadict: Dictionary of data names (str) keys to numpy matrix values for this mini-batch.
    :param pointloss: MB X 1 numpy array
    :param outfile: Where to write results.
    :param batch: The mini-batch number for these events.
    :return:
    """

    for line, sec, day, usr, red, loss in zip(datadict['line'].flatten().tolist(),
                                              datadict['second'].flatten().tolist(),
                                              datadict['day'].flatten().tolist(),
                                              datadict['user'].flatten().tolist(),
                                              datadict['red'].flatten().tolist(),
                                              pointloss.flatten().tolist()):
        outfile.write('%s %s %s %s %s %s %r\n' % (batch, line, sec, day, usr, red, loss))

CELL = {'lstm': tf.nn.rnn_cell.BasicLSTMCell}

if __name__ == '__main__':

    args = return_parser().parse_args()


    direction = ('fwd', 'bidir')[args.bidir]
    outfile_name = "simple_%s_%s_%s_%s_lr_%s_nl_%s_hs_%s_mb_%s_bc_%s_em_%s_rs_%s" % (direction,
                                                                                  args.encoding,
                                                                                  args.cell_type,
                                                                                  time.ctime(time.time()).replace(' ', '-'),
                                                                                  args.learnrate,
                                                                                  len(args.lm_layers),
                                                                                  args.lm_layers[0],
                                                                                  args.mb,
                                                                                  args.maxbadcount,
                                                                                  args.em,
                                                                                  args.random_seed)
    args = return_parser().parse_args()
    conf = json.load(open(args.config, 'r'))
    if not args.results_folder.endswith('/'):
        args.results_folder += '/'
    tf.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if not args.datafolder.endswith('/'):
        args.datafolder += '/'
    if "lanl_result" not in os.listdir("/tmp"):
        os.system("mkdir /tmp/lanl_result; chmod o+rwx /tmp/lanl_result")
    if not args.bidir:
        language_model = lm_rnn
    else:
        language_model = bidir_lm_rnn
    outfile = open('/tmp/lanl_result/' + outfile_name, 'w')
    tf.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)
    sentence_length = conf["sentence_length"]- 1 - int(args.skipsos) + int(args.bidir)
    token_set_size = conf["token_set_size"]
    x = tf.placeholder(tf.int32, [None, sentence_length])
    t = tf.placeholder(tf.int32, [None, sentence_length-2*args.bidir])
    ph_dict = {'x': x, 't': t}
    if args.jagged:
        seq_len = tf.placeholder(tf.int32, [None])
        ph_dict['lengths'] = seq_len
    else:
        seq_len = None
    token_embed = tf.Variable(tf.truncated_normal([token_set_size, args.em]))  # Initial embeddings vocab X embedding size

    token_losses, hidden_states, final_hidden = language_model(x, t, token_embed,
                                                               args.lm_layers, seq_len=seq_len,
                                                               cell=CELL[args.cell_type])
    if args.jagged:
        ph_dict['mask'] = tf.placeholder(tf.float32, [None, sentence_length-2*args.bidir])
        token_losses *= ph_dict['mask']
        line_losses = tf.reduce_sum(token_losses, axis=1)  # batch_size X 1
    else:
        line_losses = tf.reduce_mean(token_losses, axis=1)  # batch_size X 1
    avgloss = tf.reduce_mean(line_losses)  # scalar

    model = ModelRunner(avgloss, ph_dict, learnrate=args.learnrate, debug=args.debug,
                        decay=True,
                        decay_rate=0.99, decay_steps=20)

    # training loop
    start_time = time.time()
    jag = int(args.jagged)
    skipsos = int(args.skipsos)

    def trainday(is_training, f):
        batch_num = 0
        data = OnlineBatcher(args.datafolder + f, args.mb, delimiter=' ')
        raw_batch = data.next_batch()
        current_loss = sys.float_info.max
        not_early_stop = EarlyStop(args.maxbadcount)
        endx = raw_batch.shape[1] - int(not args.bidir)
        endt = raw_batch.shape[1] - int(args.bidir)
        continue_training = not_early_stop(raw_batch, current_loss)
        while continue_training:  # mat is not None and self.badcount < self.badlimit and loss != inf, nan:
            datadict = {'line': raw_batch[:, 0],
                        'second': raw_batch[:, 1],
                        'day': raw_batch[:, 2],
                        'user': raw_batch[:, 3],
                        'red': raw_batch[:, 4],
                        'x': raw_batch[:, (5+jag+skipsos):endx],
                        't': raw_batch[:, (6+jag+skipsos):endt]}
            if args.jagged:
                datadict['lengths'] = raw_batch[:, 5]
                datadict['mask'] = get_mask(datadict['lengths']-2*args.bidir-args.skipsos, sentence_length-2*args.bidir)
                assert np.all(datadict['lengths'] <= x.get_shape().as_list()[1]), 'Sequence found greater than num_tokens_predicted'
                assert np.nonzero(datadict['lengths'])[0].shape[0] == datadict['lengths'].shape[0], \
                    'Sequence lengths must be greater than zero.' \
                    'Found zero length sequence in datadict["lengths"]: %s' % datadict['lengths']
            eval_tensors = [avgloss, line_losses]
            _, current_loss, pointloss = model.train_step(datadict, eval_tensors,
                                                              update=is_training)
            if not is_training:
                write_results(datadict, pointloss, outfile, batch_num)
            batch_num += 1
            if args.verbose:
                print('%s %s %s %s %s %s %r' % (raw_batch.shape[0],
                                                datadict['line'][0],
                                                datadict['second'][0],
                                                ('fixed', 'update')[is_training],
                                                f,
                                                data.index,
                                                current_loss))
            raw_batch = data.next_batch()
            continue_training = not_early_stop(raw_batch, current_loss)
            if continue_training < 0:
                exit(0)


    weekend_days = conf["weekend_days"]
    if args.test:
        files = conf["test_files"] # 5000 lines from each of day 0, day 1 and day 2
    else:
        files = [str(i) + '.txt' for i in range(conf["num_days"]) if i not in weekend_days]
    outfile.write("batch line second day user red loss\n")

    for idx, f in enumerate(files[:-1]):
        trainday(True, f)
        trainday(False, files[idx + 1])
    outfile.close()
    total_time = time.time() - start_time
    print('elapsed time: %s' % total_time)
    os.system("mv /tmp/lanl_result/%s %s" % (outfile_name, args.results_folder + outfile_name))
