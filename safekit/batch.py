"""
Module for mini-batching data.
"""
# TODO: Make skipping header argument consistent (numpy style skiprows) for all batchers.
# TODO: Make arguments for all batchers as consistent as possible.
# TODO: Look at Replay batcher and try to fix behavior of replay DNN. If fixed combine replay batcher with OnlineBatcher.
# TODO: StateTrackingBatcher - needs additional checking and commenting

import numpy as np
import random
import math
from collections import deque


class DayBatcher:
    """
    Gives batches from a csv file on a per day basis. The first field is assumed to be the day field.
    Days are assumed to be sorted in ascending order (No out of order days in csv file).
    For batching data too large to fit into memory. Written for one pass on data!!!
    """

    def __init__(self, datafile, skiprow=0, delimiter=','):
        """
        :param datafile: (str) File to read lines from.
        :param skiprow: (int) How many lines to ignore at beginning of file (e.g. if file has a header)
        :param delimiter: (str) The delimiter for the csv file
        """
        self.f = open(datafile, 'r')
        self.delimiter = delimiter
        for i in range(skiprow):
            self.f.readline()
        self.current_line = self.f.readline()
        self.current_day = -1

    def next_batch(self):
        """
        :return: (np.array) shape=(num_rows_in_a_day, len(csv_lines)).
                 Until end of datafile, each time called,
                 returns 2D array of consecutive lines with same day stamp.
                 Returns None when no more data is available (one pass batcher!!).
        """
        matlist = []
        if self.current_line == '':
            return None
        rowtext = np.array([float(k) for k in self.current_line.strip().split(self.delimiter)])
        self.current_day = rowtext[0]
        while rowtext[0] == self.current_day:
            self.current_day = rowtext[0]
            matlist.append(rowtext)
            self.current_line = self.f.readline()
            if self.current_line == '':
                break
            rowtext = np.array([float(k) for k in self.current_line.strip().split(self.delimiter)])

        return np.array(matlist)


class OnlineBatcher:
    """
    Gives batches from a csv file.
    For batching data too large to fit into memory. Written for one pass on data!!!
    """

    def __init__(self, datafile, batch_size,
                 skipheader=False, delimiter=',',
                 alpha=0.5, size_check=None,
                 datastart_index=3, norm=False):
        """

        :param datafile: (str) File to read lines from.
        :param batch_size: (int) Mini-batch size.
        :param skipheader: (bool) Whether or not to skip first line of file.
        :param delimiter: (str) Delimiter of csv file.
        :param alpha: (float)  For exponential running mean and variance.
                      Lower alpha discounts older observations faster.
                      The higher the alpha, the further you take into consideration the past.
        :param size_check: (int) Expected number of fields from csv file. Used to check for data corruption.
        :param datastart_index: (int) The csv field where real valued features to be normalized begins.
                                Assumed that all features beginnning at datastart_index till end of line
                                are real valued.
        :param norm: (bool) Whether or not to normalize the real valued data features.
        """

        self.alpha = alpha
        self.f = open(datafile, 'r')
        self.batch_size = batch_size
        self.index = 0
        self.delimiter = delimiter
        self.size_check = size_check
        if skipheader:
            self.header = self.f.readline()
        self.datastart_index = datastart_index
        self.norm = norm
        self.replay = False

    def next_batch(self):
        """
        :return: (np.array) until end of datafile, each time called,
                 returns mini-batch number of lines from csv file
                 as a numpy array. Returns shorter than mini-batch
                 end of contents as a smaller than batch size array.
                 Returns None when no more data is available(one pass batcher!!).
        """
        matlist = []
        l = self.f.readline()
        if l == '':
            return None
        rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
        if self.size_check is not None:
            while len(rowtext) != self.size_check:
                l = self.f.readline()
                if l == '':
                    return None
                rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
        matlist.append(rowtext)
        for i in range(self.batch_size - 1):
            l = self.f.readline()
            if l == '':
                break
            rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
            if self.size_check is not None:
                while len(rowtext) != self.size_check:
                    l = self.f.readline()
                    if l == '':
                        return None
                    rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
            matlist.append(rowtext)
        data = np.array(matlist)
        if self.norm:
            batchmean, batchvariance = data[:,self.datastart_index:].mean(axis=0), data[:, self.datastart_index:].var(axis=0)
            if self.index == 0:
                self.mean, self.variance = batchmean, batchvariance
            else:
                self.mean = self.alpha * self.mean + (1 - self.alpha) * batchmean
                self.variance = self.alpha * self.variance + (1 - self.alpha) * batchvariance
                data[:, self.datastart_index:] = (data[:, self.datastart_index:] - self.mean)/(self.variance + 1e-10)
        self.index += self.batch_size
        return data


def split_batch(batch, spec):
    """
    Splits numpy matrix into separate data fields according to spec dictionary.

    :param batch: (np.array) Array with shape=(batch_size, num_features) of data collected from stream.
    :param spec: (dict) A python dict containing information about which indices in the incoming data point correspond to which features.
                  Entries for continuous features list the indices for the feature, while entries for categorical features
                  contain a dictionary- {'index': i, 'num_classes': c}, where i and c are the index into the datapoint,
                  and number of distinct categories for the category in question.
    :return: (dict) A dictionary of numpy arrays of the split 2d feature array.
    """
    assert spec['num_features'] == batch.shape[1], "Wrong number of features: spec/%s\tbatch/%s" % (spec['num_features'], batch.shape[1])
    datadict = {}
    for dataname, value in spec.iteritems():
        if dataname != 'num_features':
            if value['num_classes'] > 0:
                datadict[dataname] = batch[:, value['index'][0]].astype(int)
            else:
                datadict[dataname] = batch[:, value['index']]
    return datadict


class StateTrackingBatcher:
    """
    Aggregate RNN batcher. Reads line by line from a file or pipe being fed csv format features by a feature extractor.
    Keeps track of window of user history for truncated backpropagation through time with a shifting set of users.
    """

    def __init__(self, pipe_name,
                 specs,
                 batchsize=21,
                 num_steps=3,
                 layers=(10),
                 replay_ratio=(1, 0),
                 next_step=False,
                 warm_start_state=None,
                 delimiter=',',
                 skipheader=False,
                 alpha=0.5,
                 datastart_index=3,
                 standardize=True):
        """

        :param pipe_name: (str) Name of file or pipe to read from.
        :param specs: (dict) From a json specification of the purpose of fields in the csv input file (See docs for formatting)
        :param batchsize: (int) The maximum number of events in a mini-batch
        :param num_steps: (int) The maximum number of events for any given user per mini-batch (window size sort of)
        :param layers:  (list) A list of the sizes of hidden layers for the stacked lstm.
        :param replay_ratio: (tuple) (num_new_batches, num_replay_batches) Describes the ratio of new batches to replay batches.
        :param next_step: (boolean) False (0) if autoencoding, True (1) if next time step prediction
        :param warm_start_state: (tuple) Tuple of numpy arrays for warm_starting state of all users of RNN.
        :param delimiter: (str) Delimiter of csv file.
        :param skipheader: (bool) Whether or not to skip first line of csv file.
        :param alpha: (float)  For exponential running mean and variance.
                      Lower alpha discounts older observations faster.
                      The higher the alpha, the further you take into consideration the past.
        :param datastart_index: (int) The csv field where real valued features to be normalized begins.
                                Assumed that all features beginnning at datastart_index till end of line
                                are real valued.
        :param standardize: (bool) Whether or not to standardize the data using running mean and variance.
        """
        self.specs = specs
        self.batch_limit = batchsize
        self.num_steps = num_steps
        self.pipe_name = pipe_name
        self.pipein = open(pipe_name, 'r')
        self.pipein.readline()
        self.event_map = {}
        self.state_map = {}
        self.new_event_count = {}
        self.layers = layers
        self.event_number = 0
        self.event_deque_size = int(next_step) + self.num_steps
        self.next_step = next_step
        self.finished = False
        self.num_features = self.specs.pop('num_features', None)
        self.user_index = self.specs['user']['index'][0]
        self.count_start = self.specs['counts']['index'][0]
        self.warm_start_state = warm_start_state
        self.period = sum(replay_ratio)
        self.batch_function = [self.new_batch if i < replay_ratio[0] else self.replay_batch for i in range(self.period)]
        self.replay_indicator = [False if i < replay_ratio[0] else True for i in range(self.period)]
        self.delimiter = delimiter
        self.mod = 0
        self.day = 0
        self.alpha = alpha
        self.index = 0
        self.datastart_index = datastart_index
        self.standardize = standardize
        if skipheader:
            self.header = self.pipein.readline()

    @property
    def replay(self):
        """
        Whether or not a replay batch was just processed.
        """
        return self.replay_indicator[self.mod]

    def next_batch(self):
        """

        :return: (dict) A dictionary of numpy arrays from splitting a
                  3d (numsteps X mb_size X num_csv_fields) array into subarrays with keys pertaining to use in training.
        """
        if self.day < 20:
            return self.new_batch()
        else:
            batch = self.batch_function[self.mod]()
            self.mod = (self.mod + 1) % self.period
            return batch

    def package_data(self, batch):
        """

        :param batch: (np.array) An assembled 3 way array of data collected from the stream with shape (num_time_steps, num_users, num_features)
        :return: (dict) A dictionary of numpy arrays of the diced 3way feature array.
        """
        datadict = {}
        for dataname, value in self.specs.iteritems():
            if value['num_classes'] != 0:  # type(value) is dict:
                datadict[dataname] = batch[:, :, value['index'][0]].astype(int)
            else:
                datadict[dataname] = batch[:, :, value['index']]
        return datadict

    def blank_slate(self):
        """
        Creates and returns a zero state for one time step for 1 user

        :return: (list) A list of 1 X state_size numpy arrays the number of layers long
        """
        if self.warm_start_state is not None:
            return self.warm_start_state
        return np.stack([np.stack([np.random.normal(scale=0.1, size=(1, units)),
                                   np.random.normal(scale=0.1, size=(1, units))]) for units in self.layers])

    def avg_state(self):
        """
        :return: (list) The average of all the most recent states for each batch entity.
        """
        avg = self.blank_state
        for user, dec in self.state_map.iteritems():
            avg = [[a[0] + b[0], a[1] + b[1]] for a, b in zip(dec[-1], avg)]
        return [[a[0] / float(len(self.state_map)), a[1] / float(len(self.state_map))] for a in avg]

    def event_padding_random(self, rowtext):
        """
        Creates and returns a 'non-event' with random entries for event history padding of newly encountered user.

        :param rowtext: (int) A log line for the user
        :return: (np.array) A random event with user meta data attached.
        """
        meta = rowtext[:self.count_start]
        num_zeros = (len(rowtext) / 4) * 3
        zeros = np.zeros(num_zeros)
        vals = np.random.randint(1, high=30, size=len(rowtext) - num_zeros - self.count_start)
        zero_vals = np.concatenate([zeros, vals])  # len(rowtext) - self.count_start
        np.random.shuffle(zero_vals)  # len(rowtext) - self.count_start
        return np.concatenate((meta, zero_vals))  # len(rowtext)

    def get_new_events(self):
        """
        To get new events when not replaying old events.

        :returns: (int) 1 if not EOF 0 if EOF
        """
        if self.finished:
            return 0
        max_user_event_count = 0
        event_count = 0
        self.new_event_count = {}
        matlist = []
        while (event_count < self.batch_limit and
                       max_user_event_count < self.num_steps):
            rowtext = self.pipein.readline()[:-1].strip().split(self.delimiter)
            if rowtext[-1] == '':
                self.finished = True
                break
            assert len(rowtext) == self.num_features, 'Discrepancy in number of features of event %s. \n ' \
                                                      'Expected %s, got %s. \nFields: %r' % (self.event_number,
                                                                                             self.num_features,
                                                                                             len(rowtext),
                                                                                             rowtext)
            event_count += 1
            user = int(float(rowtext[self.user_index]))
            self.day = float(rowtext[0])

            if user not in self.new_event_count:
                self.new_event_count[user] = 1
            else:
                self.new_event_count[user] += 1
                max_user_event_count = max(self.new_event_count[user], max_user_event_count)

            try:
                rowtext = [float(entry) for entry in rowtext]
            except ValueError:
                raise ValueError('Non numeric string found in event %s' % self.event_number)
            matlist.append(rowtext)
        data = np.array(matlist)

        if self.standardize:
            batchmean, batchvariance = data[:, self.datastart_index:].mean(axis=0), data[:, self.datastart_index:].var(
                axis=0)
            if self.index == 0:
                self.mean, self.variance = batchmean, batchvariance
            else:
                self.mean = self.alpha * self.mean + (1 - self.alpha) * batchmean
                self.variance = self.alpha * self.variance + (1 - self.alpha) * batchvariance
            self.index += data.shape[0]
            data[:, self.datastart_index:] = (data[:, self.datastart_index:] - self.mean) / (self.variance + 1e-10)

        for rowtext in data:
            user = int(float(rowtext[self.user_index]))
            if user not in self.event_map:
                self.event_map[user] = deque(self.event_padding_random(rowtext)
                                             for i in range(self.event_deque_size))
                self.state_map[user] = deque(self.blank_slate()
                                             for i in range(self.num_steps + 1))

            self.event_map[user].append(rowtext)
            self.event_map[user].popleft()
            self.state_map[user].popleft()
            self.event_number += 1
        return 1

    def get_states(self):
        """
        Fetches the saved RNN states of users in current mini-batch

        :return: (list) List of user states.
        """
        state_batch = np.concatenate([self.state_map[user][self.next_step] for user in self.new_event_count], axis=2)
        state_batch = state_batch.reshape(len(self.layers) * 2, len(self.new_event_count.keys()), self.layers[0])
        return [state_batch[k, :, :] for k in range(len(self.layers) * 2)]

    def get_events(self):
        """

        :return: (np.array)  3 way array of shape (num_time_steps, num_users, num_features)
        """
        eventlist = [list(self.event_map[user]) for user in self.new_event_count]
        return np.array([[user_event_list[tme] for user_event_list in eventlist]
                         for tme in range(self.event_deque_size)])

    def make_key_map(self):
        """

        :return: (dict) For use in get_eval_indices.
        """
        return {user_index: number for user_index, number in
                zip(self.new_event_count.keys(), range(len(self.new_event_count)))}

    def get_eval_indices(self, key_map):
        """

        :param key_map: (dict)
        :return: (list) Data structure which keeps track of where to evaluate RNN.
        """
        # Reverse the order of eval_indices to lookup correct hidden state in rnn output
        return [np.array([key_map[key] for key in self.new_event_count if self.new_event_count[key] > val])
                for val in range(self.num_steps - 1, -1, -1)]

    def new_batch(self):
        """

        :return: (dict) A dictionary with keys to match to placeholders and values of numpy matrices. Entries are described as follows:

                - **states** A structured list of numpy arrays to feed as initial state for next round of training
                - **inputs** A three way numpy array of dimensions (timestep X user X (feature_size + target_size + meta_size)) where meta-size is the number of fields not used in training (user_id, timestamp, etc.)
                - **eval_indices** A num_time_steps long list of numpy vectors which contain the indices of hidden state outputs to evaluate on for each time step in this batch of training.
                - **Other entries** are split from the 'inputs' matrix using the specs dictionary which describes indices of matrices to extract.
        """
        if self.get_new_events() == 0:
            return None

        events = self.get_events()
        key_map = self.make_key_map()
        eval_indices = self.get_eval_indices(key_map)

        if self.next_step:
            eval_indices = eval_indices[1:]

        datadict = self.package_data(events)
        datadict['eval_indices'] = eval_indices
        datadict['initial_state'] = self.get_states()  # list(itertools.chain.from_iterable(states))
        return datadict

    def replay_batch(self):
        """

        :return: (dict) A dictionary with keys to match to placeholders and values of numpy matrices. Entries are described as follows:

                - **states** A structured list of numpy arrays to feed as initial state for next round of training
                - **inputs** A three way numpy array of dimensions (timestep X user X (feature_size + target_size + meta_size)) where meta-size is the number of fields not used in training (user_id, timestamp, etc.)
                - **eval_indices** A num_time_steps long list of numpy vectors which contain the indices of hidden state outputs to evaluate on for each time step in this batch of training.
                - **Other entries** are split from the 'inputs' matrix using the specs dictionary which describes indices of matrices to extract.
        """
        users = list(self.event_map.keys())
        random.shuffle(users)
        users = users[
                :int(math.ceil(float(self.batch_limit) / float(self.num_steps)))]  # numusers * numsteps = batchlimit
        self.new_event_count = {user: self.num_steps for user in users}
        events = self.get_events()
        key_map = self.make_key_map()
        eval_indices = self.get_eval_indices(key_map)

        if self.next_step:
            eval_indices = eval_indices[1:]

        datadict = self.package_data(events)
        datadict['eval_indices'] = eval_indices
        datadict['initial_state'] = self.get_states()
        return datadict

    def update_states(self, states):
        """
        For updating the deque of lstm states for each user after a minibatch of training.

        :param states: (list) The unstructured list of state matrices evaluated after a train step.
        """
        # states handed to last batch that we want to preserve for popleft rule to work

        last_states = np.concatenate([self.state_map[user][0] for user in self.new_event_count], axis=2)
        new_states = np.array(states).reshape(
            [self.num_steps, len(self.layers), 2, last_states.shape[2], self.layers[0]])
        new_states = np.concatenate([np.expand_dims(last_states, axis=0), new_states],
                                    axis=0)  # numsteps X layers X 2 X user_mb+1 X units
        new_states = np.split(new_states, new_states.shape[3], axis=3)
        for idx, user in enumerate(self.new_event_count):
            self.state_map[user] = deque([new_states[idx][t, :, :, :, :] for t in range(self.num_steps + 1)])


class OnlineLMBatcher:
    """
    For use with tiered_lm.py. Batcher keeps track of user states in upper tier RNN.
    """
    def __init__(self, datafile, initial_state_triple,
                 batch_size=100, num_steps=5, delimiter=" ",
                 skiprows=0):
        """

        :param datafile: (str) CSV file to read data from.
        :param initial_state_triple: (tuple) Initial state for users in lstm.
        :param batch_size: (int) How many users in a mini-batch.
        :param num_steps: (int) How many log lines to get for each user.
        :param delimiter: (str) delimiter for csv file.
        :param skiprows: (int) How many rows to skip at beginning of csv file.
        """
        self.user_count = 15000  # number of users in population
        self.delimiter = delimiter  # delimiter for input file
        self.mb_size = batch_size  # the number of users in a batch
        self.num_steps = num_steps  # The number of log lines for each user in a batch
        self.user_logs = [deque() for i in range(self.user_count)]  # list lists of loglines for each user. an individual user log line list
        # has length between 0 and self.mb_size - 1. When a user log line list
        # becomes self.mb_size it is transformed into np.array and moved to either
        # the current batch, or self.user_batch_overflow
        self.user_batch_overflow = []  # To store num_steps matrices of log lines for high frequency users
        self.state_triples = [initial_state_triple] * self.user_count  # lstm state for each user for top tier language model
        self.data = open(datafile, 'r')
        self.batch_user_list = []  # A record of all the users in a batch for retrieving and updating states
        for i in range(skiprows):
            garbage = self.data.readline()
        self.line_num = 1  # The line number of the file to be read next
        self.flush = False  # used by next_batch() to decide whether to call flush_batch()

    def update_state_triples(self, new_triples):
        """
        Called after training step of RNN to save current states of users.

        :param new_triples: (3-tuple) context_list = np.array shape=(users X context_rnn_hidden_size)
                                      state_list = list of np.arrays of shape=(users X context_rnn_hidden_size)
                                      hidden_list = Same type as state list

        """

        context_list = np.split(new_triples[0], new_triples[0].shape[0], axis=0)  # split on user dimension
        state_list = np.split(np.array(new_triples[1]), len(context_list), axis=1)  # split on user dimension
        hidden_list = np.split(np.array(new_triples[2]), len(context_list), axis=1)  # split on user dimension

        for idx, user in enumerate(self.batch_user_list):
            self.state_triples[int(user)] = (context_list[idx], state_list[idx], hidden_list[idx])

    def get_state_triples(self):
        """

        :return: (dict) Current states of users for all users in this mini-batch.
        """
        context_list = [None] * len(self.batch_user_list)
        state_list = [None] * len(self.batch_user_list)
        hidden_list = [None] * len(self.batch_user_list)
        for idx, user in enumerate(self.batch_user_list):
            context_list[idx], state_list[idx], hidden_list[idx] = self.state_triples[int(user)]

        state_matrix = np.concatenate(state_list, axis=1)
        hidden_matrix = np.concatenate(hidden_list, axis=1)

        return {'context_vector': np.concatenate(context_list, axis=0),  # users X context_rnn_hidden_size
                'c_state_init': [state_matrix[layer, :, :] for layer in range(state_matrix.shape[0])],
                # list of users X context_rnn_hidden_size
                'h_state_init': [hidden_matrix[layer, :, :] for layer in range(hidden_matrix.shape[0])]
                # list of users X context_rnn_hidden_size
                }

    def next_batch(self):
        """

        :return: (tuple) (batch, state_triples) Where batch is a three way array and state_triples contains current user
                 states for upper tier lstm. At beginning of file batch will be shape (batch_size X num_steps X num_feats).
                 At end of file during first stage of flushing batch will be shape (num_unique_users X num_steps X num_feats).
                 At end of file during second stage of flushing batch will be
                 shape (min(batch_size X num_steps, num_unique_users) X num_feats).
        """
        if self.flush:
            return self.flush_batch()
        else:
            return self.new_batch()

    def flush_batch(self):
        """
        Called when EOF is encountered. Returns either first stage flush batch or second stage flush batch.
        """
        print("flushing overflow matrices")
        self.batch_user_list = []
        batch, batch_user_set = self.get_batch_from_overflow()
        if len(batch) == 0:
            return self.collect_stragglers()
        else:
            return np.array(batch), self.get_state_triples()

    def collect_stragglers(self):
        """
        Second stage flushing
        """
        self.batch_user_list = [deq[0][3] for deq in self.user_logs if len(deq) > 0]
        if len(self.batch_user_list) == 0:
            return None, None
        else:
            straggler_mb_size = min(len(self.batch_user_list), self.mb_size * self.num_steps)
            batch = [self.user_logs[int(i)].popleft() for i in self.batch_user_list[:straggler_mb_size]]
            self.batch_user_list = self.batch_user_list[:straggler_mb_size]
            return np.array(batch), self.get_state_triples()

    def get_batch_from_overflow(self):
        """
        Called at beginning of each new batch to see if users have any premade matrix of events ready.
        """
        batch = []
        batch_user_set = set()
        idx = 0
        while len(batch) < self.mb_size and idx < len(self.user_batch_overflow):
            matrix = self.user_batch_overflow[idx]
            user = matrix[0, 3]
            if user not in batch_user_set:
                batch.append(matrix)
                batch_user_set.add(user)
                self.batch_user_list.append(user)
                self.user_batch_overflow.pop(idx)
            else:
                idx += 1
        return batch, batch_user_set

    def new_batch(self):
        """
        First checks user_batch_overflow to see if there are user batches ready for the new mini-batch.
        Iterates over the file, adding user's loglines to user_logs. When a user gets
        num_steps loglines, those num_steps loglines are added to the batch or if the user is already present
        in the batch to the user_batch_overflow. Now, when we have minibatch number of user batches, we return
        those as a batch. At most one user-batch for each user is allowed in a mini-batch
        """

        self.batch_user_list = []
        # First check overflow buffer for num_steps X sentence_length matrices for minibatch
        batch, batch_user_set = self.get_batch_from_overflow()

        # Now get more log lines from log file to make more num_steps X sentence_length matrices for minibatch
        while len(batch) < self.mb_size:
            l = self.data.readline()
            self.line_num += 1
            if l == '':
                self.flush = True
                if len(self.batch_user_list) > 0:
                    return np.array(batch), self.get_state_triples()  # batch is mb(user) X numsteps X sentence_length
                else:
                    return self.flush_batch()
            rowtext = [float(k) for k in l.strip().split(self.delimiter)]
            user = int(rowtext[3])
            self.user_logs[user].append(rowtext)
            if len(self.user_logs[user]) == self.num_steps:
                if user in batch_user_set:
                    self.user_batch_overflow.append(np.array(self.user_logs[user]))
                else:
                    batch.append(np.array(self.user_logs[user]))
                    batch_user_set.add(user)
                    self.batch_user_list.append(user)
                self.user_logs[user] = deque()

        return np.array(batch), self.get_state_triples()  # batch is mb(user) X numsteps X sentence_length


class NormalizingReplayOnlineBatcher:
    """
    For replay batching on aggregate DNN model.
    For batching data too large to fit into memory. Written for one pass on data!!!
    """

    def __init__(self, datafile, batch_size, skipheader=False,
                 delimiter=',', size_check=None, refresh_ratio=.5,
                 ratio=(1, 0), pool_size=5, alpha=0.5, datastart_index=3):
        """

        :param datafile: File to read data from
        :param batch_size: For mini-batching
        :param skipheader: Use if there is a header on the data file
        :param delimiter: Typically ' ' or ',' which delimits columns in data file
        :param size_check: Ignore this
        :param refresh_ratio: The proportion of the new mini-batch to use in refreshing the pool.
        :param ratio:  (tuple) (num_new, num_replay) The batcher will provide num_new new batches of data points
                                and then num_replay batches of old data points from the pool.
        :param pool_size: The scale of the pool. The pool will be pool_size * batchsize data points.
        :param alpha: (float)  For exponential running mean and variance.
                      Lower alpha discounts older observations faster.
                      The higher the alpha, the further you take into consideration the past.
        :param datastart_index: The csv field where real valued features to be normalized begins.
                                Assumed that all features beginnning at datastart_index till end of line
                                are real valued.
        """
        assert ratio[0] > 0 and ratio[1] > 0, "Ratio values must be greater than zero."
        assert pool_size >= batch_size, "Pool size must be larger than batch size."
        assert refresh_ratio <= 1.0 and refresh_ratio > 0.0, "Refresh ratio must be between 1 an 0. This is the percentage of the minibatch to put into the replay pool."
        self.pool_size = pool_size
        self.index = 0
        self.mod = 0
        self.period = sum(ratio)
        self.batch_function = [self.new_batch if i < ratio[0] else self.replay_batch for i in range(self.period)]
        self.batch_size = batch_size
        self.delimiter = delimiter
        self.size_check = size_check
        self.num_new = int(refresh_ratio*batch_size)
        # initialize replay pool
        self.f = open(datafile, 'r')
        if skipheader:
            self.f.readline()
        pool_list = []
        if skipheader:
                self.f.readline()
        for i in range(pool_size):
            l = self.f.readline()
            rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
            pool_list.append(rowtext)
        self.pool = np.array(pool_list)
        np.random.shuffle(self.pool)
        self.datastart_index = datastart_index
        self.mean = self.pool[:,self.datastart_index:].mean(axis=0)
        self.variance = self.pool[:, self.datastart_index:].var(axis=0)
        self.f.seek(0)
        if skipheader:
            self.header = self.f.readline()
        self.mod = 0
        self.alpha = alpha
        self.index = 0
        self.replay = False

    def next_batch(self):
        batch = self.batch_function[self.mod]()
        self.mod = (self.mod + 1) % self.period
        return batch

    def replay_batch(self):
        batch_idxs = np.random.choice(range(self.pool_size), size=self.batch_size)
        data = self.pool[batch_idxs]
        data[:, self.datastart_index:] = (data[:, self.datastart_index:] - self.mean)/(self.variance + 1e-10)
        self.replay = True
        return data

    def new_batch(self, initialize=False):
        """
        :return: until end of datafile, each time called,
                 returns mini-batch number of lines from csv file
                 as a numpy array. Returns shorter than mini-batch
                 end of contents as a smaller than batch size array.
                 Returns None when no more data is available(one pass batcher!!).
        """
        matlist = []
        l = self.f.readline()
        if l == '':
            return None
        rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
        if self.size_check is not None:
            while len(rowtext) != self.size_check:
                l = self.f.readline()
                if l == '':
                    return None
                rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
        matlist.append(rowtext)
        for i in range(self.batch_size - 1):
            l = self.f.readline()
            if l == '':
                break
            rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
            if self.size_check is not None:
                while len(rowtext) != self.size_check:
                    l = self.f.readline()
                    if l == '':
                        return None
                    rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
            matlist.append(rowtext)

        data = np.array(matlist)
        batchmean, batchvariance = (data[:, self.datastart_index:].mean(axis=0),
                                    data[:, self.datastart_index:].var(axis=0))

        self.mean = self.alpha * self.mean + (1 - self.alpha) * batchmean
        self.variance = self.alpha * self.variance + (1 - self.alpha) * batchvariance
        self.index += self.batch_size
        
        if not initialize:
            replace_idxs = np.random.choice(range(self.pool_size), size=self.num_new)
            new_recruits_idxs = np.random.choice(range(data.shape[0]), size=self.num_new)
            self.pool[replace_idxs] = data[new_recruits_idxs]
            self.index += self.batch_size
        data[:, self.datastart_index:] = (data[:, self.datastart_index:] - self.mean)/(self.variance + 1e-10)
        self.replay = False
        return data
