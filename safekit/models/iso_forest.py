"""
Isolation Forest baseline.
"""

import sys
import os
cyberpath = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, cyberpath)
import argparse
from sklearn.ensemble import IsolationForest
from safekit.batch import DayBatcher
from safekit.util import apr
import time
import math


def sample_hyps_iso_forest(nest, contam, boot):
    """

    :param nest:
    :param contam:
    :param boot:
    :return: An IsolationForest object with specified hyperparameters, used to detect anomaly.
    """

    n_estimators = nest # random.choice(range(20, 300))  # default is 100
    max_samples = 'auto'
    contamination = contam #randrange_float(0.0, 0.5, 0.05)
    max_features = 1.0 # default is 1.0 (use all features)
    bootstrap = boot # random.choice(['True', 'False'])
    n_jobs = -1  # Uses all cores
    verbose = 0

    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                            contamination=contamination, max_features=max_features,
                            bootstrap=bootstrap, n_jobs=n_jobs, verbose=verbose)
    return model


def train_model(model, batcher, res_file):
    """
    Run model
    
    :param model: A sklearn anomaly detection model. Needs to have the decision_function() function.
    :param batcher: A Batcher object that delivers batches of training data.
    :param outfile: (file obj) Where to write results.
    """

    resultsfile = open(res_file, 'w')
    
    resultsfile.write('day user red loss\n')
    
    mat = batcher.next_batch()
    batch_num = 0
    while mat is not None:
        datadict = {'features': mat[:, 3:], 'red': mat[:, 2], 'user': mat[:, 1], 'day': mat[:, 0]}
        model.fit(datadict['features'])
        anomaly_scores = model.decision_function(datadict['features'])
        for day, user, red, score in zip(datadict['day'], datadict['user'], datadict['red'], anomaly_scores):
            if math.isnan(score) and not math.isinf(score):
                print('exiting due divergence')
                exit(1)
            else:       
                resultsfile.write(str(day) + ' ' + str(user) + ' ' + str(red) + ' ' + str(-1*score) + '\n')
        batch_num += 1
        print('finished batch num: ' + str(batch_num))
        mat = batcher.next_batch()


def return_parser():
    parser = argparse.ArgumentParser("Run anomaly detection with Isolation Forest.")
    parser.add_argument('datafile', type=str, help='Input data for anomaly detection.')
    parser.add_argument('result_path', type=str, help='Results dir.')
    parser.add_argument('-loss_fn', type=str, default='/tmp/' + str(time.time()), help='Loss file param for spearmint')
    parser.add_argument('-nest', type=int, default=100, help='Number of estimators.')
    parser.add_argument('-contam', type=float, default=0.25, help='Contamination.')
    parser.add_argument('-bootstrap', type=str, default=False, help='Bootstrap t/f.')
    return parser 


if __name__ == '__main__':

    args = return_parser().parse_args()
    day_batcher = DayBatcher(args.datafile, skiprow=1)
    
    if not args.result_path.endswith('/'):
        args.result_path += '/'

    resultsfile = (args.result_path + str(time.time()) + 'iso_forest' +
                       '__nEstimators_' + str(args.nest) +
                       '__maxSamples_' + 'auto' +
                       '__contamination_' + str(args.contam) +
                       '__max_features_' + '2' +
                       '__bootstrap_' + str(args.bootstrap) +
                       '__nJobs_' + '-1')
    
    model = sample_hyps_iso_forest(args.nest, args.contam, args.bootstrap)
    start_time = time.time()
    train_model(model, day_batcher, resultsfile)
    
    with open(args.loss_fn, 'w') as lf:
        lf.write(str(apr(resultsfile, [0, 12], inverse=True)))