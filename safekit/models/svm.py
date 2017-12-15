"""One class support vector machine baseline
"""

import sys
import os
cyberpath = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, cyberpath)

import argparse
from sklearn.svm import OneClassSVM
from safekit.batch import DayBatcher
import time
from safekit.util import apr


def return_parser():
    parser = argparse.ArgumentParser("Run anomaly detection with One Class Support Vector Machine.")
    parser.add_argument('datafile', type=str,
                        help='Input data for anomaly detection.')
    parser.add_argument('result_path', type=str,
                        help='Results dir.')
    parser.add_argument('-loss_fn', type=str, default='/tmp/' + str(time.time()),
                        help='Loss file param for spearmint')
    parser.add_argument('-kern', type=str, default='sigmoid',
                        help="Specifies the kernel type to be used in the algorithm. It must be one of linear, "
                             "poly, rbf, sigmoid, or a callable. If none is given, sigmoid will be used.")
    parser.add_argument('-nu', type=float, default=0.5,
                        help="An upper bound on the fraction of training errors and a lower bound of the fraction "
                             "of support vectors. Should be in the interval (0, 1]. By default 0.5 will be taken.")
    parser.add_argument('-deg', type=int, default=3,
                        help="Degree of the polynomial kernel function (poly). "
                             "Ignored by all other kernels. Default is 3.")
    parser.add_argument('-shrink', type=str, default='False',
                        help='Whether to use the shrinking heuristic.Default is False.')

    return parser


def sample_hyps_svm(kern, nu_, deg, shrink):
    """
    :return: A OneClassSVM object with randomly sampled hyperparams, used to detect anomaly.
    """

    kernel = kern #random.choice(['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'])
    nu = nu_ #randrange_float(0.0, 0.9999, 0.05)
    degree = deg #random.randint(1,10)
    gamma = 'auto'  # default. uses 1/n_features
    coef0 = 0.0  # default. No suggested values given in documentation
    shrinking = shrink #random.choice([True, False])

    model = OneClassSVM(kernel=kernel, nu=nu,
                        degree=degree, gamma=gamma,
                        coef0=coef0, shrinking=True)

    resultsfile = open('model_OneClassSVM' +
                       '__kernel_' + str(kernel) +
                       '__nu_' + str(nu) +
                       '__degree_' + str(degree) +
                       '__gamma_' + str(gamma) +
                       '__coef0_' + str(coef0) +
                       '__shrinking_' + str(shrinking),
                       'w')

    return model


def train_model(model, batcher, res_file):
    """
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
            resultsfile.write(str(day) + ' ' + str(user) + ' ' + str(red) + ' ' + str(score[0]) + '\n')
        batch_num += 1
        print('finished batch num: ' + str(batch_num))
        mat = batcher.next_batch()


if __name__ == '__main__':

    args = return_parser().parse_args()
    day_batcher = DayBatcher(args.datafile, skiprow=1)
    if not args.result_path.endswith('/'):
        args.result_path += '/'

    resultsfile = (args.result_path + str(time.time()) + '_OneClassSVM' +
                       '__kernel_' + str(args.kern) +
                       '__nu_' + str(args.nu) +
                       '__degree_' + str(args.deg) +
                       '__coef0_' + '0.0' +
                       '__shrinking_' + str(args.shrink))

    # add args here
    model = sample_hyps_svm(args.kern, args.nu, args.deg, bool(args.shrink))
    start_time = time.time()
    train_model(model, day_batcher, resultsfile)
    
    with open(args.loss_fn, 'w') as lf: 
        lf.write(str(apr(resultsfile, [0, 12], inverse=True)))

    os.system('mv %s %s' % (resultsfile, resultsfile + '__' + str(time.time() - start_time)))
