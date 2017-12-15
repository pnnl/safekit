"""Principal Components Analysis autoencoder baseline"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import argparse


class Batcher:
    """
    For batching data too large to fit into memory. Written for one pass on data!!!
    """

    def __init__(self, datafile):
        """
        :param datafile: File to read lines from.
        :param batch_size: Mini-batch size.
        """
        self.f = open(datafile, 'r')
        self.f.readline()   # added for header
        self.current_line = self.f.readline()
        self.current_day = -1

    def next_batch(self):
        """
        :return: until end of datafile, each time called,
                 returns mini-batch number of lines from csv file
                 as a numpy array. Returns shorter than mini-batch
                 end of contents as a smaller than batch size array.
                 Returns None when no more data is available(one pass batcher!!).
        """
        matlist = []
        if self.current_line == '':
            return None
        rowtext = np.array([float(k) for k in self.current_line.strip().split(',')])
        self.current_day = rowtext[0]
        while rowtext[0] == self.current_day:
            self.current_day = rowtext[0]
            matlist.append(rowtext)
            self.current_line = self.f.readline()
            if self.current_line == '':
                break
            rowtext = np.array([float(k) for k in self.current_line.strip().split(',')])

        return np.array(matlist)


def train(train_data, outfile):
        """
        :param train_data: A Batcher object that delivers batches of train data.
        :param outfile: (str) Where to print results.
        """
        outfile.write('day user red loss\n')
        mat = train_data.next_batch()
        while mat is not None:
            datadict = {'features': mat[:, 3:], 'red': mat[:,2], 'user': mat[:,1], 'day': mat[:,0]}
            batch = scale(datadict['features'])
            pca = PCA(n_components=1)
            pca.fit(batch)
            data_reduced = np.dot(batch, pca.components_.T) # pca transform
            data_original = np.dot(data_reduced, pca.components_) # inverse_transform
            pointloss = np.mean(np.square(batch - data_original), axis=1)
            loss = np.mean(pointloss)
            for d, u, t, l, in zip(datadict['day'].tolist(), datadict['user'].tolist(),
                                   datadict['red'].tolist(), pointloss.flatten().tolist()):
                outfile.write('%s %s %s %s\n' % (d, u, t, l))
            print('loss: %.4f' % loss)
            mat = train_data.next_batch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PCA autoencoder")
    parser.add_argument('datafile', type=str, help='Input data for anomaly detection')
    parser.add_argument('results', type=str, help='Where to print results.')
    parser.add_argument('-components', type=int, help='Number of principal components to use in reconstruction.')
    args = parser.parse_args()
    with open(args.results, 'w') as outfile:
        data = Batcher(args.datafile)
        train(data, outfile)
