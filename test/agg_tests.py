"""

"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lanl_datapath', type=str, help="path to lanl parent directory of aggregate model features.")
parser.add_argument('cert_datapath', type=str, help="path to cert parent directory of aggregate model features.")
parser.add_argument('logfile', type=str, help="File to write stderr messages.")
args = parser.parse_args()

if not args.lanl_datapath.endswith('/'):
    args.lanl_datapath += '/'
if not args.cert_datapath.endswith('/'):
    args.cert_datapath += '/'
modelpath = '/'.join(os.path.realpath(__file__).split('/')[:-2]) + '/safekit/models'
specpath = '/'.join(os.path.realpath(__file__).split('/')[:-2]) + '/safekit/features/specs/agg'

num_finished = 0
all_okay = 0


# =================================cert agg dnn stuff==================================================================


with open(args.logfile, 'a') as log:
    log.write('cert dnn autoencoder w/diag covariance, batch norm, input norm, replay\n\n\n')
print('\n\ncert dnn autoencoder w/diag covariance, batch norm, input norm, replay\n\n\n')
ok = os.system('python %s/dnn_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist diag -norm batch -replay 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert dnn autoencoder w/diag covariance, batch norm, input norm\n\n\n')
print('\n\ncert dnn autoencoder w/diag covariance, batch norm, input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist diag -input_norm -alpha 0.5 -norm batch 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert dnn autoencoder w/diag covariance, layer norm, input norm\n\n\n')
print('\n\ncert dnn autoencoder w/diag covariance, layer norm, input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist diag -input_norm -alpha 0.5 -norm layer 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('\n\ncert dnn autoencoder w/ identity covariance, no model norm, input norm\n\n\n')
print('\n\ncert dnn autoencoder w/ identity covariance, no model norm, input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist ident -input_norm -alpha 0.5 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert dnn autoencoder w/ full covariance, no model norm, no input norm\n\n\n')
print('\n\ncert dnn autoencoder w/ full covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist full -input_norm -alpha 0.5 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert dnn autoencoder w/ diag covariance, no model norm, no input norm\n\n\n')
print('\n\ncert dnn autoencoder w/ diag covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist diag 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert dnn autoencoder w/ identity covariance, no model norm, no input norm\n\n\n')
print('\n\ncert dnn autoencoder w/ identity covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist ident 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert dnn autoencoder w/ full covariance, no model norm, no input norm\n\n\n')
print('\n\ncert dnn autoencoder w/ full covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist full 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0
# lanl agg dnn stuff

with open(args.logfile, 'a') as log:
    log.write('dnn autoencoder w/diag covariance, batch norm, input norm, replay\n\n\n')
print('\n\ndnn autoencoder w/diag covariance, batch norm, input norm, replay\n\n\n')
ok = os.system('python %s/dnn_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist diag -norm batch -replay -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('dnn autoencoder w/diag covariance, batch norm, input norm\n\n\n')
print('\n\ndnn autoencoder w/diag covariance, batch norm, input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist diag -input_norm -alpha 0.5 -norm batch -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('dnn autoencoder w/diag covariance, layer norm, input norm\n\n\n')
print('\n\ndnn autoencoder w/diag covariance, layer norm, input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist diag -input_norm -alpha 0.5 -norm layer -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('dnn autoencoder w/ identity covariance, no model norm, input norm\n\n\n')
print('\n\ndnn autoencoder w/ identity covariance, no model norm, input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist ident -input_norm -alpha 0.5 -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('dnn autoencoder w/ full covariance, no model norm, input norm\n\n\n')
print('\n\ndnn autoencoder w/ full covariance, no model norm, input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist full -input_norm -alpha 0.5 -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('dnn autoencoder w/ diag covariance, no model norm, no input norm\n\n\n')
print('\n\ndnn autoencoder w/ diag covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist diag -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('dnn autoencoder w/ identity covariance, no model norm, no input norm\n\n\n')
print('\n\ndnn autoencoder w/ identity covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist ident -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('dnn autoencoder w/ full covariance, no model norm, no input norm\n\n\n')
print('\n\ndnn autoencoder w/ full covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/dnn_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist full -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0


# ==========================================cert agg lstm stuff=====================================================

with open(args.logfile, 'a') as log:
    log.write('cert rnn autoencoder w/diag covariance, batch norm, input norm, replay\n\n\n')
print('\n\ncert rnn autoencoder w/diag covariance, batch norm, input norm, replay\n\n\n')
ok = os.system('python %s/lstm_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist diag -norm batch -replay_ratio 2 2 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert rnn autoencoder w/diag covariance, batch norm, input norm\n\n\n')
print('\n\ncert rnn autoencoder w/diag covariance, batch norm, input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist diag  -input_norm -alpha 0.5 -norm batch 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert rnn autoencoder w/diag covariance, layer norm, input norm\n\n\n')
print('\n\ncert rnn autoencoder w/diag covariance, layer norm, input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist diag -input_norm -alpha 0.5 -norm layer 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert rnn autoencoder w/ identity covariance, no model norm, input norm\n\n\n')
print('\n\ncert rnn autoencoder w/ identity covariance, no model norm, input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist ident -input_norm -alpha 0.5 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert rnn autoencoder w/ full covariance, no model norm, input norm\n\n\n')
print('\n\ncert rnn autoencoder w/ full covariance, no model norm, input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist full -input_norm -alpha 0.5 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert rnn autoencoder w/ diag covariance, no model norm, no input norm\n\n\n')
print('\n\ncert rnn autoencoder w/ diag covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist diag 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert rnn dnn autoencoder w/ identity covariance, no model norm, no input norm\n\n\n')
print('\n\ncert rnn dnn autoencoder w/ identity covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist ident 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('cert rnn dnn autoencoder w/ full covariance, no model norm, no input norm\n\n\n')
print('\n\ncert rnn dnn autoencoder w/ full covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %scert_head.csv /tmp %s/cert_all_in_all_out_agg.json -dist full 2>> %s' % (modelpath, args.cert_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

# ======================================lanl agg rnn stuff=======================================================

with open(args.logfile, 'a') as log:
    log.write('rnn autoencoder w/diag covariance, batch norm, input norm, replay\n\n\n')
print('\n\nrnn autoencoder w/diag covariance, batch norm, input norm, replay\n\n\n')
ok = os.system('python %s/lstm_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist diag -norm batch -replay_ratio 2 2 -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('rnn autoencoder w/diag covariance, batch norm, input norm\n\n\n')
print('\n\nrnn autoencoder w/diag covariance, batch norm, input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist diag -alpha 0.5 -norm batch -input_norm -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('rnn autoencoder w/diag covariance, layer norm, input norm\n\n\n')
print('\n\nrnn autoencoder w/diag covariance, layer norm, input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist diag -input_norm -alpha 0.5 -norm layer -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('rnn autoencoder w/ identity covariance, no model norm, input norm\n\n\n')
print('\n\nrnn autoencoder w/ identity covariance, no model norm, input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist ident -input_norm -alpha 0.5 -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('rnn autoencoder w/ full covariance, no model norm, input norm\n\n\n')
print('\n\nrnn autoencoder w/ full covariance, no model norm, input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist full -input_norm -alpha 0.5 -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('rnn autoencoder w/ diag covariance, no model norm, no input norm\n\n\n')
print('\n\nrnn autoencoder w/ diag covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist diag -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('rnn autoencoder w/ identity covariance, no model norm, no input norm\n\n\n')
print('\n\nrnn autoencoder w/ identity covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist ident -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('rnn autoencoder w/ full covariance, no model norm, no input norm\n\n\n')
print('\n\nrnn autoencoder w/ full covariance, no model norm, no input norm\n\n\n')
ok = os.system('python %s/lstm_agg.py %slanl_agg_head.txt /tmp %s/lanl_count_in_count_out_agg.json -dist full -delimiter , -skipheader 2>> %s' % (modelpath, args.lanl_datapath, specpath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

print('Number finished: %s\nNumber failed: %s' % (num_finished, all_okay))


