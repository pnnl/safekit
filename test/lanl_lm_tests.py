"""

"""
#  TODO: Change test calls to reflect good hyper-parameters for reference
#  TODO: Add more tests for different model configurations
#  TODO: Make new test script for running all tests on full data set

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str, help="path to lanl parent directory of language model features.")
parser.add_argument('logfile', type=str, help="File to write stderr messages.")

args = parser.parse_args()

if not args.datapath.endswith('/'):
    args.datapath += '/'
modelpath = '/'.join(os.path.realpath(__file__).split('/')[:-2]) + '/safekit/models'
specpath = '/'.join(os.path.realpath(__file__).split('/')[:-2]) + '/safekit/features/specs'

num_finished = 0
all_okay = 0

# # ============================================================================
# # ================== SIMPLE LSTM =============================================
# # ============================================================================
with open(args.logfile, 'a') as log:
    log.write('simple word forward lstm\n\n\n')
print('simple word forward lstm\n\n\n')
ok = os.system('python %s/simple_lm.py ./ %s/lm/lanl_word_config.json %sword_day_split/ -encoding word -skipsos -test -delimiter , 2>> %s' % (modelpath, specpath, args.datapath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('simple word bidirectional lstm\n\n\n')
print('simple word bidirectional lstm\n\n\n')
ok = os.system('python %s/simple_lm.py ./ %s/lm/lanl_word_config.json %sword_day_split/ -encoding word -bidir -test -delimiter , 2>> %s' % (modelpath, specpath, args.datapath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('simple raw forward lstm\n\n\n')
print('simple raw forward lstm\n\n\n')
ok = os.system('python %s/simple_lm.py ./ %s/lm/lanl_char_config.json %sraw_day_split/ -encoding raw -jagged -test -delimiter , 2>> %s' % (modelpath, specpath, args.datapath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('simple raw bidirectional lstm\n\n\n')
print('simple raw bidirectional lstm\n\n\n')
ok = os.system('python %s/simple_lm.py ./ %s/lm/lanl_char_config.json %sraw_day_split/ -encoding raw -bidir -jagged -test -delimiter , 2>> %s' % (modelpath, specpath, args.datapath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

# # ============================================================================
# # ================== TIERED LSTM =============================================
# # ============================================================================
with open(args.logfile, 'a') as log:
    log.write('tiered word forward lstm\n\n\n')
print('tiered word forward lstm\n\n\n')
ok = os.system('python %s/tiered_lm.py ./ %s/lm/lanl_word_config.json %sword_day_split/ -encoding word -skipsos -test -delimiter , 2>> %s' % (modelpath, specpath, args.datapath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('tiered word bidirectional lstm\n\n\n')
print('tiered word bidirectional lstm\n\n\n')
ok = os.system('python %s/tiered_lm.py ./ %s/lm/lanl_word_config.json %sword_day_split/ -encoding word -bidir -test -delimiter , 2>> %s' % (modelpath, specpath, args.datapath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('tiered raw forward lstm\n\n\n')
print('tiered raw forward lstm\n\n\n')
ok = os.system('python %s/tiered_lm.py ./ %s/lm/lanl_char_config.json %sraw_day_split/ -encoding raw -skipsos -jagged -test -delimiter , 2>> %s' % (modelpath, specpath, args.datapath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

with open(args.logfile, 'a') as log:
    log.write('tiered raw bidirectional lstm\n\n\n')
print('tiered raw bidirectional lstm\n\n\n')
ok = os.system('python %s/tiered_lm.py ./ %s/lm/lanl_char_config.json %sraw_day_split/ -encoding raw -bidir -jagged -test -delimiter , 2>> %s' % (modelpath, specpath, args.datapath, args.logfile))
num_finished += ok == 0
all_okay += ok != 0

print('Number finished: %s\nNumber failed: %s' % (num_finished, all_okay))

