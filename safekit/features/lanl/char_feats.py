"""
SOS = 0, EOS = 1, all other chars are ASCII values - 30
,, Note:: Line numbers in raw_char are off by one from original raw data in auth_h.txt. However, no data is changed.

.. Note:: The first field time stamp is not transliterated here, just used for meta data
"""

import argparse
from word_feats import second_to_day

# TODO: CHECK THAT PADDING WORKS, LINE LENGTHS ARE CORRECT AND WRITING PROPERLY


def return_parser():
    """Configures and returns argparse ArgumentParser object for command line script."""
    parser = argparse.ArgumentParser("Convert raw loglines to ASCII code (minus 30) integer representation.")
    parser.add_argument('-datapath',
                        type=str,
                        help='Path to files to transliterate.')
    parser.add_argument('-outfile',
                        type=str,
                        help='Where to write derived features.')
    return parser


def translate_line(string, pad_len):
    """

    :param string:
    :param pad_len:
    :return:
    """
    return "0 " + " ".join([str(ord(c) - 30) for c in string]) + " 1 " + " ".join(["0"] * pad_len) + "\n"

if __name__ == '__main__':

    LONGEST_LEN = 120  # Length of the longest line in auth_h.txt, used for padding
    weekend_days = [3, 4, 10, 11, 17, 18, 24, 25, 31, 32, 38, 39, 45, 46, 47, 52, 53]
    args = return_parser().parse_args()

    if not args.datapath.endswith('/'):
        args.datapath += '/'

    with open(args.datapath + 'redevents.txt', 'r') as red:
        redevents = set(red.readlines())

    with open(args.datapath + "auth_h.txt", 'r') as infile, open(args.outfile, 'w') as outfile:
        outfile.write('line_number second day user red seq_len start_sentence\n')
        infile.readline()
        for line_num, line in enumerate(infile):
            if line_num % 10000 == 0:
                print line_num
            line_minus_time = ','.join(line.strip().split(',')[1:])
            diff = LONGEST_LEN - len(line_minus_time)
            raw_line = line.split(",")
            if len(raw_line) != 9:
                print('bad length')
                continue
            sec = raw_line[0]
            user = raw_line[1].strip().split('@')[0]
            day = second_to_day(int(sec))
            red = 0
            red += int(line in redevents)
            if user.startswith('U') and day not in weekend_days:
                index_rep = translate_line(line_minus_time, diff)
                outfile.write("%s %s %s %s %s %s %s" % (line_num, 
                                                        sec, 
                                                        day, 
                                                        user.replace("U", ""), 
                                                        red, 
                                                        len(line_minus_time) + 1,
                                                        index_rep))
