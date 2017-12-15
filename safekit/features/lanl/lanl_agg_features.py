"""
New method for determining general notion of common/uncommon: >< mean.
Also this script doesn't print weekend events.
"""
import os
import sys
# So we can run this code on arbitrary environment which has tensorflow but not safekit installed
cyberpath = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, cyberpath)

from safekit.features import merge_streams
import argparse
from itertools import product
from collections import Counter
from pprint import pprint
import numpy as np


def return_parser():
    """Configures and returns argparse ArgumentParser object for command line script."""
    parser = argparse.ArgumentParser("Crisp aggregate feature derivation script.")
    parser.add_argument('-datapath',
                        type=str,
                        help='Path to files to transliterate.')
    parser.add_argument('-outfile',
                        type=str,
                        help='Where to write derived features.')
    parser.add_argument('-redpath',
                        type=str,
                        help='Where the json of completely specified redteam events is.')
    return parser


def popularness(id, counter):
    if float(counter[id])/float(counter['total']) < .05:
        return 'unpop_'
    else:
        return 'common_'


def gen_popularness(id, counter):

    diff = counter['mean'] - counter[id]
    if diff <= 0:
        return 'common_'
    else:
        return 'unpop_'


class ID:

    def __init__(self):
        self.id = 0
        self.map = {}

    def __call__(self, value):
        if value not in self.map:
            eyedee = self.id
            self.map[value] = self.id
            self.id += 1
        else:
            eyedee = self.map[value]
        return eyedee


def second_to_day(seconds):
    """

    :param seconds:
    :return:
    """
    day = int(seconds)/86400
    assert day < 58, 'Too many seconds, reached day %s' % day
    return day

daymap = {0: '12am-6am',
          1: '6am-12pm',
          2: '12pm-6pm',
          3: '6pm-12am'}


def part_of_day(seconds):
    """

    :param seconds:
    :return:
    """
    time_of_day_in_seconds = int(seconds) % 86400
    daypart = time_of_day_in_seconds / 21600
    return daymap[daypart]


def hour_num(seconds):
    """

    :param seconds:
    :return:
    """
    return int(seconds)/3600


def print_events(of, day, redevents, usercounts):
    """

    :param of:
    :param day:
    :param redevents:
    :param usercounts:
    :return:
    """
    for user in usercounts.keys():
        of.write('%s,%s,%s,' % (day, user, redevents[user]) + ','.join([str(k) for k in usercounts[user].values()]) + '\n')

if __name__ == '__main__':

    weekend_days = {3, 4, 10, 11, 17, 18, 24, 25, 31, 32, 38, 39, 45, 46, 47, 52, 53}
    popularity = ['unpop_', 'common_']
    specificity = ['user_', 'all_']
    objects = ['src_pc_', 'dst_pc_', 'dst_user_', 'proc_pc_', 'proc_']
    times = ['all_day', '12am-6am', '6am-12pm', '12pm-6pm', '6pm-12am']

    # src_cmp_time_pop_user src_cmp_time_pop_all src_cmp_
    c = list(product(popularity, specificity, objects, times))
    counts = [''.join(k) for k in c]
    pprint(counts)
    print(len(counts))
    other = ['auth_h.txt',
            'proc_h.txt',
            'Start',
            'End',
            '?',
            'MICROSOFT_AUTH',
            'Kerberos',
            'TivoliAP',
            'NTLM',
            'Negotiate',
            'Wave',
            'CygwinLsa',
            'ACRONIS_RELOGON_AUTHENTICATION_PACKAGE',
            'N',
            'NETWARE_AUTHENTICATION_PACKAGE_V1_0',
            'Setuid',
            'Network',
            'Service',
            'RemoteInteractive',
            'Batch',
            'CachedInteractive',
            'NetworkCleartext',
            'Unlock',
            'Interactive',
            'NewCredentials',
            'TGS',
            'TGT',
            'ScreenUnlock',
            'ScreenLock',
            'LogOff',
            'LogOn',
            'AuthMap',
            'Success',
            'Fail']

    counts += other
    print(len(counts))

    pcid = ID()
    uid = ID()
    prid = ID()

    args = return_parser().parse_args()

    if not args.datapath.endswith('/'):
        args.datapath += '/'
    if not args.redpath.endswith('/'):
        args.redpath += '/'
    with open(args.redpath + 'redevents.txt', 'r') as red:
        redevents = set(red.readlines())

    data = merge_streams.Merge(filepath=args.datapath, file_list=['auth_h.txt', 'proc_h.txt'],
                               sort_column='time', date_format='int', delimiter=',')

    with open(args.outfile, 'w') as of:
        usercounts = {}
        dst_user_counts = Counter()
        src_pc_counts = Counter()
        dst_pc_counts = Counter()
        process_pc_counts = Counter()
        process_counts = Counter()
        user_dst_user_counts = {}
        user_pc_counts = {}
        user_src_pc_counts = {}
        user_dst_pc_counts = {}
        user_process_pc_counts = {}
        user_process_counts = {}
        point = 0
        day = 0
        hour_counts = [0]*1392
        red = Counter()
        usercounts['header'] = {k: 0 for k in counts}
        of.write('day,user,red,' + ','.join([str(k) for k in usercounts['header'].keys()]) + '\n')
        del usercounts['header']
        for event_type, event in data():

            if point % 10000 == 0:
                print(point)
            point += 1

            # Only use lines labelled with a source user that is a person.
            if event[1].startswith('U'):
                time = event[0]
                hour_counts[int(hour_num(time))] += 1
                current_day = second_to_day(time)
                if current_day > day:
                    print_events(of, day, red, usercounts)
                    day = current_day
                    usercounts = {}
                    red = Counter()
                if int(current_day) not in weekend_days:
                    timeslice = part_of_day(time)
                    user = uid(event[1].split('@')[0].replace('$', ''))
                    if ','.join(event) + '\n' in redevents:
                        redteamcount = 1
                    else:
                        redteamcount = 0
                    red[user] += redteamcount

                    if user not in usercounts:
                        usercounts[user] = {k: 0 for k in counts}
                        user_dst_user_counts[user] = Counter()
                        user_pc_counts[user] = Counter()
                        user_src_pc_counts[user] = Counter()
                        user_dst_pc_counts[user] = Counter()
                        user_process_counts[user] = Counter()
                        user_process_pc_counts[user] = Counter()
                    if event_type == 'auth_h.txt':

                        # destination user
                        dst_user = uid(event[2].split('@')[0].replace('$', ''))
                        # all
                        dst_user_counts[dst_user] += 1
                        dst_user_counts['total'] += 1
                        if len(dst_user_counts) == 2 and dst_user_counts[dst_user] == 1:
                            dst_user_counts['mean'] = 1.0
                        elif dst_user_counts[dst_user] > 1:
                            dst_user_counts['mean'] += 1.0/(len(dst_user_counts) - 2.0)
                        elif dst_user_counts[dst_user] == 1:
                            dst_user_counts['mean'] += (1 - dst_user_counts['mean'])/float(len(dst_user_counts) - 2)
                        p = gen_popularness(dst_user, dst_user_counts)
                        usercounts[user][p + 'all_dst_user_' + timeslice] += 1
                        usercounts[user][p + 'all_dst_user_' + 'all_day'] += 1
                        # user
                        user_dst_user_counts[user][dst_user] += 1
                        user_dst_user_counts[user]['total'] += 1
                        p = popularness(dst_user, user_dst_user_counts[user])
                        usercounts[user][p + 'user_dst_user_' + timeslice] += 1
                        usercounts[user][p + 'user_dst_user_' + 'all_day'] += 1

                        # source pc
                        src_pc = pcid(event[3].split('@')[0].replace('$', ''))
                        # all
                        src_pc_counts[src_pc] += 1
                        src_pc_counts['total'] += 1

                        if len(src_pc_counts) == 2 and src_pc_counts[src_pc] == 1:
                            src_pc_counts['mean'] = 1.0
                        elif src_pc_counts[src_pc] > 1:
                            src_pc_counts['mean'] += 1.0 / (len(src_pc_counts) - 2.0)
                        elif src_pc_counts[src_pc] == 1:
                            src_pc_counts['mean'] += (1 - src_pc_counts['mean']) / float(len(src_pc_counts) - 2)
                        p = gen_popularness(src_pc, src_pc_counts)
                        usercounts[user][p + 'all_src_pc_' + timeslice] += 1
                        usercounts[user][p + 'all_src_pc_' + 'all_day'] += 1
                        # user
                        user_src_pc_counts[user][src_pc] += 1
                        user_src_pc_counts[user]['total'] += 1
                        p = popularness(src_pc, user_src_pc_counts[user])
                        usercounts[user][p + 'user_src_pc_' + timeslice] += 1
                        usercounts[user][p + 'user_src_pc_' + 'all_day'] += 1

                        # dst pc
                        dst_pc = pcid(event[4].split('@')[0].replace('$', ''))
                        # all
                        dst_pc_counts[dst_pc] += 1
                        dst_pc_counts['total'] += 1
                        if len(dst_pc_counts) == 2 and dst_pc_counts[dst_pc] == 1:
                            dst_pc_counts['mean'] = 1.0
                        elif dst_pc_counts[dst_pc] > 1:
                            dst_pc_counts['mean'] += 1.0 / (len(dst_pc_counts) - 2.0)
                        elif dst_pc_counts[dst_pc] == 1:
                            dst_pc_counts['mean'] += (1 - dst_pc_counts['mean']) / float(len(dst_pc_counts) - 2)

                        p = gen_popularness(dst_pc, dst_pc_counts)
                        usercounts[user][p + 'all_dst_pc_' + timeslice] += 1
                        usercounts[user][p + 'all_dst_pc_' + 'all_day'] += 1
                        # user
                        user_dst_pc_counts[user][dst_pc] += 1
                        user_dst_pc_counts[user]['total'] += 1

                        p = popularness(dst_pc, user_dst_pc_counts[user])
                        usercounts[user][p + 'user_dst_pc_' + timeslice] += 1
                        usercounts[user][p + 'user_dst_pc_' + 'all_day'] += 1

                        # rest of auth.txt fields
                        if event[5].startswith('MICROSOFT_AUTH'):
                            usercounts[user]['MICROSOFT_AUTH'] += 1 # auth_type
                        else:
                            usercounts[user][event[5]] += 1  # auth_type
                        usercounts[user][event[6]] += 1  # logon_type
                        usercounts[user][event[7]] += 1  # auth_orient
                        usercounts[user][event[8]] += 1  # success/fail

                    elif event_type == 'proc_h.txt':

                        # proc pc
                        pc = pcid(event[2])
                        # all
                        process_pc_counts[pc] += 1
                        process_pc_counts['total'] += 1
                        if len(process_pc_counts) == 2 and process_pc_counts[pc] == 1:
                            process_pc_counts['mean'] = 1.0
                        elif process_pc_counts[pc] > 1:
                            process_pc_counts['mean'] += 1.0 / (len(process_pc_counts) - 2.0)
                        elif process_pc_counts[pc] == 1:
                            process_pc_counts['mean'] += (1 - process_pc_counts['mean']) / float(len(process_pc_counts) - 2)
                        p = gen_popularness(pc, process_pc_counts)
                        usercounts[user][p + 'all_proc_pc_' + timeslice] += 1
                        usercounts[user][p + 'all_proc_pc_' + 'all_day'] += 1
                        # user
                        user_process_pc_counts[user][pc] += 1
                        user_process_pc_counts[user]['total'] += 1

                        p = popularness(pc, user_process_pc_counts[user])
                        usercounts[user][p + 'user_proc_pc_' + timeslice] += 1
                        usercounts[user][p + 'user_proc_pc_' + 'all_day'] += 1

                        # process
                        proc = prid(event[3])
                        #all
                        process_counts[proc] += 1
                        process_counts['total'] += 1

                        p = popularness(proc, process_counts)
                        usercounts[user][p + 'all_proc_' + timeslice] += 1
                        usercounts[user][p + 'all_proc_' + 'all_day'] += 1
                        # user
                        user_process_counts[user][proc] += 1
                        user_process_counts[user]['total'] += 1

                        p = popularness(proc, user_process_counts[user])
                        usercounts[user][p + 'user_proc_' + timeslice] += 1
                        usercounts[user][p + 'user_proc_' + 'all_day'] += 1

                        # start/stop
                        usercounts[user][event[4]] += 1
        print_events(of, day, red, usercounts)
        with(open('usermap.txt', 'w')) as u:
            for k,v in uid.map.iteritems():
                u.write('%s %s\n' % (k, v))
        with(open('pcmap.txt', 'w')) as u:
            for k, v in pcid.map.iteritems():
                u.write('%s %s\n' % (k, v))
        with(open('procmap.txt', 'w')) as u:
            for k, v in prid.map.iteritems():
                u.write('%s %s\n' % (k, v))
        np.savetxt('log_line_count_by_hour.txt', np.array(hour_counts))
        np.savetxt('dst_user_counts.txt', np.array(dst_user_counts.values()))
        np.savetxt('src_pc_counts.txt', np.array(src_pc_counts.values()))
        np.savetxt('dst_pc_counts.txt', np.array(dst_pc_counts.values()))
        np.savetxt('process_pc_counts.txt', np.array(process_pc_counts.values()))
        np.savetxt('process_counts.txt', np.array(process_counts.values()))
