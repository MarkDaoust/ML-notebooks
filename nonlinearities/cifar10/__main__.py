from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import subprocess

from . import nonlinearities as nls

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--clear',
    help='clear training directories instead of continuing',
    dest='clear',
    action='store_true',
    default=False)

parser.add_argument('--max_train_hours',
    help='number of hours to run each nonlinearity',
    default=2,
    type=float)

parser.add_argument('--n_runs',
    help='number of times to try each nonlinearity',
    default=1,
    type=int)

parser.add_argument('--train_dir',
    help='dir to store training progress',
    default='logs',
    type=str)

parser.add_argument('nonlinearities',
                    help='list of nonlinearities to run',
                    nargs=argparse.REMAINDER,
                    type=str,
                    choices=nls.NAMES)

def main(args):
    args = parser.parse_args(args)

    try:
        os.mkdir(args.train_dir)
    except OSError as E:
        if E.strerror != 'File exists':
            raise

    if not args.nonlinearities:
        args.nonlinearities=nls.NAMES

    for nl in args.nonlinearities:


        for n in range(args.n_runs):
            if args.n_runs != 1:
                id = nl+'_#'+str(n)
            else:
                id = nl

            print(80*'*')
            print(id)
            print(80*'*')

            my_dir = os.path.join(args.train_dir,id)

            try:
                os.mkdir(my_dir)
            except OSError as E:
                if E.strerror != 'File exists':
                    raise


            cmd = [
                'python','-m','cifar10.train',
                    '--data_dir','cifar10/data',
                    '--max_train_hours',str(args.max_train_hours),
                    '--train_dir',my_dir,'--nonlinearity',nl]

            if args.clear:
                cmd.append('--clear')

            subprocess.call(cmd)


if __name__ == "__main__":
    main(sys.argv[1:])
