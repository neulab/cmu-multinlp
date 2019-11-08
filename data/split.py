import argparse
import os
import shutil
from random import shuffle


if __name__ == '__main__':
    parser = argparse.ArgumentParser('split a brat dir into two dirs')
    parser.add_argument('--inp', type=str, required=True, help='input dir')
    parser.add_argument('--out', type=str, required=True, help='output dir')
    parser.add_argument('--ratio', type=float, default=None, help='ratio to split a dir')
    args = parser.parse_args()

    if args.ratio:
        for root, dirs, files in os.walk(args.inp):
            files = [file for file in files if file.endswith('.txt')]
            shuffle(files)
            ratio = int(len(files) * args.ratio)
            print('move {} files'.format(ratio * 2))
            for file in files[:ratio]:
                from_file = os.path.join(root, file)
                shutil.move(from_file, args.out)
                file = file.rsplit('.', 1)[0] + '.ann'
                from_file = os.path.join(root, file)
                shutil.move(from_file, args.out)
