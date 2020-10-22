from typing import List, Tuple
import argparse
import os
import torch
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict


random.seed(0)
np.random.seed(0)


def load_emb_sim(filename: str):
    sim_li = []
    count = 0
    with open(filename, 'r') as fin:
        for l in fin:
            sent = list(map(float, l.split(' ')))
            if len(sent) <= 1:
                count += 1
                continue
            sim_li.append(sent)
    print('#one-word sent: {}'.format(count))
    return sim_li


def coef(sim_li1, sim_li2):
    cs = []
    for s1, s2 in zip(sim_li1, sim_li2):
        assert len(s1) == len(s2)
        cs.append(pearsonr(s1, s2))
    return np.mean(cs)


def emb_ana(root: str, filenames: List[str]):
    sim_li_li = []
    for fn in filenames:
        sim_li = load_emb_sim(os.path.join(root, fn))
        sim_li_li.append(sim_li)
        print([len(sl) for sl in sim_li[:10]])

    for i, sl in enumerate(sim_li_li[1:]):
        c = coef(sim_li_li[0], sl)
        print(filenames[i+1], c)


def model_parameter_copy(from_path, to_path, out_path):
    from_model = torch.load(from_path, map_location='cpu')
    to_model = torch.load(to_path, map_location='cpu')

    for k, v in from_model.items():
        if '_span_label_proj' in k:
            continue
        if '_span_pair_label_proj' in k:
            continue
        if k in to_model:
            to_model[k] = v

    torch.save(to_model, out_path)


class attn_iter():
    NUM_HEAD = 12
    NUM_LAYER = 12

    def __init__(self, file):
        self.file = file
        self.ind = -1
        self.batch = -1

    def __del__(self):
        self.file.close()

    @property
    def attn_head(self):
        return self.ind % self.NUM_HEAD

    @property
    def attn_layer(self):
        return self.batch % self.NUM_LAYER

    def next(self):
        for l in self.file:
            if l.startswith('* '):
                self.ind += 1
                bid, l = l[1:].strip().split('\t', 1)
                bid = int(bid)
                if bid == 0 and self.ind % self.NUM_HEAD == 0:
                    self.batch += 1
                attn = np.array(list(map(float, l.split(' '))))
                attn_len = int(np.sqrt(attn.shape[0]))
                return attn.reshape(attn_len, attn_len), self.attn_head, self.attn_layer
        raise StopIteration()

    def skip(self):
        for l in self.file:
            if l.startswith('* '):
                self.ind += 1
                bid, l = l[1:].strip().split('\t', 1)
                bid = int(bid)
                if bid == 0 and self.ind % self.NUM_HEAD == 0:
                    self.batch += 1
                return
        raise StopIteration()


def attn_ana(root: str, filenames: List[str], sample: float=1.0, detect: bool=False):
    files = []
    for fn in filenames:
        files.append(attn_iter(open(os.path.join(root, fn), 'r')))

    pearsonr_li = [[] for _ in range(len(filenames) - 1)]
    pbar = tqdm()
    layerhead2sim = defaultdict(list)
    while True:
        try:
            pbar.update(1)
            if random.random() > sample:
                files[0].skip()
                [file.skip() for file in files[1:]]
                continue
            attn, head, layer = files[0].next()
            other_attns = [file.next()[0] for file in files[1:]]
            for i, oa in enumerate(other_attns):
                if len(attn) != len(oa):
                    raise Exception('not the same example')
                #sim = pearsonr(attn.reshape(-1), oa.reshape(-1))
                sim = -np.linalg.norm(attn - oa, ord='fro')
                pearsonr_li[i].append(sim)
                layerhead2sim['{}-{}'.format(layer, head)].append(sim)
        except StopIteration:
            break
    pbar.close()
    pearsonr_li = [np.mean(p) for p in pearsonr_li]
    print(list(zip(filenames[1:], pearsonr_li)))

    if detect:
        layerhead2sim = dict((k, np.mean(v)) for k, v in layerhead2sim.items())
        #layerhead2sim = dict((k, len(v)) for k, v in layerhead2sim.items())
        layerhead2sim_sorted = sorted(layerhead2sim.items(), key=lambda x: x[1])
        for k, v in layerhead2sim_sorted:
            print('{}\t{:.3f}'.format(k, v))
        for l in range(12):
            for h in range(12):
                print('{:.3f}'.format(layerhead2sim['{}-{}'.format(11 - l, h)]), end='\t')
            print('')


def get_sig_test_data(filename: str) -> Tuple[List[int], List[int], List[int]]:
    count, precision, recall = [], [], []
    with open(filename, 'r') as fin:
        for l in fin:
            if not l.startswith('ST'):
                continue
            _, c, p, r  = l.strip().split('\t')
            count.append(int(c))
            precision.append(int(p))
            recall.append(int(r))
    return np.array(count), np.array(precision), np.array(recall)


def get_f1(count: List[int], precision: List[int], recall: List[int]) -> float:
    c = np.sum(count)
    p = np.sum(precision)
    r = np.sum(recall)
    p = c / (p + 1e-10)
    r = c / (r + 1e-10)
    f = 2 * p * r / (p + r + 1e-10)
    return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='copy model parameters')
    parser.add_argument('--task', type=str, help='task', required=True,
                        choices=['combine_weight', 'ana', 'attn_ana', 'sig_test'])
    parser.add_argument('--from_path', type=str, help='from model file', required=True)
    parser.add_argument('--to_path', type=str, help='to model file', required=False)
    parser.add_argument('--out', type=str, help='output file', required=False)
    parser.add_argument('--sample', type=float, help='the sample probability of attn_ana', default=1.0)
    parser.add_argument('--st_ratio', type=float, help='the ratio of samples to take every time', default=0.5)
    parser.add_argument('--st_num_samples', type=int, help='the number of bootstrap samples to tak', default=1000)
    args = parser.parse_args()

    if args.task == 'combine_weight':
        model_parameter_copy(args.from_path, args.to_path, args.out)
    elif args.task == 'ana':
        '''
        sim_files = [args.from_path]  # the main file
        for root, dirs, files in os.walk(os.path.dirname(args.from_path)):
            for file in files:
                file = os.path.join(root, file)
                if file != args.from_path:
                    sim_files.append(file)
        '''
        root, files = args.from_path.split(':')
        files = files.split(',')
        emb_ana(root, files)
    elif args.task == 'attn_ana':
        root, files = args.from_path.split(':')
        files = files.split(',')
        attn_ana(root, files, args.sample, detect=True)
    elif args.task == 'sig_test':
        stl, mtl = args.from_path.split(':')
        stl_c, stl_p, stl_r = get_sig_test_data(stl)
        mtl_c, mtl_p, mtl_r = get_sig_test_data(mtl)
        assert np.all(stl_r == mtl_r), 'recall should be the same, otherwise the output lost order'
        n = len(stl_c)
        ids = list(range(n))
        stl_scores: List[float] = []
        mtl_scores: List[float] = []
        for i in range(args.st_num_samples):
            if args.st_ratio % 1 != 0:  # float
                #np.random.shuffle(ids)
                #reduced_ids = ids[:int(len(ids) * args.st_ratio)]
                reduced_ids = np.random.choice(len(ids), int(len(ids) * args.st_ratio), replace=True)
            else:
                reduced_ids = np.random.choice(len(ids), int(args.st_ratio), replace=True)
            sc, sp, sr = stl_c[reduced_ids], stl_p[reduced_ids], stl_r[reduced_ids]
            mc, mp, mr = mtl_c[reduced_ids], mtl_p[reduced_ids], mtl_r[reduced_ids]
            sf1 = get_f1(sc, sp, sr)
            mf1 = get_f1(mc, mp, mr)
            stl_scores.append(sf1)
            mtl_scores.append(mf1)
        stl_socres = np.array(stl_scores)
        mtl_scores = np.array(mtl_scores)
        score = (mtl_scores > stl_socres).astype(float).mean()
        p_value = 1 - score
        print('mtl is better than stl with p-value {}'.format(p_value))
