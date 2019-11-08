#!/usr/bin/env python

import argparse
import os
import json
import xml.etree.ElementTree as ET


if __name__ == '__main__':
    parser = argparse.ArgumentParser('convert xml format into brat format')
    parser.add_argument('--inp', type=str, required=True, help='input file')
    parser.add_argument('--out', type=str, required=True, help='output dir')
    args = parser.parse_args()

    num_sent_per_doc = 10

    used_num_sent = 0
    used_num_span = 0
    nolabel_num_span = 0
    doc_id = 1
    doc_out, ann_out = None, None

    for files in args.inp.split('::'):
        xml_file = files.split(':')
        if len(xml_file) == 2:
            xml_file, json_data = xml_file
            with open(json_data, 'r') as fin:
                json_data = json.load(fin)
        elif len(xml_file) == 1:
            xml_file, json_data = xml_file[0], None
        else:
            raise Exception

        sents = ET.parse(xml_file).getroot()
        print('totally {} sentences'.format(len(sents)))

        for sent_ind, sent in enumerate(sents):
            sid = sent.attrib['id']
            sent_str = sent.find('text').text
            ats = sent.find('aspectTerms')
            if ats is None:  # skip sentences without aspect terms
                continue

            if doc_out is None:
                doc_out = open(os.path.join(args.out, '{}.txt'.format(doc_id)), 'w')
                ann_out = open(os.path.join(args.out, '{}.ann'.format(doc_id)), 'w')
                num_sent = 0
                sent_off = 0
                span_id = 1

            has_at = False
            for at_ind, at in enumerate(ats):
                term = at.attrib['term']
                from_ind = int(at.attrib['from'])
                to_ind = int(at.attrib['to'])
                if sent_str[from_ind:to_ind] != term:
                    raise Exception('term not correct')
                if 'polarity' in at.attrib:
                    pol = at.attrib['polarity']
                elif json_data:
                    find = False
                    for k, v in json_data.items():
                        if k.startswith('{}'.format(sid)) and v['term'] == term:
                            if find:
                                print(sid)
                                raise Exception('ambiguous')
                            find = True
                            pol = v['polarity']
                    if not find:
                        #raise Exception('not found annotation in json data')
                        nolabel_num_span += 1
                        continue
                else:
                    raise Exception('not ground truth')

                if pol == 'conflict':
                    continue  # drop this sentiment

                # write aspect
                ann_out.write('{}\t{} {} {}\t{}\n'.format(
                    'T{}'.format(span_id ), pol, from_ind + sent_off, to_ind + sent_off, term))
                span_id += 1
                used_num_span += 1
                has_at = True

            if not has_at:
                continue

            # write sentence
            doc_out.write('{}\n'.format(sent_str))
            num_sent += 1
            sent_off += len(sent_str) + 1
            used_num_sent += 1

            if num_sent % num_sent_per_doc == 0:
                doc_out = ann_out = None
                doc_id += 1

    print('#sent {} used, #span used {}, #aspects without label {}'.format(used_num_sent, used_num_span, nolabel_num_span))
