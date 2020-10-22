import argparse
import json
import collections


def update(orig_dict, new_dict, should_num_epochs=None, this_num_epochs=None):
    for key, val in new_dict.items():
        # special cases
        if key in {'train_data_path', 'validation_data_path', 'test_data_path'}:
            orig_dict[key] += ':' + val
            continue
        if key in {'validation_metric', 'sorting_keys', 'regularizer', 'special_metric',
                   'text_field_embedder', 'num_epochs', 'patience'}:
            continue
        if key == 'eval_span_pair_skip':
            if len(orig_dict.get(key, {})) < len(val):
                orig_dict[key] = val
            continue
        if key == 't_total':  # add steps for bert
            if should_num_epochs and this_num_epochs:
                orig_dict[key] += int((val / this_num_epochs) * should_num_epochs)
            else:
                orig_dict[key] += val
            continue

        # general rules
        if isinstance(val, collections.Mapping):
            tmp = update(orig_dict.get(key, {}), val,
                         should_num_epochs=should_num_epochs, this_num_epochs=this_num_epochs)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='combine single-task config for multitask learning')
    parser.add_argument('--inp', type=str, help='config files separated by :', required=True)
    parser.add_argument('--num_samples', type=str, help='number of samples for each task', default=None)
    parser.add_argument('--remove_metric', type=str, help='remove metrics, e.g., consti', default=None)
    parser.add_argument('--extend_step', action='store_true', help='extend step for bert')
    parser.add_argument('--out', type=str, help='output file', required=True)
    parser.add_argument('--specify_task_layer', action='store_true',
                        help='whether to specify the layers in bert used for different tasks')
    args = parser.parse_args()

    conf_files = args.inp.split(':')
    assert len(conf_files) >= 1, 'no config exist'

    num_task = len(conf_files)
    print('#conf {}'.format(num_task))
    if not args.num_samples:
        num_samples = None
    else:
        num_samples = list(map(lambda x: None if x == 'null' else int(x), args.num_samples.split(':')))
        assert len(num_samples) == num_task

    confs = []
    for conf_file in conf_files:
        with open(conf_file, 'r') as fin:
            conf = json.load(fin)
            if args.remove_metric:
                special_metric = {}
                for k, v in conf['model']['special_metric'].items():
                    special_metric[k] = [m for m in v if m != args.remove_metric]
                conf['model']['special_metric'] = special_metric
            confs.append(conf)

    # update number of samples
    if num_samples is not None:
        for conf, ns in zip(confs, num_samples):
            ns_conf = conf['dataset_reader']['max_num_sample']
            assert len(ns_conf) == 1, 'multiple task in one config'
            for k in ns_conf:
                ns_conf[k] = ns

    new_conf = {}
    # init new config using the first task (this first task in major task)
    new_conf.update(confs[0])

    # num epochs
    num_epochs = new_conf['trainer']['num_epochs']

    if num_task > 1:
        # add sampler
        new_conf['dataset_reader']['task_sampler'] = 'proportional'

    # update new config using the other tasks
    for conf in confs[1:]:
        this_ne = conf['trainer']['num_epochs']
        update(new_conf, conf, should_num_epochs=num_epochs, this_num_epochs=this_ne)

    # extend steps for bert
    if args.extend_step and new_conf['trainer']['optimizer']['type'] == 'bert_adam':
        ostep = new_conf['trainer']['optimizer']['t_total']
        nstep = int(ostep * 1.1)
        new_conf['trainer']['optimizer']['t_total'] = nstep
        print('extend step {} -> {}'.format(ostep, nstep))

    if args.specify_task_layer:
        tasks = {'oie': -1, 'ner': -3}
        new_conf['model']['text_field_embedder']['token_embedders']['bert']['top_layer_only'] = False
        new_conf['model']['text_field_embedder']['token_embedders']['bert']['method'] = 'hier'
        new_conf['model']['text_field_embedder']['token_embedders']['bert']['tasks'] = tasks

    with open(args.out, 'w') as fout:
        json.dump(new_conf, fout, indent=True)
