from logging_utils import initialize_logger
import argparse
from train import train
from data_utils import *
import logging
import json
import random
import os
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTL for SRL and ORL',
                                     add_help=False, conflict_handler='resolve')

    ''' input '''
    parser.add_argument('--emb_size', type=int, default=100, help='dimension of embeddings')
    parser.add_argument('--emb_type', default='glove', help='type of embeddings')
    parser.add_argument('--window_size', type=int, default=2, help='window_size')
    parser.add_argument('--word_freq',  type=int, default=1, help='word frequency')
    parser.add_argument('--embeddings_trainable', type=str, default='False', help='train the embeddings or not')
    parser.add_argument('--srl_labels_inventory', type=str, default='flexi', help='all or restricted label inventory')

    ''' architecture '''
    parser.add_argument('--hidden_size', type=int, default=100, help='dimension of LSTM hidden layer')
    parser.add_argument('--n_layers_shared',  type=int, default=3, help='number of shared layers')
    parser.add_argument('--n_layers_orl',  type=int, default=0, help='number of ORL layers')
    parser.add_argument('--keep_rate_input', type=float, default=0.7, help='keep input rate')
    parser.add_argument('--keep_rate_output', type=float, default=0.85, help='keep output rate')
    parser.add_argument('--keep_state_rate', type=float, default=0.85, help='keep state rate')
    parser.add_argument('--cell', type=str, default='lstm', help='gru/lstm')

    ''' training options '''
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--adv_coef', type=float, default=0.0, help='the coefficient of the adversarial loss')
    parser.add_argument('--reg_coef', type=float, default=0.0, help='the L2 regularization coefficient')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=10000, help='number of epochs to train')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients to this value')

    ''' misc '''
    parser.add_argument('--seed', type=int, default=24, help='random seed')
    parser.add_argument('--model', type=str, default=None, help='fs, sp, asp, wsp, hmtl')
    parser.add_argument('--begin_fold', type=int, default=0, help='start fold')
    parser.add_argument('--end_fold', type=int, default=None, help='end fold')
    parser.add_argument('--out_dir', type=str, default='outputs', help='out dir prefix')
    parser.add_argument('--log_dir', type=str, default='logs', help='log dir prefix')
    parser.add_argument('--gpu_fraction', type=float, default=0.3, help='gpu fraction')
    parser.add_argument('--exp_setup_id', type=str, default=None, help='prior or new')
    parser.add_argument('--att_link_obligatory', type=str, default='false', help='alternative is true')
    argv = parser.parse_args()

    # important for reproducibility and comparision
    np.random.seed(argv.seed)
    random.seed(argv.seed)

    # imporant arguments
    assert argv.model

    if argv.model in ['sp', 'fs', 'hmtl', 'wsp']:
        assert argv.adv_coef == 0.0

    if argv.model in ['asp', 'wasp']:
        assert argv.adv_coef > 0.0

    if argv.model == 'hmtl':
        assert argv.n_layers_shared == 2
        assert argv.n_layers_orl == 1

    if argv.model == 'fs':
        assert argv.n_layers_orl == 0

    out_dir_full = argv.out_dir + '/' + argv.exp_setup_id + '/' + str(argv.seed) + '/' + argv.model + '/'
    if not os.path.exists(out_dir_full):
        os.makedirs(out_dir_full)
    parser.add_argument('--out_dir', type=str, default=out_dir_full, help='outputs/fs,sp,asp/')

    argv = parser.parse_args()

    # Logging
    # ==================================================
    log_dir = argv.log_dir + '/' + str(argv.seed) + '/' + argv.model + '/' + str(argv.begin_fold + 1) + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    initialize_logger(log_dir)

    # load CoNLL'05 corpus
    #logging.info('Loading CoNLL05 corpus...')
    #srl_train_corpus = load_conll('../corpora/conll_srl-2005/train-set.txt', 5)
    #srl_dev_corpus = load_conll('../corpora/conll_srl-2005/dev-set.txt', 5)
    #test_brown = load_conll('../corpora/conll_srl-2005/test-set-brown.txt', 4)
    #test_wsj = load_conll('../corpora/conll_srl-2005/test-set-wsj.txt', 4)
    #srl_test_corpus = test_brown + test_wsj

    # retrieve sentences from SRL train data for constructing the vocabulary
    #train_sentences_srl = [[w[0].lower() for w in sent] for sent in srl_train_corpus]

    average_fscore_binary = [0.0]*3
    average_fscore_proportional = [0.0]*3

    num_folds = argv.end_fold - argv.begin_fold

    # dev set is the same for all folds
    json_name = 'jsons/' + str(argv.exp_setup_id) + '/dev.json'
    with open(json_name) as data_file:
        orl_dev_corpus = json.load(data_file)

    for fold in range(argv.begin_fold, argv.end_fold):
        # Data Preparation
        # ==================================================

        # retrieve sentences from ORL train data of the first fold for constructing the vocabulary
        json_name = 'jsons/' + str(argv.exp_setup_id) + '/train_fold_' + str(fold) + '.json'
        with open(json_name) as data_file:
            orl_train_corpus = json.load(data_file)

        train_sentences_orl = []
        for doc_num in range(orl_train_corpus['documents_num']):
            document_name = 'document' + str(doc_num)
            doc = orl_train_corpus[document_name]

            for sent_num in range(doc['sentences_num']):
                sentence_name = 'sentence' + str(sent_num)
                sentence_lower = map(lambda x: x.lower(), doc[sentence_name]['sentence_tokenized'])
                train_sentences_orl.append(sentence_lower)

        # construct a joint SRL, ORL vocabulary
        eval_orl_sent = eval_orl_sentences(argv.exp_setup_id, fold)
        #train_sentences_srl_filter = [s for s in train_sentences_srl if s not in eval_orl_sent]
        #train_sentences = train_sentences_srl_filter + train_sentences_orl
        train_sentences = train_sentences_orl

        embeddings, vocabulary, _ = get_emb_vocab(train_sentences, argv.emb_type, argv.emb_size, argv.word_freq)

        # load test set
        json_name = 'jsons/' + str(argv.exp_setup_id) + '/test_fold_' + str(fold) + '.json'
        with open(json_name) as data_file:
            orl_test_corpus = json.load(data_file)

        # transform words to vocabulary ids
        # SRL data
        label_dict = {'O': 0}
        if argv.srl_labels_inventory == 'fix':
            label_dict = {'O': 0, 'B-V': 1, 'B-A0': 2, 'I-A0': 3, 'B-A1': 4, 'I-A1': 5, 'PAD': 6}

        #srl_train, label_dict = transform_srl_data(srl_train_corpus, argv.window_size, vocabulary, argv.exp_setup_id,
        #                                           fold, label_dict, argv.srl_labels_inventory, 'train')
        #srl_dev, label_dict = transform_srl_data(srl_dev_corpus, argv.window_size, vocabulary, argv.exp_setup_id,
        #                                         fold, label_dict, argv.srl_labels_inventory, None)
        #srl_test, label_dict = transform_srl_data(srl_test_corpus, argv.window_size, vocabulary, argv.exp_setup_id, fold,
        #                                          label_dict, argv.srl_labels_inventory, None)
        #label_dict_size = len(label_dict)
        #label_dict["<PAD>"] = label_dict_size

        #assert len(label_dict.keys()) == len(list(set(label_dict.keys())))

        #label_dict_inv = [0] * len(label_dict)
        #for w, i in zip(list(label_dict.keys()), list(label_dict.values())):
        #    label_dict_inv[i] = w

        # ORL data
        '''
        orl_train, _, _ = transform_orl_data(orl_train_corpus, vocabulary, argv.window_size, 'train', fold+1,
                                             argv.att_link_obligatory)
        orl_test, _, _ = transform_orl_data(orl_test_corpus, vocabulary, argv.window_size, 'test', fold+1,
                                            argv.att_link_obligatory)
        orl_dev, _, _ = transform_orl_data(orl_dev_corpus, vocabulary, argv.window_size, 'dev', fold+1,
                                           argv.att_link_obligatory)
        '''
        orl_train, _, _ = transform_orl_data(orl_train_corpus, vocabulary, argv.window_size, 'train',
                                             argv.exp_setup_id, argv.att_link_obligatory)
        orl_test, _, _ = transform_orl_data(orl_test_corpus, vocabulary, argv.window_size, 'test',
                                            argv.exp_setup_id, argv.att_link_obligatory)
        orl_dev, _, _ = transform_orl_data(orl_dev_corpus, vocabulary, argv.window_size, 'dev',
                                           argv.exp_setup_id, argv.att_link_obligatory)

        print(fold, len(orl_train), len(orl_test), len(orl_dev), len(vocabulary))
        with open('prep/train.pkl', 'wb') as fout:
            pickle.dump(orl_train, fout)
        with open('prep/dev.pkl', 'wb') as fout:
            pickle.dump(orl_dev, fout)
        with open('prep/test.pkl', 'wb') as fout:
            pickle.dump(orl_test, fout)
        with open('prep/vocab.pkl', 'wb') as fout:
            pickle.dump(vocabulary, fout)
        exit(0)


        # construct data iters
        train_iter = train_data_iter(srl_train, orl_train, argv.batch_size, vocabulary, label_dict, argv.n_epochs)

        srl_train_iter_eval = eval_data_iter(srl_train, argv.batch_size, vocabulary, label_dict)
        srl_test_iter = eval_data_iter(srl_test, argv.batch_size, vocabulary, label_dict)
        srl_dev_iter = eval_data_iter(srl_dev, argv.batch_size, vocabulary, label_dict)

        orl_train_iter_eval = eval_data_iter(orl_train, argv.batch_size, vocabulary, None)
        orl_test_iter = eval_data_iter(orl_test, argv.batch_size, vocabulary, None)
        orl_dev_iter = eval_data_iter(orl_dev, argv.batch_size, vocabulary, None)

        # start training
        logging.info('start training...')
        parser.add_argument('--fold', type=int, default=fold, help='current fold')
        parser.add_argument('--embeddings', default=embeddings, help='embeddings')
        parser.add_argument('--vocabulary', default=vocabulary, help='vocabulary')
        parser.add_argument('--srl_label_dict', default=label_dict, help='srl label dictionary')
        parser.add_argument('--srl_label_dict_inv', default=label_dict_inv, help='srl inverse label dictionary')
        parser.add_argument('--n_classes_srl', default=len(label_dict), help='the number of SRL labels')
        parser.add_argument('--n_classes_orl', default=8, help='the number of the ORL label')
        parser.add_argument('--srl_train_iter_eval', default=srl_train_iter_eval, help='srl train iterator')
        parser.add_argument('--orl_train_iter_eval', default=orl_train_iter_eval, help='orl train iterator')
        parser.add_argument('--train_iter', default=train_iter, help='train iterator')
        parser.add_argument('--srl_dev_iter', default=srl_dev_iter, help='srl dev iterator')
        parser.add_argument('--srl_test_iter', default=srl_test_iter, help='srl test iterator')
        parser.add_argument('--orl_dev_iter', default=orl_dev_iter, help='orl dev iterator')
        parser.add_argument('--orl_test_iter', default=orl_test_iter, help='orl test iterator')
        parser.add_argument('--eval_every', default=int(2*len(orl_train_iter_eval)), help='eval every eval_every iter')
        parser.add_argument('--early_stop', default=25, help='stop if not better in eval_every*early_stop'
                                                             'iterations')
        argv = parser.parse_args()

        #train(argv)
