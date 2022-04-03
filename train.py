from torch.autograd import grad
from cords.utils.data.datasets.SL.builder import SSTDataset, loadGloveModel, GlueDataset
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from cords.utils.models import LSTMClassifier, SimplifiedClassifier, BiLSTMClassifier
import argparse
import time
from cords.utils.data.dataloader.SL.adaptive import GLISTERDataLoader, OLRandomDataLoader, CRAIGDataLoader, GradMatchDataLoader, RandomDataLoader

import logging
from dotmap import DotMap
import torchtext
import os
import utils
from datasets import load_dataset
import numpy as np

'''
source: https://github.com/VictoriousRaptor/sst-clf-torch
'''

def collate_fn_weighted(data):
    """Pad data in a batch.
    Parameters
    ----------
    data : list((tensor, int), )
        data and label in a batch
    Returns
    -------
    tuple(tensor, tensor)
    """
    # data: [(tensor, label), ...]
    max_len = max([i[0].shape[0] for i in data])
    labels = torch.tensor([i[1] for i in data], dtype=torch.long)
    weights = torch.tensor([i[2] for i in data], dtype=torch.float)
    padded = torch.zeros((len(data), max_len), dtype=torch.long)
    # randomizing might be better
    for i, _ in enumerate(padded):
        padded[i][:data[i][0].shape[0]] = data[i][0]
    return padded, labels, weights

def collate_fn(data):
    """Pad data in a batch.
    Parameters
    ----------
    data : list((tensor, int), )
        data and label in a batch
    Returns
    -------
    tuple(tensor, tensor)
    """
    max_len = max([i[0].shape[0] for i in data])
    labels = torch.tensor([i[1] for i in data], dtype=torch.long)
    padded = torch.zeros((len(data), max_len), dtype=torch.long)
    # randomizing might be better
    for i, _ in enumerate(padded):
        padded[i][:data[i][0].shape[0]] = data[i][0]
    return padded, labels


def evaluation(data_iter, model, args):
    # Evaluating the given model
    model.eval()
    with torch.no_grad():
        corrects = 0
        avg_loss = 0
        for data, label in data_iter:
            sentences = data.to(args.device, non_blocking=True)
            labels = label.to(args.device, non_blocking=True)
            logit = model(sentences)
            corrects += (torch.max(logit, 1)[1].view(labels.size()).data == labels.data).sum().item()
        size = len(data_iter.dataset)
        model.train()
        return 100.0 * corrects / size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--label_num', type=int, default=2, help='Target label numbers')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--wordvec_dim', type=int, default=300, help='Dimension of GloVe vectors')
    parser.add_argument('--model_name', type=str, default='BiLSTM', help='Which model to use')
    parser.add_argument('--ss', type=int, default=0, help='which ss to use. 0 = no ss, 1 = gradmatch, 2 = gradmatchpb')
    parser.add_argument('--fraction', type=float, default=0.1, help='fraction in subset selection')
    parser.add_argument('--select_every', type=int, default=3, help='perform subset selection every _ epochs')
    parser.add_argument('--dataset_path', type=str, default='/home/x/xuxilie/GLUE/SST-2/original/', help='PATH to dataset')
    parser.add_argument('--method', type=str, default='GradMatch')
    parser.add_argument('--emb_weight', type=float, default=0)
    parser.add_argument('--dataset', type=str, default='sst2')
    parser.add_argument('--max_len', type=int, default=36)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    args.ss = int(args.ss)
    start = time.time()
    wordvec = loadGloveModel(r'/home/x/xuxilie/glove/glove.6B.' + str(args.wordvec_dim) + 'd.txt')
    args.weight = torch.from_numpy(wordvec.values) 
    # args.weight = torch.from_numpy(wordvec.values, dtype=torch.float)  # word embedding for the embedding layer

    # Datasets
    if args.dataset == 'sst2':
        train_set = load_dataset('glue', args.dataset, split='train')
        # test_set = load_dataset('glue', args.dataset, split='test')
        validation_set = load_dataset('glue', args.dataset, split='validation')

        training_iter = utils.create_dataset(train_set['sentence'][:int(len(train_set['sentence'])*0.8)], train_set['label'][:int(len(train_set['sentence'])*0.8)], batch_size=args.batch_size, shuffle=True, sentence_list2=None, max_len=args.max_len, fraction=0.8, begin=0)
        testing_iter = utils.create_dataset(train_set['sentence'][int(len(train_set['sentence'])*0.8):], train_set['label'][int(len(train_set['sentence'])*0.8):], batch_size=args.batch_size, shuffle=True, sentence_list2=None, max_len=args.max_len, fraction=0.2, begin=0.8)
        validation_iter = utils.create_dataset(validation_set['sentence'], validation_set['label'], batch_size=args.batch_size, shuffle=True, sentence_list2=None, max_len=args.max_len)

        args.length = args.max_len
        # training_set = SSTDataset(args.dataset_path, 'train', args.label_num, args.wordvec_dim, wordvec)
        # testing_set = SSTDataset(args.dataset_path, 'test', args.label_num, args.wordvec_dim, wordvec)
        # validation_set = SSTDataset(args.dataset_path, 'dev', args.label_num, args.wordvec_dim, wordvec)

        # training_iter = DataLoader(dataset=training_set,
        #                         batch_size=args.batch_size,
        #                         num_workers=0, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        # testing_iter = DataLoader(dataset=testing_set,
        #                         batch_size=args.batch_size,
        #                         num_workers=0, collate_fn=collate_fn, pin_memory=True)
        # validation_iter = DataLoader(dataset=validation_set,
        #                             batch_size=args.batch_size,
        #                             num_workers=0, collate_fn=collate_fn, pin_memory=True)

    elif args.dataset == 'cola':
        train_set = load_dataset('glue', args.dataset, split='train')
        # test_set = load_dataset('glue', args.dataset, split='test')
        validation_set = load_dataset('glue', args.dataset, split='validation')

        training_iter = utils.create_dataset(train_set['sentence'][:int(len(train_set['sentence'])*0.8)], train_set['label'][:int(len(train_set['sentence'])*0.8)], batch_size=args.batch_size, shuffle=True, sentence_list2=None, max_len=args.max_len, fraction=0.8, begin=0)
        validation_iter = utils.create_dataset(train_set['sentence'][int(len(train_set['sentence'])*0.8):], train_set['label'][int(len(train_set['sentence'])*0.8):], batch_size=args.batch_size, shuffle=True, sentence_list2=None, max_len=args.max_len, fraction=0.2, begin=0.8)
        testing_iter = utils.create_dataset(validation_set['sentence'], validation_set['label'], batch_size=args.batch_size, shuffle=True, sentence_list2=None, max_len=args.max_len)

        args.length = args.max_len

    elif args.dataset == 'rte':
        name = ['sentence1', 'sentence2']
        train_set = load_dataset('glue', args.dataset, split='train')
        # test_set = load_dataset('glue', args.dataset, split='test')s
        validation_set = load_dataset('glue', args.dataset, split='validation')

        training_iter = utils.create_dataset(train_set[name[0]][:int(len(train_set[name[0]])*0.8)], train_set['label'][:int(len(train_set[name[0]])*0.8)], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set[name[1]][:int(len(train_set[name[0]])*0.8)], max_len=args.max_len, fraction=0.8, begin=0)
        testing_iter = utils.create_dataset(train_set[name[0]][int(len(train_set[name[0]])*0.8):], train_set['label'][int(len(train_set[name[0]])*0.8):], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set[name[1]][int(len(train_set[name[0]])*0.8):], max_len=args.max_len, fraction=0.2, begin=0.8)
        validation_iter = utils.create_dataset(validation_set[name[0]], validation_set['label'], batch_size=args.batch_size, shuffle=True, sentence_list2=validation_set[name[1]], max_len=args.max_len)

        args.length = args.max_len * 2

    elif args.dataset == 'wnli':
        name = ['sentence1', 'sentence2']
        train_set = load_dataset('glue', args.dataset, split='train')
        # test_set = load_dataset('glue', args.dataset, split='test')s
        validation_set = load_dataset('glue', args.dataset, split='validation')

        training_iter = utils.create_dataset(train_set[name[0]][:int(len(train_set[name[0]])*0.8)], train_set['label'][:int(len(train_set[name[0]])*0.8)], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set[name[1]][:int(len(train_set[name[0]])*0.8)], max_len=args.max_len, fraction=0.8, begin=0)
        testing_iter = utils.create_dataset(train_set[name[0]][int(len(train_set[name[0]])*0.8):], train_set['label'][int(len(train_set[name[0]])*0.8):], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set[name[1]][int(len(train_set[name[0]])*0.8):], max_len=args.max_len, fraction=0.2, begin=0.8)
        validation_iter = utils.create_dataset(validation_set[name[0]], validation_set['label'], batch_size=args.batch_size, shuffle=True, sentence_list2=validation_set[name[1]], max_len=args.max_len)

        args.length = args.max_len * 2

    elif args.dataset == 'mrpc' or  args.dataset == 'stsb':
        train_set = load_dataset('glue', args.dataset, split='train')
        test_set = load_dataset('glue', args.dataset, split='test')
        validation_set = load_dataset('glue', args.dataset, split='validation')

        training_iter = utils.create_dataset(train_set['sentence1'], train_set['label'], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set['sentence2'], max_len=args.max_len)
        testing_iter = utils.create_dataset(test_set['sentence1'], test_set['label'], batch_size=args.batch_size, shuffle=True, sentence_list2=test_set['sentence2'], max_len=args.max_len)
        validation_iter =  utils.create_dataset(validation_set['sentence1'], validation_set['label'], batch_size=args.batch_size, shuffle=True, sentence_list2=validation_set['sentence2'], max_len=args.max_len)

        args.length = args.max_len * 2

    elif args.dataset == 'qqp':
        name = ['question1', 'question2']
        train_set = load_dataset('glue', args.dataset, split='train')
        # test_set = load_dataset('glue', args.dataset, split='test')s
        validation_set = load_dataset('glue', args.dataset, split='validation')

        training_iter = utils.create_dataset(train_set[name[0]][:int(len(train_set[name[0]])*0.7)], train_set['label'][:int(len(train_set[name[0]])*0.7)], batch_size=args.batch_size*32, shuffle=True, sentence_list2=train_set[name[1]][:int(len(train_set[name[0]])*0.7)], max_len=args.max_len, fraction=0.8, begin=0)
        testing_iter = utils.create_dataset(train_set[name[0]][int(len(train_set[name[0]])*0.8):], train_set['label'][int(len(train_set[name[0]])*0.8):], batch_size=args.batch_size*32, shuffle=True, sentence_list2=train_set[name[1]][int(len(train_set[name[0]])*0.8):], max_len=args.max_len, fraction=0.2, begin=0.8)
        validation_iter = utils.create_dataset(validation_set[name[0]], validation_set['label'], batch_size=args.batch_size*32, shuffle=True, sentence_list2=validation_set[name[1]], max_len=args.max_len)

        # if args.method == 'GLISTERD' or args.method == 'CRAIG':
        #     training_iter = utils.create_dataset(train_set[name[0]][:int(len(train_set[name[0]])*0.15)], train_set['label'][:int(len(train_set[name[0]])*0.15)], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set[name[1]][:int(len(train_set[name[0]])*0.15)], max_len=args.max_len, fraction=0.8, begin=0)
        # else:
        #     training_iter = utils.create_dataset(train_set[name[0]][:int(len(train_set[name[0]])*0.8)], train_set['label'][:int(len(train_set[name[0]])*0.8)], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set[name[1]][:int(len(train_set[name[0]])*0.8)], max_len=args.max_len, fraction=0.8, begin=0)
        # testing_iter = utils.create_dataset(train_set[name[0]][int(len(train_set[name[0]])*0.8):], train_set['label'][int(len(train_set[name[0]])*0.8):], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set[name[1]][int(len(train_set[name[0]])*0.8):], max_len=args.max_len, fraction=0.2, begin=0.8)
        # validation_iter = utils.create_dataset(validation_set[name[0]][:int(len(validation_set[name[0]])*0.2)], validation_set['label'][:int(len(validation_set[name[0]])*0.2)], batch_size=args.batch_size, shuffle=True, sentence_list2=validation_set[name[1]][:int(len(validation_set[name[0]])*0.2)], max_len=args.max_len)
        
        args.length = args.max_len * 2

    # elif args.dataset == 'mnli':
    #     name = ['premiss', 'hypothesis']
    #     train_set = load_dataset('glue', args.dataset, split='train')
    #     test_set = load_dataset('glue', args.dataset, split='test')
    #     validation_set = load_dataset('glue', args.dataset, split='validation')

    #     training_iter = utils.create_dataset(train_set[name[0]], train_set['label'], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set[name[1]], max_len=args.max_len)
    #     testing_iter = utils.create_dataset(test_set[name[0]], test_set['label'], batch_size=args.batch_size, shuffle=True, sentence_list2=test_set[name[1]], max_len=args.max_len)
    #     validation_iter = utils.create_dataset(validation_set[name[0]], validation_set['label'], batch_size=args.batch_size, shuffle=True, sentence_list2=validation_set[name[1]], max_len=args.max_len)

    elif args.dataset == 'qnli':
        name = ['question', 'sentence']
        train_set = load_dataset('glue', args.dataset, split='train')
        # test_set = load_dataset('glue', args.dataset, split='test')s
        validation_set = load_dataset('glue', args.dataset, split='validation')

        training_iter = utils.create_dataset(train_set[name[0]][:int(len(train_set[name[0]])*0.8)], train_set['label'][:int(len(train_set[name[0]])*0.8)], batch_size=args.batch_size*32, shuffle=True, sentence_list2=train_set[name[1]][:int(len(train_set[name[0]])*0.8)], max_len=args.max_len, fraction=0.8, begin=0)
        testing_iter = utils.create_dataset(train_set[name[0]][int(len(train_set[name[0]])*0.8):], train_set['label'][int(len(train_set[name[0]])*0.8):], batch_size=args.batch_size*32, shuffle=True, sentence_list2=train_set[name[1]][int(len(train_set[name[0]])*0.8):], max_len=args.max_len, fraction=0.2, begin=0.8)
        validation_iter = utils.create_dataset(validation_set[name[0]], validation_set['label'], batch_size=args.batch_size*32, shuffle=True, sentence_list2=validation_set[name[1]], max_len=args.max_len)

        # if args.method == 'GLISTERD' or args.method == 'CRAIG':
        #     training_iter = utils.create_dataset(train_set[name[0]][:int(len(train_set[name[0]])*0.45)], train_set['label'][:int(len(train_set[name[0]])*0.45)], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set[name[1]][:int(len(train_set[name[0]])*0.45)], max_len=args.max_len, fraction=0.8, begin=0)
        # else:
        #     training_iter = utils.create_dataset(train_set[name[0]][:int(len(train_set[name[0]])*0.8)], train_set['label'][:int(len(train_set[name[0]])*0.8)], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set[name[1]][:int(len(train_set[name[0]])*0.8)], max_len=args.max_len, fraction=0.8, begin=0)
        # testing_iter = utils.create_dataset(train_set[name[0]][int(len(train_set[name[0]])*0.8):], train_set['label'][int(len(train_set[name[0]])*0.8):], batch_size=args.batch_size, shuffle=True, sentence_list2=train_set[name[1]][int(len(train_set[name[0]])*0.8):], max_len=args.max_len, fraction=0.2, begin=0.8)
        # validation_iter = utils.create_dataset(validation_set[name[0]][:int(len(validation_set[name[0]])*0.1)], validation_set['label'][:int(len(validation_set[name[0]])*0.1)], batch_size=args.batch_size, shuffle=True, sentence_list2=validation_set[name[1]][:int(len(validation_set[name[0]])*0.1)], max_len=args.max_len)
        
        args.length = args.max_len * 2

    print('Time for loading glove',args.wordvec_dim,'and creating torch dataloaders:', time.time() - start)

    model_name = args.model_name
    print("Model:", model_name)

    # Select model
    if model_name == 'BiLSTM':
        model = BiLSTMClassifier(args.label_num, args.wordvec_dim, '/home/x/xuxilie/glove/', num_layers=args.num_layers).to(device)
        model_coreset = SimplifiedClassifier(args.label_num, args.wordvec_dim, '/home/x/xuxilie/glove/').to(device)
    elif model_name == 'LSTM':
        model = LSTMClassifier(args.label_num, args.wordvec_dim, '/home/x/xuxilie/glove/', num_layers=args.num_layers).to(device)
        model_coreset = SimplifiedClassifier(args.label_num, args.wordvec_dim, '/home/x/xuxilie/glove/').to(device)
    del wordvec  # Save some memory

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_coreset = torch.optim.Adam(model_coreset.parameters(), lr=args.lr)
    
    if args.ss == 1:
        dss_args=dict(type="GradMatch",
                            fraction=args.fraction,
                            select_every=args.select_every,
                            lam=0.5,
                            selection_type='PerClassPerGradient',
                            v1=True,
                            valid=False,
                            kappa=0,
                            eps=1e-100,
                            linear_layer=True,
                            model=model,
                            loss=criterion_nored,
                            eta = args.lr,
                            num_classes = args.label_num,
                            device = args.device
                            )
    elif args.ss == 2:
        dss_args=dict(type="GradMatchPB",
                            fraction=args.fraction,
                            select_every=args.select_every,
                            lam=0,
                            selection_type='PerBatch',
                            v1=True,
                            valid=False,
                            eps=1e-100,
                            linear_layer=True,
                            kappa=0,
                            model=model,
                            loss=criterion_nored,
                            eta = args.lr,
                            num_classes = args.label_num,
                            device = args.device
                            )
    elif args.ss == 3: # GLISTERD
        # if args.dataset == 'qnli':
        #     dss_args = dict(model=model_coreset,
        #                     loss=criterion_nored,
        #                     eta=0.01,
        #                     num_classes=args.label_num,
        #                     num_epochs=args.epoch,
        #                     # device=args.device,
        #                     device = torch.device('cpu'),
        #                     fraction=args.fraction,
        #                     select_every=args.select_every,
        #                     kappa=0,
        #                     linear_layer=False,
        #                     selection_type='PerBatch',
        #                     greedy='Stochastic',
        #                     emb_weight=args.emb_weight,
        #                     length = args.length,
        #                     emb_dim = args.wordvec_dim
        #                     )
        # else:
        dss_args = dict(model=model_coreset,
                            loss=criterion_nored,
                            eta=0.01,
                            num_classes=args.label_num,
                            num_epochs=args.epoch,
                            device=args.device,
                            # device = torch.device('cpu'),
                            fraction=args.fraction,
                            select_every=args.select_every,
                            kappa=0,
                            linear_layer=False,
                            selection_type='PerBatch',
                            greedy='Stochastic',
                            emb_weight=args.emb_weight,
                            length = args.length,
                            emb_dim = args.wordvec_dim
                            )
    
    elif args.ss == 4: # CRAIG
        # if args.dataset == 'qnli' or  args.dataset == 'sst2':
        #     dss_args = dict(model=model_coreset,
        #                 loss=criterion_nored,
        #                 num_classes=args.label_num,
        #                 num_epochs=args.epoch,
        #                 device=torch.device('cpu'),
        #                 fraction=args.fraction,
        #                 select_every=args.select_every,
        #                 kappa=0,
        #                 linear_layer=False,
        #                 selection_type='PerBatch',
        #                 optimizer='lazy',
        #                 if_convex=False
        #                 )
        # else:
        dss_args = dict(model=model_coreset,
                        loss=criterion_nored,
                        num_classes=args.label_num,
                        num_epochs=args.epoch,
                        device=args.device,
                        fraction=args.fraction,
                        select_every=args.select_every,
                        kappa=0,
                        linear_layer=False,
                        selection_type='PerBatch',
                        optimizer='lazy',
                        if_convex=False
                        )
    
    if args.ss > 0:
        # logger = logging.getLogger(__name__)
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='results/{}.log'.format(args.method),
                filemode='w')
                
        dss_args = DotMap(dss_args)
        if args.method == 'GradMatch':
            dataloader = GradMatchDataLoader(training_iter, validation_iter, dss_args, logger,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                collate_fn=collate_fn_weighted)
        elif args.method == 'GLISTERD':
            dataloader = GLISTERDataLoader(training_iter, validation_iter, dss_args, logger, 
                                            batch_size=args.batch_size, 
                                            shuffle=True,
                                            pin_memory=False, 
                                            collate_fn=collate_fn_weighted)
        
        elif args.method == 'CRAIG':
            dataloader = CRAIGDataLoader(training_iter, validation_iter, dss_args, logger, 
                                            batch_size=args.batch_size, 
                                            shuffle=True,
                                            pin_memory=False, 
                                            collate_fn=collate_fn_weighted)

        elif args.method == 'Random':
            dataloader = RandomDataLoader(training_iter, dss_args, logger, 
                                            batch_size=args.batch_size, 
                                            shuffle=True,
                                            pin_memory=False, 
                                            collate_fn=collate_fn_weighted)


    step = 0
    loss_sum = 0
    test_acc = []
    best_acc = 0
    best_epoch = 0

    timing = []
    # timing_process = []
    if args.ss > 0:
        for epoch in range(1, args.epoch+1):
            subtrn_loss = 0
            subtrn_correct = 0.0
            subtrn_total = 0.0
            model.train()
            start_time = time.time()
            # start_time_process = time.process_time()
            for _, (inputs, targets, weights) in enumerate(dataloader):
                inputs = inputs.to(args.device)
                targets = targets.to(args.device, non_blocking=True)
                weights = weights.to(args.device)
                # print(inputs.shape, targets.shape)
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs_coreset = model_coreset(inputs)
                losses = criterion_nored(outputs, targets)
                losses_coreset = criterion_nored(outputs_coreset, targets)
                loss = torch.dot(losses, weights / (weights.sum()))
                loss_coreset = torch.dot(losses_coreset, weights / (weights.sum()))
                loss.backward()
                loss_coreset.backward()
                # g = model.embedding.weight.grad
                subtrn_loss += loss.item()

                loss_sum += subtrn_loss
                if step % args.log_interval == 0:
                    print("epoch", epoch, end='  ')
                    print("avg loss: %.5f" % (loss_sum / args.log_interval))
                    loss_sum = 0
                    step = 0
                step += 1

                optimizer.step()
                optimizer_coreset.step()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
            
            epoch_time = time.time() - start_time
            timing.append(epoch_time)
            # timing_process.append(time.process_time()-start_time_process)
            
            acc = evaluation(testing_iter, model, args)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
            test_acc.append(acc)
            print('test acc {:.4f}'.format(acc))
            print('train acc {:.4f}, {:.4f}'.format(evaluation(training_iter, model, args), subtrn_correct/subtrn_total))
    elif args.ss == 0:
        for epoch in range(1, args.epoch + 1):
            model.train()
            start_time = time.time()
            # start_time_process = time.process_time()
            for data, label in training_iter:
                sentences = data.to(device, non_blocking=True)  # Asynchronous loading
                labels = label.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(sentences)
                loss = criterion(logits, labels)
                loss_sum += loss.data

                if step % args.log_interval == 0:
                    print("epoch", epoch, end='  ')
                    print("avg loss: %.5f" % (loss_sum / args.log_interval))
                    loss_sum = 0
                    step = 0
                step += 1

                loss.backward()
                optimizer.step()

            epoch_time = time.time() - start_time
            timing.append(epoch_time)
            # timing_process.append(time.process_time()-start_time_process)

            acc = evaluation(testing_iter, model, args)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
            test_acc.append(acc)
            print('test acc {:.4f}'.format(acc))
            print('train acc {:.4f}'.format(evaluation(training_iter, model, args)))

    print('')
    print('best: epoch {}, test acc {:.4f}'.format(best_epoch, best_acc))
    timing = [round(i, 5) for i in timing]
    print('total wall clock time(s):', sum(timing), 'avg wall clock time per epoch(s):',sum(timing)/len(timing))
    print('wall clock time for eah epoch:', timing)
    # print('total process time:', sum(timing_process), 'avg process time:',sum(timing_process)/len(timing_process))
    # print('process time for eah epoch:', timing_process)

    print("Parameters:")
    delattr(args, 'weight')
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    if args.ss > 0:
        print("dss_args:")
        delattr(dss_args, 'model')
        delattr(dss_args, 'loss')
        for attr, value in sorted(dss_args.__dict__.items()):
            print("\t{}={}".format(attr.upper(), value))


if __name__ == "__main__":
    main()