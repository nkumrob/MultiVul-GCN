import warnings
from sklearn import metrics, model_selection
from models.GCN_baseline import GCN_Baseline
from models.gat import GAT
from load_data import GraphData, collect_batch
from parser_1 import _parser
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import os
import time
import numpy as np
import matplotlib
from DataReader import DataReader
from models.GCN_new import GCN_NEW
from models.multigraph import MGCN

matplotlib.use('agg')
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    print('using torch', torch.__version__)
    args = _parser()
    args.filters = list(map(int, args.filters.split(',')))
    args.lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))
    for arg in vars(args):
        print(arg, getattr(args, arg))

    n_folds = args.folds
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rnd_state = np.random.RandomState(args.seed)

    print('Loading training_data...')

    # Creating datareader object and reading the training data file from directory

    datareader = DataReader(data_dir='./training_data/%s/' % args.dataset, rnd_state=rnd_state,
                            use_cont_node_attr=args.use_cont_node_attr, folds=args.folds)

    # train and test
    result_folds = []
    for fold_id in range(n_folds):
        loaders = []
        for split in ['train', 'test']:
            graph_data = GraphData(
                fold_id=fold_id, datareader=datareader, split=split)
            loader = DataLoader(graph_data, batch_size=args.batch_size, shuffle=split.find('train') >= 0,
                                num_workers=args.threads, collate_fn=collect_batch)
            loaders.append(loader)
        print('FOLD {}, train {}, test {}'.format(
            fold_id, len(loaders[0].dataset), len(loaders[1].dataset)))

        if args.model == 'gat':
            model = GAT(nfeat=loaders[0].dataset.num_features,
                        nhid=64,
                        nclass=loaders[0].dataset.num_classes,
                        dropout=args.dropout,
                        alpha=args.alpha,
                        nheads=args.multi_head).to(args.device)

        elif args.model == 'GCN_baseline':
            model = GCN_Baseline(n_feature=loaders[0].dataset.num_features,
                                 n_hidden=64,
                                 n_class=loaders[0].dataset.num_classes,
                                 dropout=args.dropout).to(args.device)

        elif args.model == 'GCN_new':
            model = GCN_NEW(in_features=loaders[0].dataset.num_features,
                            out_features=loaders[0].dataset.num_classes,
                            n_hidden=args.n_hidden,
                            filters=args.filters,
                            dropout=args.dropout,
                            adj_sq=args.adj_sq,
                            scale_identity=args.scale_identity).to(args.device)

        elif args.model == 'multigraph':
            model = MGCN(in_features=loaders[0].dataset.num_features,
                         out_features=loaders[0].dataset.num_classes,
                         n_relations=2,
                         n_hidden=args.n_hidden,
                         n_hidden_edge=args.n_hidden_edge,
                         filters=args.filters,
                         dropout=args.dropout,
                         adj_sq=args.adj_sq,
                         scale_identity=args.scale_identity).to(args.device)

        else:
            raise NotImplementedError(args.model)

        print('Initialize model...')

        train_parameters = list(
            filter(lambda p: p.requires_grad, model.parameters()))
        print('N trainable parameters:', np.sum(
            [p.numel() for p in train_parameters]))
        optimizer = optim.Adam(train_parameters, lr=args.lr, betas=(
            0.5, 0.999), weight_decay=args.wd)
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, args.lr_decay_steps, gamma=0.1)  # dynamic adjustment lr
        # loss_fn = F.nll_loss  # model is gat or gcn, use this
        loss_fn = F.cross_entropy  # when model is gcn_new, and multigraph use this

        def train(train_loader):
            model.train()
            start = time.time()
            train_loss, n_samples = 0, 0
            for batch_idx, data in enumerate(train_loader):
                for i in range(len(data)):
                    data[i] = data[i].to(args.device)
                optimizer.zero_grad()
                # output = model(data[0], data[1])  # for gat and gcn baseline
                # when model is gcn_new, and multigraph use this
                output = model(data)
                loss = loss_fn(output, data[4])
                loss.backward()
                optimizer.step()
                time_iter = time.time() - start
                train_loss += loss.item() * len(output)
                n_samples += len(output)
                scheduler.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} (avg: {:.6f})  sec/iter: {:.4f}'.format(
                epoch + 1, n_samples, len(train_loader.dataset), 100. *
                (batch_idx + 1) / len(train_loader),
                loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1)))

        def test(test_loader):
            model.eval()
            start = time.time()
            test_loss, n_samples, count = 0, 0, 0
            tn, fp, fn, tp = 0, 0, 0, 0  # calculate recall, precision, F1 score
            accuracy, recall, precision, F1 = 0, 0, 0, 0

            for batch_idx, data in enumerate(test_loader):
                for i in range(len(data)):
                    data[i] = data[i].to(args.device)
                # when model is gcn baseline or gat, use this
                # output = model(data[0], data[1])
                # when model is gcn_new, and multigraph use this
                output = model(data)
                loss = loss_fn(output, data[4], reduction='sum')
                test_loss += loss.item()
                n_samples += len(output)
                count += 1
                pred = output.detach().cpu().max(1, keepdim=True)[1]

                for k in range(len(pred)):
                    if (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (
                            np.array(data[4].detach().cpu()[k]).tolist() == 1):
                        # TP predict == 1 & label == 1
                        tp += 1
                        continue
                    elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                            np.array(data[4].detach().cpu()[k]).tolist() == 0):
                        # TN predict == 0 & label == 0
                        tn += 1
                        continue
                    elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                            np.array(data[4].detach().cpu()[k]).tolist() == 1):
                        # FN predict == 0 & label == 1
                        fn += 1
                        continue
                    elif (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (
                            np.array(data[4].detach().cpu()[k]).tolist() == 0):
                        # FP predict == 1 & label == 0
                        fp += 1
                        continue

                accuracy += metrics.accuracy_score(
                    data[4], pred.view_as(data[4]))
                recall += metrics.recall_score(data[4], pred.view_as(data[4]))
                precision += metrics.precision_score(
                    data[4], pred.view_as(data[4]))
                F1 += metrics.f1_score(data[4], pred.view_as(data[4]))

            print('\nTrue Positive = ', tp)
            print('\nTrue Negative = ', tn)
            print('\nFalse Positive = ', fp)
            print('\nFalse Negative = ', fn, '\n')
            accuracy = 100. * accuracy / count
            recall = 100. * recall / count
            precision = 100. * precision / count
            F1 = 100. * F1 / count
            FPR = fp / (fp + tn)
            TPR = tp / (tp + fn)

            print(
                'Test set (epoch {}): \n   Average loss: {:.4f}, \n   Accuracy: ({:.2f}%),'
                '\n   Recall: ({:.2f}%), \n   Precision: ({:.2f}%), \n   F1-Score: ({:.2f}%), '
                '\n   TPR: ({:.2f}%), \n   FPR: ({:.2f}%)  \n   sec/iter: {:.4f}\n'.format(
                    epoch + 1, test_loss / n_samples, accuracy, recall, precision, F1, TPR, FPR,
                    (time.time() - start) / len(test_loader))
            )

            return accuracy, recall, precision, F1, FPR, TPR

        for epoch in range(args.epochs):
            train(loaders[0])
        accuracy, recall, precision, F1, FPR, TPR = test(loaders[1])
        result_folds.append([accuracy, recall, precision, F1, FPR, TPR])

    accuracy_list = []
    recall_list = []
    precision_list = []
    F1_list = []
    FPR_list = []
    TPR_list = []

    for i in range(len(result_folds)):
        accuracy_list.append(result_folds[i][0])
        recall_list.append(result_folds[i][1])
        precision_list.append(result_folds[i][2])
        F1_list.append(result_folds[i][3])
        FPR_list.append(result_folds[i][4])
        TPR_list.append(result_folds[i][5])

    print(
        '{}-fold cross validation with average accuracy(+- Standard deviation): {}% ({}%), Recall (+- Standard deviation): {}% ({}%), Precision (+- Standard deviation): {}% ({}%), '
        'F1-Score (+- Standard deviation): {}% ({}%), FPR (+- fpr): {}% ({}%), TPR (+- fpr): {}% ({}%)'.format(
            n_folds, np.mean(accuracy_list), np.std(
                accuracy_list), np.mean(recall_list), np.std(recall_list),
            np.mean(precision_list), np.std(precision_list), np.mean(
                F1_list), np.std(F1_list), np.mean(FPR_list),
            np.std(FPR_list), np.mean(TPR_list), np.std(TPR_list))
    )

    import matplotlib.pyplot as plt

    plt.plot(FPR_list, TPR_list)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
