import torch
from torch import nn
import argparse
import utils
from utils import EarlyStopping
import logging
import os
import time
from models.SimCLR import SimCLR
from data_loader import DatasetSplit, ExampleDataset
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main(data_seed=0, random_sample_seed=1, logger=None, args=None, c_percent=0):
    main_worker(args, data_seed, random_sample_seed, logger, c_percent=c_percent)


def main_worker(args, data_seed, random_sample_seed, logger=None, c_percent=0):
    if args.seed != -1:
        print("init seed")
        logger.info("init seed")
        utils.init_seed(args)
    print(args)
    logger.info(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimCLR(args=args).to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.scheduler['lr'],
                                    momentum=args.scheduler['momentum'],
                                    weight_decay=float(args.scheduler['weight_decay']))
    train_data, train_label, test_data, test_label = DatasetSplit(dataset_name=args.dataset_name,
                                                                  c_percent=c_percent, seed=data_seed).get_dataset()
    train_dataset = ExampleDataset(train_data, train_label, random_sample_seed, args)
    test_dataset = ExampleDataset(test_data, test_label, random_sample_seed, args)
    eval_dataset = train_dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    args.queue_length -= args.queue_length % args.batch_size
    early_stopping = EarlyStopping()
    for epoch in range(args.epochs):
        queue = torch.zeros(args.aug_num, args.queue_length, args.hidden_dim).cuda()  # aug_num为增强操作的种类
        use_the_queue = True
        utils.adjust_learning_rate(optimizer, epoch, args)
        info_loss, swav_loss, sum_loss = train(train_loader, model, optimizer, device, queue, epoch, use_the_queue, args)
        print("Epoch:{:3d}/{:3d} Info Loss: {:.8f} Swav Loss: {:.8f} Loss:{:.8f}".format(epoch + 1, args.epochs, info_loss, swav_loss, sum_loss))

        """
        if epoch >= 10:
            early_stopping(loss, model, epoch, detect_res)
        if early_stopping.early_stop:
            # detect_res = detect(early_stopping.best_model, train_loader, test_loader, args)
            detect_res = early_stopping.detect_res
            print_res(detect_res, args, logger, early_stopping.best_epoch, stop="Early stopping! ", best="Best! ")
            break
        """
        if epoch >= args.epochs - 1:
            detect_res = detect(model, train_loader, test_loader, args)
            print_res(detect_res, args, logger, epoch, stop="Stopping! ", best="Best! ")
        else:
            pass


def print_res(detect_res, args, logger, epoch, stop="", best=""):
    print("{}Epoch:{:3d}/{:3d} Precision: {:.4f} Recall: {:.4f} F1: {:.4f} AUC: {:.4f}".format(
        stop, epoch + 1, args.epochs, detect_res[0], detect_res[1], detect_res[2], detect_res[3]))
    logger.info("{}Epoch:{:3d}/{:3d} Precision: {:.4f} Recall: {:.4f} F1: {:.4f} AUC: {:.4f}".format(
        best, epoch + 1, args.epochs, detect_res[0], detect_res[1], detect_res[2], detect_res[3]))


def detect(model, train_loader, test_loader, args):
    model.eval()
    feature_train, _, proto_labels = model.get_feature(train_loader)
    feature_test, labels, _ = model.get_feature(test_loader)
    solver = 'lsqr'
    distance_type = 'mahalanobis'
    gda = LinearDiscriminantAnalysis(solver=solver, shrinkage=None, store_covariance=True)

    gda.fit(feature_train, proto_labels)
    scores = confidence(feature_test, gda.means_, distance_type, gda.covariance_)
    ratios = labels.sum() / len(labels)
    return f_score(scores, labels.cpu().numpy(), ratios.cpu())


def confidence(features, means, distance_type, cov):
    features = features.numpy()
    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_classes = means.shape[0]
    if distance_type == "euclidean":
        cov = np.identity(num_features)
    features = features.reshape(num_samples, 1, num_features).repeat(num_classes, axis=1)
    means = means.reshape(1, num_classes, num_features).repeat(num_samples, axis=0)
    vectors = features - means
    cov_inv = np.linalg.inv(cov)
    bef_sqrt = np.matmul(np.matmul(vectors.reshape(num_samples, num_classes, 1, num_features), cov_inv),
                         vectors.reshape(num_samples, num_classes, num_features, 1)).squeeze()
    result = np.sqrt(bef_sqrt)
    result[np.isnan(result)] = 1e12  # solve nan
    if len(result.shape) > 1:
        return result.min(axis=1)
    else:
        return result


def f_score(scores, labels, ratio):
    scores = np.squeeze(scores)
    ratio = 100 - ratio * 100.0
    thresh = np.percentile(scores, ratio)
    y_pred = (scores > thresh).astype(int)
    y_true = labels.astype(int)
    precision, recall, f1, support = prf(y_true, y_pred, average='binary')
    scores = np.array(scores)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores)
    auc = metrics.auc(fpr, tpr)
    return precision, recall, f1, auc


def train(train_loader, model, optimizer, device, queue, epoch, use_the_queue, args):
    sample_num = len(train_loader.dataset)
    model.train()
    loss_sum = 0.0
    info_loss_sum = 0.0
    swav_loss_sum = 0.0
    for x, y, ind, rand_sample in train_loader:
        bs = y.shape[0]
        x = x.type(torch.FloatTensor).to(device)
        rand_sample = rand_sample.type(torch.FloatTensor).to(device)
        input_cat = torch.cat((x, rand_sample), dim=0)
        # normalize the prototypes
        with torch.no_grad():
            w = model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.prototypes.weight.copy_(w)
        head_feature, prototype = model(input_cat)
        hard_q, hard_q_label, neg_q, z, q_all = [], [], [], [], []
        swav_loss = 0
        for aug_id in range(args.aug_num):
            with torch.no_grad():
                out = prototype[bs * aug_id: bs * (aug_id + 1)].detach()
                if use_the_queue:
                    out = torch.cat((torch.mm(queue[aug_id], model.prototypes.weight.t()), out))
                    queue[aug_id, bs:] = queue[aug_id, :-bs].clone()
                    queue[aug_id, :bs] = head_feature[aug_id * bs: (aug_id + 1) * bs]
                q = distributed_sinkhorn(out, args)[-bs:]
                q_all.append(q.t())
                hard_q_label.append(torch.argmax(q, dim=1))
                hard_q.append(2 ** torch.argmax(q, dim=1))
                neg_q.append(torch.sum(2 ** torch.argsort(q, dim=1)[:, 0:-1], dim=1))
            for v in np.delete(np.arange(args.aug_num), aug_id):
                x_p = prototype[bs * v: bs * (v + 1)] / args.temperature
                z.append(head_feature[bs * v: bs * (v + 1), :])
                swav_loss -= torch.mean(torch.sum(q * F.log_softmax(x_p, dim=1), dim=1))
        if epoch < args.warm_up:
            contrastive_loss = model.loss_cal(z[1], z[0], hard_q, neg_q, 'warmup', hard_q_label)
        else:
            contrastive_loss = model.loss_cal(z[1], z[0], hard_q, neg_q, 'train', hard_q_label)
        loss = contrastive_loss + args.swav_alpha * swav_loss
        info_loss_sum += contrastive_loss.clone()
        swav_loss_sum += swav_loss.clone()
        loss_sum += loss.clone()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return info_loss_sum / sample_num, swav_loss_sum / sample_num, loss_sum / sample_num


@torch.no_grad()
def distributed_sinkhorn(out, args):
    Q = torch.exp(out / args.epsilon).t()
    B = Q.shape[1] * args.world_size
    K = Q.shape[0]
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()


def get_logger(logger_name, log_folder, log_file_name):
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(level=logging.INFO)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    time_segment = time.strftime("%Y%m%d%H%M%S", time.localtime())
    log_file_name = log_file_name + "_" + time_segment + ".txt"
    log_path = os.path.join(log_folder, log_file_name)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def exp_anom_score():
    iters = 20
    args = get_args()
    experiment_name = args.dataset_name
    log_folder = os.path.join("results", args.dataset_name, "score")
    logger = get_logger(experiment_name, log_folder, experiment_name)
    print("Ratio: {:2d}".format(0))
    logger.info("Ratio: {:2d}".format(0))
    for i in range(iters):
        print("Iteration: {:2d}/{:2d}".format(i + 1, iters))
        logger.info("Iteration: {:2d}/{:2d}".format(i + 1, iters))
        main(data_seed=i, random_sample_seed=i, logger=logger, args=args, c_percent=0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_thyroid.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='thyroid')
    args = utils.read_config_file(parser)
    return args


if __name__ == "__main__":
    exp_anom_score()



