import torch
import torch.nn as nn
from models.TabNets import TabEncoder
import torch.nn.functional as F


class SimCLR(nn.Module):
    def __init__(self, args):
        super(SimCLR, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.nmb_prototypes = args.nmb_prototypes
        self.encoder = TabEncoder(self.input_dim, self.hidden_dim)
        self.prototypes = nn.Linear(self.hidden_dim, self.nmb_prototypes, bias=False)

    def forward(self, x):
        embedding, head_feature = self.encoder(x)
        prototype = self.prototypes(head_feature)
        return head_feature, prototype

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.args.epsilon).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * self.args.world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.args.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            # dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def get_feature(self, data_loader):
        features = torch.zeros(len(data_loader.dataset), self.hidden_dim).cuda()
        labels = torch.zeros(len(data_loader.dataset)).cuda()
        proto_labels = torch.zeros(len(data_loader.dataset)).cuda()
        for x, y, ind, rand_sample in data_loader:
            with torch.no_grad():
                x = x.type(torch.FloatTensor).cuda()
                feat, head_feature = self.encoder(x)
                prototype = self.prototypes(head_feature)
                ind = ind.type(torch.LongTensor).cuda()
                proto_labels[ind] = torch.argmax(self.distributed_sinkhorn(prototype), dim=1).type(torch.FloatTensor).cuda()
                features[ind] = feat
                y = y.type(torch.FloatTensor).cuda()
                labels[ind] = y
        return features.cpu(), labels.cpu(), proto_labels.cpu()

    def loss_cal(self, x, x_aug, hard_q, neg_q, mode, hard_q_label):
        temperature = 0.07
        batch_size = x.size(0)
        assert mode in ['warmup', 'train']

        if mode == 'warmup':
            labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            labels = labels.to(x.device)

            out_1 = F.normalize(x, dim=1)
            out_2 = F.normalize(x_aug, dim=1)
            out = torch.cat([out_1, out_2], dim=0)
            sim_matrix = torch.mm(out, out.t().contiguous())
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()

            sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.shape[0], -1)
            labels = labels.masked_select(mask).view(labels.shape[0], -1)
            positives = sim_matrix.masked_select(labels.bool()).view(labels.shape[0], -1)
            negatives = sim_matrix.masked_select(~labels.bool()).view(labels.shape[0], -1)
            logits = torch.cat([positives, negatives], dim=1)
            logits = logits / temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(x.device)

            criterion = nn.CrossEntropyLoss().to(x.device)
            return criterion(logits, labels)
        else:
            hard_q_label = torch.cat([hard_q_label[0], hard_q_label[1]], dim=0)
            hard_q = torch.cat([hard_q[0], hard_q[1]], dim=0)
            neg_q = torch.cat([neg_q[0], neg_q[1]], dim=0)
            hard_q = hard_q.reshape(-1, 1).repeat(1, batch_size * 2)
            neg_q = neg_q.reshape(1, -1).repeat(batch_size * 2, 1)
            out_1 = F.normalize(x, dim=1)
            out_2 = F.normalize(x_aug, dim=1)
            out = torch.cat([out_1, out_2], dim=0)
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()

            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            proto_dist = 1.0 - F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
            K = w.shape[0]
            N, D = out.shape
            alpha_weight = torch.zeros(K, N).cuda()
            for i in range(K):
                alpha_weight[i] = proto_dist[i][hard_q_label]
                alpha_weight[i][i == hard_q_label] = 2.0
                a_min = torch.min(alpha_weight[i], dim=0).values
                alpha_weight[i][i == hard_q_label] = a_min
            a_min = torch.min(alpha_weight, dim=1).values.unsqueeze(-1)
            a_max = torch.max(alpha_weight, dim=1).values.unsqueeze(-1)
            alpha_weight = (alpha_weight - a_min) / (a_max - a_min + 1e-12)

            alpha_weight = alpha_weight.unsqueeze(-1)
            alpha_weight = alpha_weight.repeat(1, 1, D)

            if self.args.w_mode == "anchor":
                out_unsq_0 = out.unsqueeze(1)
                out_unsq_0 = out_unsq_0.repeat(1, N, 1)
                out_unsq_1 = out.unsqueeze(0)
                out_unsq_1 = out_unsq_1.repeat(N, 1, 1)
                alpha_weight_0 = alpha_weight[hard_q_label]
                weighting_out = out_unsq_1 * (1 - alpha_weight_0) + out_unsq_0 * alpha_weight_0
                weighting_out = nn.functional.normalize(weighting_out, dim=-1, p=2)
                weighting_out = weighting_out.permute(0, 2, 1)
            else:
                out_unsq = out.unsqueeze(0)
                centers = w.unsqueeze(1)
                out_unsq = out_unsq.repeat(K, 1, 1)
                centers = centers.repeat(1, N, 1)
                o_plus_c = out_unsq * (1 - alpha_weight) + centers * alpha_weight
                o_plus_c = nn.functional.normalize(o_plus_c, dim=-1, p=2)
                weighting_out = o_plus_c[hard_q_label]
                weighting_out = weighting_out.permute(0, 2, 1)

            sim_matrix_for_neg = torch.exp(torch.einsum('nc,nck->nk', [out, weighting_out]) / temperature)
            tmp_mask = (hard_q & neg_q).bool()
            hard_q_label = hard_q_label.unsqueeze(-1)
            mask_pos = torch.eq(hard_q_label, hard_q_label.permute(1, 0)).float().cuda()

            mask_neg = tmp_mask
            sim_matrix = sim_matrix * mask_pos + sim_matrix_for_neg * mask_neg
            mask = mask.type(torch.FloatTensor).cuda()
            sim_matrix = sim_matrix * mask

            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            pos = torch.cat([pos, pos], dim=0)
            loss = (- torch.log(pos / sim_matrix.sum(dim=-1))).mean()
            return loss

