# -*- coding: utf-8 -*-
import os
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn import metrics


class Trainer():
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.critic = ['auc', 'auc_pr', 'mrr']
        self.train_data = train

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=0.9, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        # self.criterion = nn.BCELoss()
        if self.params.loss:
            logging.info('using abs loss!')

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0

        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True,
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        model_params = list(self.graph_classifier.parameters())
        # self.graph_classifier.train()
        for b_idx, batch in enumerate(dataloader):
            # data_pos: g_dgl_pos, cg_dgl_pos, r_labels_pos
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            # print(data_pos[1].ndata['f1'])
            # print(data_pos[1].ndata['f2'])
            # print(data_pos[1].ndata['f3'])
            # print(data_neg[1].ndata['f3'])
            # print(data_neg[1].ndata['f3'].size())
            # print(data_pos[1].ndata['index1'])
            # print(data_pos[1].ndata['index2'])
            # print(data_pos[1].ndata['index3'])
            # print(len(data_neg[0].nodes()))
            # print(len(data_neg[1].nodes()))

            self.graph_classifier.train()
            self.optimizer.zero_grad()
            score_pos = self.graph_classifier(data_pos)  # torch.Size([16, 1])
            score_neg = self.graph_classifier(data_neg)  # torch.Size([16, 1])


            if self.params.loss == 1:
                # score_pos, score_neg = F.relu(score_pos), F.relu(score_neg)
                # loss = torch.abs(torch.sum(score_neg) + torch.sum(torch.clamp(self.params.margin - score_pos, min=0)))
                loss = torch.abs(torch.sum(torch.sum(score_neg, dim=1) + torch.clamp(self.params.margin - score_pos, min=0)))
            else:
                # 进行broadcast  正负样本进行排列组合
                loss = self.criterion(score_pos, score_neg.mean(dim=1), torch.Tensor([1]).to(device=self.params.device))
                # 对应位置
                # loss = self.criterion(score_pos.squeeze(1), score_neg.view(len(score_pos), -1).mean(dim=1),
                #                       torch.Tensor([1]).to(device=self.params.device))


            # # 二分类损失函数
            # score_pos_ = score_pos.squeeze(1)
            # score_neg_ = score_neg.squeeze(1)
            # tar_pos = torch.ones_like(score_pos_)
            # tar_neg = torch.zeros_like(score_neg_)
            # pred = torch.cat([score_pos_, score_neg_], dim=0)
            # target = torch.cat([tar_pos, tar_neg], dim=0)
            # loss = self.criterion(pred, target)


            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                # print(score_pos.size())
                # print(score_neg.size())
                all_scores += score_pos.squeeze(1).detach().cpu().tolist() + score_neg.squeeze(1).detach().cpu().tolist()
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                total_loss += loss

            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                result = self.valid_evaluator.eval()
                # self.graph_classifier.train()   # 验证之后转换为train模式

                logging.info('Performance: ' + str(result) + 'in ' + str(time.time() - tic))

                if result[self.critic[self.params.critic]] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result[self.critic[self.params.critic]]
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(
                            f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result[self.critic[self.params.critic]]

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, auc, auc_pr, weight_norm

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            self.params.epoch = epoch
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start
            logging.info(
                f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            if epoch % self.params.save_every == 0:  # 保存
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))
        logging.info(f'Better models found w.r.t {self.critic[self.params.critic]}. Saved it!')
