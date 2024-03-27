import torch
import torch.nn.functional as F
import torch.optim as optim
import toolkit.model as model
from torch.autograd import Variable
class CLASSIFIER:
    # train_Y is interger
    def __init__(self, log_p0_Y, prototype_layer_sizes, _train_X, _train_Y, att, _cuda, _lr=0.0001,
                 _beta1=0.5, _nepoch=20,
                 _batch_size=100, generalized=True, tem=0.04, pretrain_model=None):

        self.train_X = _train_X
        self.train_Y = _train_Y
        self.att = att
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model = model.netP(prototype_layer_sizes, self.att.size(-1))
        self.Linear_Classifier = model.LINEAR_LOGSOFTMAX(self.input_dim, self.att.size(0))
        self.optimizerL = optim.Adam(self.Linear_Classifier.parameters(), lr=_lr, betas=(_beta1, 0.999))
        self.optimizerP = optim.Adam(self.model.parameters(), _lr, betas=(_beta1, 0.999),weight_decay=0.0001)
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)
        self.lr = _lr
        self.beta1 = _beta1
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizerP, gamma=0.5, step_size=30)
        self.tem = tem
        self.log_p0_Y=log_p0_Y

        if pretrain_model is not None:
            self.model.load_state_dict(pretrain_model.state_dict())

        if self.cuda:
            self.model = self.model.cuda()
            self.Linear_Classifier = self.Linear_Classifier.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
            self.att = self.att.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        if generalized:
            self.fit()


    def fit(self):

        for epoch in range(self.nepoch):
            self.model.train()
            # self.Linear_Classifier.train()
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                # self.Linear_Classifier.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                batch_test = F.normalize(batch_input, dim=-1).cuda()
                batch_label = batch_label.cuda()
                proto_a=F.normalize(self.model(self.att),dim=-1)
                logits = batch_test@proto_a.t()/self.tem
                # logits = self.Linear_Classifier(batch_input.cuda())
                """
                Logit Adjustment
                """
                logits = logits+self.log_p0_Y

                loss = F.cross_entropy(logits,batch_label)
                loss.backward()
                self.optimizerP.step()
                # self.optimizerL.step()
            self.scheduler.step()
            self.model.eval()
            # self.Linear_Classifier.eval()


    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            # print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]
