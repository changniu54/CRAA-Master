from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import toolkit.util as util
import sys
import toolkit.model as model
import numpy as np
from toolkit.center_loss import TripCenterLoss_margin
from sklearn.cluster import KMeans
import toolkit.classifier_adj as classifier_adj
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='../Data_PS', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr_b', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--lr_i', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--task_mode', type=str, default='gzsl', help='the task mode: gzsl, zsl or il')
# parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="gpu_id", dest='gpu_id')

## incremental learning
parser.add_argument('--nepoch_base', type=int, default=2000, help='number of epochs to train for the base model')
parser.add_argument('--nepoch_incremental', type=int, default=2000, help='number of epochs to train for the incremental model')
parser.add_argument('--base_class_num', type=int, default=20, help='the number of base classes for the first training')
parser.add_argument('--seen_class_num', type=int, default=150, help='the number of all the seen classes')
parser.add_argument('--task_num', type=int, default=5, help='the total number of tasks')
parser.add_argument('--log_dir', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--kd_weight', type=float, default=1, help='the weight of knowledge loss')
parser.add_argument('--syn_replay_num', type=int, default=50, help='number features to replay for previous classes in training stage')
parser.add_argument('--syn_previous_seen_num', type=int, default=200, help='number features to generate for previous classes in test stage')
parser.add_argument('--syn_unseen_num', type=int, default=300, help='number features to generate for unseen classes in test stage')


## netDFE
parser.add_argument('--epr_lr', type=float, default=0.0001, help='learning rate to train encoder_pr ')
parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')
parser.add_argument('--center_margin', type=float, default=150, help='the margin in the center loss')
parser.add_argument('--incenter_weight', type=float, default=0.5, help='the weight for the center loss')
parser.add_argument('--protoSize', type=int, default=2048, help='size of prototype features')
parser.add_argument('--refine_fea', action='store_true', default=False, help='use the refined feature in test stage')
parser.add_argument('--hSize', type=int, default=4096, help='size of the hidden units in EncoderPR')

## Logit Adjustment
parser.add_argument('--gamma', type=float, default=30, help='the ratio for seen and unseen prior')
parser.add_argument('--epsilon', type=float, default=2, help='the ratio for previous seen and current seen prior')
parser.add_argument('--proto_layer_sizes', type=list, default=[1024,2048], help='size of the hidden and output units in prototype learner')
parser.add_argument('--tem', type=float, default=0.04,help='temprature (Eq. 16)')


opt = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# print(opt)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
# print("# of training samples: ", data.ntrain)

save_path = os.path.join(opt.log_dir,'craa',opt.dataset)

if os.path.exists(save_path) == False:
    os.makedirs(save_path)
txt_name = 'result_ours_'+ opt.dataset + '.txt'
result = open(os.path.join(save_path, txt_name), 'w')

def compute_refine_fea(test_X, new_size, batch_size, netDFE):
    start = 0
    ntest = test_X.size()[0]
    new_test_X = torch.zeros(ntest, new_size)
    for i in range(0, ntest, opt.batch_size):
        end = min(ntest, start+batch_size)
        if opt.cuda:
            with torch.no_grad():
                inputX = Variable(test_X[start:end].cuda())
        else:
            with torch.no_grad():
                inputX = Variable(test_X[start:end])
        # proto_x, _, feat2, _ = netDFE(inputX)
        proto_x, _ = netDFE(inputX)
        feat1 = netDFE.getLayersOutDet()
        # new_test_X[start:end] = inputX.data.cpu()  ## x
        # new_test_X[start:end] = proto_x.data.cpu()  ## proto_x
        new_test_X[start:end] = torch.cat([inputX, feat1], dim=1).data.cpu()  ## [x,h]
        # new_test_X[start:end] = feat1.data.cpu()  ## h
        start = end
    return new_test_X

def next_batch(batch_size, index):
    ntrain = train_feature.size(0)
    start = index
    if start==0:
        perm = torch.randperm(ntrain)
        train_feature.copy_(train_feature[perm])
        train_label.copy_(train_label[perm])
    if start+batch_size > ntrain:
        rest_num_smp = ntrain-start
        if rest_num_smp > 0:
            X_rest_part = train_feature[start:ntrain]
            Y_rest_part = train_label[start:ntrain]
        perm = torch.randperm(ntrain)
        train_feature.copy_(train_feature[perm])
        train_label.copy_(train_label[perm])
        start = 0
        index = batch_size - rest_num_smp
        end = index
        X_new_part = train_feature[start:end]
        Y_new_part = train_label[start:end]
        if rest_num_smp>0:
            batch_feature = torch.cat([X_rest_part, X_new_part], 0)
            batch_label = torch.cat([Y_rest_part, Y_new_part], 0)
            batch_att = data.attribute[batch_label]
        else:
            batch_feature = X_new_part
            batch_label = Y_new_part
            batch_att = data.attribute[batch_label]
    else:
        index += batch_size
        end = index
        batch_feature = train_feature[start:end]
        batch_label = train_label[start:end]
        batch_att = data.attribute[batch_label]
    return index,batch_feature,batch_att,util.map_label(batch_label, train_label_set)

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(Variable(syn_noise), Variable(syn_att))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)
    return syn_feature, syn_label

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def train_base_model(opt):

    center_criterion = TripCenterLoss_margin(num_classes=train_label_set.size(0), feat_dim=opt.protoSize,
                                             use_gpu=opt.cuda, pre_center=None)
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_b, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_b, betas=(opt.beta1, 0.999))
    optimizerEPR = optim.Adam(netDFE.parameters(), lr=opt.epr_lr, betas=(opt.beta1, 0.999))
    optimizer_center = optim.Adam(center_criterion.parameters(), lr=opt.lr_b, betas=(opt.beta1, 0.999))
    input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
    input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
    noise = torch.FloatTensor(opt.batch_size, opt.nz)
    one = torch.FloatTensor([1])
    mone = one * -1
    input_label = torch.LongTensor(opt.batch_size)
    if opt.cuda:
        input_res = input_res.cuda()
        input_att = input_att.cuda()
        noise = noise.cuda()
        one = one.cuda()
        mone = mone.cuda()
        center_criterion.cuda()
        input_label = input_label.cuda()
        netG.cuda()
        netD.cuda()
        netDFE.cuda()
        center.cuda()
        netCI.cuda()

    for epoch in range(opt.nepoch_base):
        index_in_epoch = 0
        netG.train()
        netDFE.train()
        for i in range(0, train_feature.size(0), opt.batch_size):
            # (1) Update D network
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set as False below in netG update
            for p in netDFE.parameters():
                p.requires_grad = True

            for iter_d in range(opt.critic_iter):
                # for iter_pr in range(3):
                index_in_epoch, batch_fea, batch_att, batch_label = next_batch(opt.batch_size, index_in_epoch)
                input_res.copy_(batch_fea)
                input_att.copy_(batch_att)
                input_label.copy_(batch_label)
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)
                noise.normal_(0, 1)
                noisev = Variable(noise)
                fake = netG(noisev, input_attv)
                ## update netDFE
                netDFE.zero_grad()
                protoR, _ = netDFE(input_resv)
                center_loss_real = center_criterion(protoR, input_label, margin=opt.center_margin, incenter_weight=opt.incenter_weight)
                PR_Loss = center_loss_real
                PR_Loss.backward()
                optimizerEPR.step()
                optimizer_center.step()

                ## update netD
                netD.zero_grad()
                criticD_real = netD(input_resv, input_attv)
                criticD_real = criticD_real.mean()
                criticD_real.backward(mone.mean())
                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(one.mean())
                # gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
                gradient_penalty.backward()
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()
            # (2) Update G network
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation
            if  opt.freeze_dec:
                for p in netDFE.parameters():  # freeze decoder
                    p.requires_grad = False
            netG.zero_grad()
            input_attv = Variable(input_att)
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            criticG_fake = netD(fake, input_attv)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake
            errG = G_cost

            errG.backward()
            optimizerG.step()
        # print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f, Loss_PR:%.4f'
        #       % (epoch + 1, opt.nepoch_base, D_cost.item(), G_cost.item(), Wasserstein_D.item(), c_errG.item(), PR_Loss.item()))

        ### testing stage
        # if epoch+1 >= max(opt.nepoch_base-20,0):
        # if epoch+1==opt.nepoch_base or (epoch+1)%5==0:
        #     test(opt, task_id, epoch, netG, netDFE)

    # test(opt, task_id, epoch, netG, netDFE)
    torch.save(netG.state_dict(), os.path.join(save_path, 'task_' + str(task_id) + '_generator' + '.t7'))
    torch.save(netD.state_dict(), os.path.join(save_path, 'task_' + str(task_id) + '_discriminator' + '.t7'))
    torch.save(netDFE.state_dict(), os.path.join(save_path, 'task_' + str(task_id) + '_encoder_pr' + '.t7'))
    torch.save(center, os.path.join(save_path, 'task_' + str(task_id) + '_center' + '.t7'))
    torch.save(netCI.state_dict(), os.path.join(save_path, 'task_' + str(task_id) + '_classifier_netp' + '.t7'))

    # result.writelines("Seen Weighted Accuracy of Model %d is  %.2f\n" % (task_id, 100 * best_s))
    # result.writelines("Unseen Accuracy of Model %d is         %.2f\n" % (task_id, 100 * best_u))
    # result.writelines("Harmonic Accuracy of Model %d is       %.2f\n" % (task_id, 100 * best_h))

    return netG, netDFE

def train_incremental_model(opt):

    center_criterion = TripCenterLoss_margin(num_classes=train_label_set.size(0), feat_dim=opt.protoSize,
                                             use_gpu=opt.cuda, pre_center=None)
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_i, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_i, betas=(opt.beta1, 0.999))
    optimizerEPR = optim.Adam(netDFE.parameters(), lr=opt.epr_lr, betas=(opt.beta1, 0.999))
    optimizer_center = optim.Adam(center_criterion.parameters(), lr=opt.lr_i, betas=(opt.beta1, 0.999))

    input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
    input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
    noise = torch.FloatTensor(opt.batch_size, opt.nz)
    one = torch.FloatTensor([1])
    mone = one * -1
    input_label = torch.LongTensor(opt.batch_size)

    if opt.cuda:
        input_res = input_res.cuda()
        noise, input_att = noise.cuda(), input_att.cuda()
        one = one.cuda()
        mone = mone.cuda()
        input_label = input_label.cuda()
        netG.cuda()
        netD.cuda()
        netDFE.cuda()
        center.cuda()
        netCI.cuda()

    for epoch in range(opt.nepoch_incremental):
        index_in_epoch = 0
        netG.train()
        netDFE.train()
        for i in range(0, train_feature.size(0), opt.batch_size):
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set as False below in netG update
            for p in netDFE.parameters():
                p.requires_grad = True
            for iter_d in range(opt.critic_iter):
                # for iter_pr in range(3):
                index_in_epoch, batch_fea, batch_att, batch_label = next_batch(opt.batch_size, index_in_epoch)
                input_res.copy_(batch_fea)
                input_att.copy_(batch_att)
                input_label.copy_(batch_label)

                netD.zero_grad()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)
                noise.normal_(0, 1)
                noisev = Variable(noise)
                fake = netG(noisev, input_attv)

                ## update netDFE
                netDFE.zero_grad()
                protoR, _ = netDFE(input_resv)
                center_loss_real = center_criterion(protoR, input_label, margin=opt.center_margin, incenter_weight=opt.incenter_weight)
                PR_Loss = center_loss_real
                PR_Loss.backward()
                optimizerEPR.step()
                optimizer_center.step()

                ## update netD
                criticD_real = netD(input_resv, input_attv)
                criticD_real = criticD_real.mean()
                criticD_real.backward(mone.mean())

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(one.mean())
                # gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
                gradient_penalty.backward()
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()
            # (2) Update G network
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation
            if opt.freeze_dec:
                for p in netDFE.parameters():  # freeze decoder
                    p.requires_grad = False
            netG.zero_grad()
            input_attv = Variable(input_att)
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            criticG_fake = netD(fake, input_attv)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake
            errG = G_cost

            errG.backward()
            optimizerG.step()
        # print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f, KD_G: %.4f, KD_D: %.4f'
        #           % (epoch + 1, opt.nepoch_incremental, D_cost.item(), G_cost.item(), Wasserstein_D.item(), c_errG.item(), kd_g_loss.item(), kd_d_loss.item()))
        # print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f, Loss_PR:%.4f, KD_G: %.4f'
        #           % (epoch + 1, opt.nepoch_incremental, D_cost.item(), G_cost.item(), Wasserstein_D.item(), c_errG.item(), PR_Loss.item(), kd_g_loss.item()))

        # if epoch+1 >= max(opt.nepoch_incremental-20,0):
        # if (epoch+1)%5==0 or epoch+1==opt.nepoch_incremental:
        #     test(opt, task_id, epoch)
    # test(opt, task_id, epoch, netG, netDFE)

    torch.save(netG.state_dict(), os.path.join(save_path, 'task_' + str(task_id) + '_generator' + '.t7'))
    torch.save(netD.state_dict(), os.path.join(save_path, 'task_' + str(task_id) + '_discriminator' + '.t7'))
    torch.save(netDFE.state_dict(), os.path.join(save_path, 'task_' + str(task_id) + '_encoder_pr' + '.t7'))
    torch.save(center, os.path.join(save_path, 'task_' + str(task_id) + '_center' + '.t7'))
    torch.save(netCI.state_dict(),os.path.join(save_path, 'task_' + str(task_id) + '_classifier_netp' + '.t7'))
    # torch.save(best_center[-nclass_per_task:], os.path.join(save_path, 'task_' + str(task_id) + '_center' + '.t7'))


    # result.writelines("Seen Weighted Accuracy of Model %d is  %.2f\n" % (task_id, 100 * best_s))
    # result.writelines("Unseen Accuracy of Model %d is         %.2f\n" % (task_id, 100 * best_u))
    # result.writelines("Harmonic Accuracy of Model %d is       %.2f\n" % (task_id, 100 * best_h))
    return netG, netDFE

def test(opt, tid, epoch, netG, netDFE):
    netG.eval()
    netDFE.eval()
    ## train the classifier on all class data
    unseen_syn_feature, unseen_syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute,
                                                                opt.syn_unseen_num)
    if opt.task_mode == 'gzsl':
        if tid==0:
            train_X = torch.cat([train_feature, unseen_syn_feature], 0)
            train_Y = torch.cat([train_label, unseen_syn_label], 0)
        else:
            ## compensate the previous classes by generating previous samples additionally
            pre_seen_fea, pre_seen_label = generate_syn_feature(netG, previous_seen_label_set,
                                                                data.attribute, opt.syn_previous_seen_num)
            train_X = torch.cat([pre_seen_fea, current_train_feature, unseen_syn_feature], 0)
            train_Y = torch.cat([pre_seen_label, current_train_label, unseen_syn_label], 0)

        train_X = compute_refine_fea(train_X, refine_dim, 128, netDFE)
        gzsl_classifier = classifier_adj.CLASSIFIER(log_p0_Y[test_label_set],
                                                    opt.proto_layer_sizes, train_X,
                                                    util.map_label(train_Y, test_label_set),
                                                    data.attribute[test_label_set], opt.cuda, 0.0001, 0.5, 60,
                                                    512, True, opt.tem)

        torch.save(gzsl_classifier.model.state_dict(),
                   os.path.join(save_path, 'task_' + str(tid) + '_classifier_netp' + '.t7'))
        for p in gzsl_classifier.model.parameters():  # set requires_grad to False
            p.requires_grad = False
        # refine the test feature
        test_seen_feature_r = compute_refine_fea(test_seen_feature, refine_dim, 128, netDFE)
        test_unseen_feature_r = compute_refine_fea(data.test_unseen_feature, refine_dim, 128, netDFE)
        acc_unseen = compute_one_task_acc(gzsl_classifier.model, test_unseen_feature_r,
                                          util.map_label(data.test_unseen_label, test_label_set),
                                          util.map_label(data.unseenclasses, test_label_set))
        acc_seen = compute_seen_acc(tid, random_perm, gzsl_classifier.model, test_seen_feature_r,
                                    test_seen_label, test_label_set)
        acc_h = 2 * acc_unseen * acc_seen / (acc_unseen + acc_seen)

        print('S ACC of Model %d at Epoch %d:   %.2f' % (tid, epoch, 100 * acc_seen))
        print('U ACC of Model %d at Epoch %d:   %.2f' % (tid, epoch, 100 * acc_unseen))
        print('H ACC of Model %d at Epoch %d:   %.2f' % (tid, epoch, 100 * acc_h))

    elif opt.task_mode == 'zsl':

        ## train the classifier(embedding att-->latent) on unseen data
        train_X = unseen_syn_feature
        train_Y = unseen_syn_label
        train_X = compute_refine_fea(train_X, refine_dim, 128, netDFE)
        zsl_classifier = classifier_adj.CLASSIFIER(log_p0_Y[data.unseenclasses],
                                                   opt.proto_layer_sizes, train_X,
                                                   util.map_label(train_Y, data.unseenclasses),
                                                   data.attribute[data.unseenclasses], opt.cuda, 0.0001, 0.5, 60,
                                                   512, True, opt.tem)
        torch.save(zsl_classifier.model.state_dict(),
                   os.path.join(save_path, 'task_' + str(tid) + '_classifier_netp' + '.t7'))
        for p in zsl_classifier.model.parameters():  # set requires_grad to False
            p.requires_grad = False
        test_unseen_feature = compute_refine_fea(data.test_unseen_feature, refine_dim, 128, netDFE)
        acc_zsl = compute_one_task_acc(zsl_classifier.model, test_unseen_feature,
                                       util.map_label(data.test_unseen_label, test_label_set),
                                       util.map_label(data.unseenclasses, test_label_set))
        print('Unseen ACC of Model %d at Epoch %d:   %.2f' % (tid, epoch, 100 * acc_zsl))

    elif opt.task_mode == 'il':  ## test on ever seen label set
        if tid==0:
            train_X = train_feature
            train_Y = train_label
        else:
            ## compensate the previous classes by generating previous samples additionally
            pre_seen_fea, pre_seen_label = generate_syn_feature(netG, previous_seen_label_set,
                                                                data.attribute, opt.syn_previous_seen_num)
            train_X = torch.cat([pre_seen_fea, current_train_feature], 0)
            train_Y = torch.cat([pre_seen_label, current_train_label], 0)

        train_X = compute_refine_fea(train_X, refine_dim, 128, netDFE)
        il_classifier = classifier_adj.CLASSIFIER(log_p0_Y[test_label_set],
                                                  opt.proto_layer_sizes, train_X,
                                                  util.map_label(train_Y, test_label_set),
                                                  data.attribute[test_label_set], opt.cuda, 0.0001, 0.5,
                                                  60, 512, True, opt.tem)

        torch.save(il_classifier.model.state_dict(),
                   os.path.join(save_path, 'task_' + str(tid) + '_classifier_netp' + '.t7'))
        for p in il_classifier.model.parameters():  # set requires_grad to False
            p.requires_grad = False
        test_seen_feature_r = compute_refine_fea(test_seen_feature, refine_dim, 128, netDFE)
        acc_il = compute_seen_acc(tid, random_perm, il_classifier.model, test_seen_feature_r,
                                  test_seen_label, test_label_set)
        # _,_ = compute_seen_forget(tid, random_perm, il_classifier.model, test_seen_feature_r,
        #                           data.test_seen_label, test_label_set)
        print('Seen ACC of Model %d at Epoch %d:   %.2f' % (tid, epoch, 100 * acc_il))

    else:
        assert 0, 'Unavailable task mode, please set it as "gzsl", "zsl" or "il"! '

def compute_per_class_acc(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        acc_per_class += torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx)
    acc_per_class /= target_classes.size(0)
    return acc_per_class

def compute_one_task_acc(netP, x_unseen, y_unseen, y_set):
    netP.eval()
    start = 0
    ntest = y_unseen.size(0)
    predicted_label = torch.LongTensor(y_unseen.size())
    batch_size = 64
    att = data.attribute[test_label_set].cuda()
    for i in range(0, ntest, batch_size):
        end = min(ntest, start+batch_size)
        if opt.cuda:
            with torch.no_grad():
                test_batch = F.normalize(x_unseen[start:end], dim=-1).cuda()
                proto = F.normalize(netP(att), dim=-1).cuda()
                output = test_batch@proto.t()
        else:
            with torch.no_grad():
                test_batch = F.normalize(x_unseen[start:end], dim=-1)
                proto = F.normalize(netP(att), dim=-1)
                output = test_batch@proto.t()
        _, predicted_label[start:end] = torch.max(output.data,1)
        start = end
    acc = compute_per_class_acc(y_unseen, predicted_label, y_set)
    return acc

def compute_seen_acc(task_id, random_perm, classifier, test_feature, test_label, y_set):
    acc_ave = 0.0
    forget_rate = 0.0
    for itask in range(task_id+1): ## the total number of tasks currently
        if itask == 0:
            idx_test = random_perm[:opt.base_class_num]
        else:
            idx_test = random_perm[opt.base_class_num + (itask - 1) * nclass_per_task: opt.base_class_num + itask * nclass_per_task]
        itask_label_set = data.seenclasses[idx_test]
        itask_smp_idx = []
        for i in range(itask_label_set.size(0)):
            icls = itask_label_set[i]
            itask_smp_idx.append(torch.where(test_label==icls)[0])
        itask_smp_idx = torch.cat(itask_smp_idx,0)
        itask_smp_idx = itask_smp_idx[torch.randperm(itask_smp_idx.size(0))]
        itask_test_feature = test_feature[itask_smp_idx]
        itask_test_label   = test_label[itask_smp_idx]
        itask_test_label   = util.map_label(itask_test_label, y_set)
        itask_label_set    = util.map_label(itask_label_set, y_set)
        itask_acc = compute_one_task_acc(classifier, itask_test_feature, itask_test_label, itask_label_set)

        if opt.task_mode=='il':
            print("Accuracy of Model %d on Task %d is %.2f" % (task_id, itask, 100 * itask_acc))
            result.writelines("Accuracy of Model %d on Task %d is %.2f" % (task_id, itask, 100 * itask_acc))
            if task_id<opt.task_num-1 and itask_acc>pre_acc[itask]:
                pre_acc[itask] = itask_acc
            if task_id == opt.task_num-1 and itask!=opt.task_num-1:
                forget_rate += (pre_acc[itask]-itask_acc)

        if itask == 0:
            acc_ave += itask_acc * (float(opt.base_class_num) /
                              (opt.base_class_num + task_id * nclass_per_task))
        else:
            acc_ave += itask_acc * (float(nclass_per_task) /
                              (opt.base_class_num + task_id * nclass_per_task))

    if opt.task_mode == 'il' and task_id==opt.task_num-1:
        forget_rate = forget_rate/(opt.task_num-1)
        print("Forget Rate is %.2f" % (forget_rate*100))
    return acc_ave

def compute_one_task_forget(netP, x_unseen, y_unseen, y_set):
    netP.eval()
    start = 0
    ntest = y_unseen.size(0)
    predicted_label = torch.LongTensor(y_unseen.size())
    batch_size = 64
    att = data.attribute[y_set].cuda()
    for i in range(0, ntest, batch_size):
        end = min(ntest, start+batch_size)
        if opt.cuda:
            with torch.no_grad():
                test_batch = F.normalize(x_unseen[start:end], dim=-1).cuda()
                proto = F.normalize(netP(att), dim=-1).cuda()
                output = test_batch@proto.t()
        else:
            with torch.no_grad():
                test_batch = F.normalize(x_unseen[start:end], dim=-1)
                proto = F.normalize(netP(att), dim=-1)
                output = test_batch@proto.t()
        _, predicted_label[start:end] = torch.max(output.data,1)
        start = end
    acc = compute_per_class_acc(y_unseen, predicted_label, util.map_label(y_set,y_set))
    return acc

def compute_seen_forget(task_id, random_perm, classifier, test_feature, test_label, y_set):
    ## know task-ID at test stage
    acc_ave = 0.0
    forget_rate = 0.0
    for itask in range(task_id+1): ## the total number of tasks currently
        if itask == 0:
            idx_test = random_perm[:opt.base_class_num]
        else:
            idx_test = random_perm[opt.base_class_num + (itask - 1) * nclass_per_task: opt.base_class_num + itask * nclass_per_task]
        itask_label_set = data.seenclasses[idx_test]
        itask_smp_idx = []
        for i in range(itask_label_set.size(0)):
            icls = itask_label_set[i]
            itask_smp_idx.append(torch.where(test_label==icls)[0])
        itask_smp_idx = torch.cat(itask_smp_idx,0)
        itask_smp_idx = itask_smp_idx[torch.randperm(itask_smp_idx.size(0))]
        itask_test_feature = test_feature[itask_smp_idx]
        itask_test_label   = test_label[itask_smp_idx]
        itask_test_label   = util.map_label(itask_test_label, itask_label_set)

        itask_acc = compute_one_task_forget(classifier, itask_test_feature, itask_test_label, itask_label_set)


        print("Intra-task ACC of Model %d on Task %d is %.2f" % (task_id, itask, 100 * itask_acc))
        # result.writelines("Intra-task ACC of Model %d on Task %d is %.2f" % (task_id, itask, 100 * itask_acc))
        if itask<opt.task_num-1 and itask_acc>pre_acc[itask]:
            pre_acc[itask] = itask_acc
        if itask<opt.task_num-1 and itask<task_id :
            forget_rate += (pre_acc[itask]-itask_acc)

        if itask == 0:
            acc_ave += itask_acc * (float(opt.base_class_num) /
                              (opt.base_class_num + task_id * nclass_per_task))
        else:
            acc_ave += itask_acc * (float(nclass_per_task) /
                              (opt.base_class_num + task_id * nclass_per_task))
    if task_id!=0:
        forget_rate = forget_rate/task_id
        print("Forget Rate is %.2f" % (forget_rate*100))

    return acc_ave,forget_rate

nclass_per_task = int((opt.seen_class_num - opt.base_class_num) / (opt.task_num-1))  ### all seen classes as the training classes

#random_perm = np.random.permutation(data.seen_class_num)
random_perm = np.arange(opt.seen_class_num)
pre_acc = torch.zeros(opt.task_num-1)
for task_id in range(opt.task_num):
    print('--------------------------------------------')
    print('---> Task Mode: %s    Current Task ID: %d' % (opt.task_mode, task_id))
    ## prepare the training data
    index = random_perm[:opt.base_class_num + task_id * nclass_per_task]
    ever_seen_label_set = data.seenclasses[index] ## all the ever seen classes (including previous and current seen classes)
    previous_seen_label_set = torch.LongTensor(0)
    if opt.task_mode=='gzsl':
        test_label_set = torch.cat([ever_seen_label_set,data.unseenclasses],0)
    elif opt.task_mode=='zsl':
        test_label_set = data.unseenclasses
    elif opt.task_mode=='il':
        test_label_set = ever_seen_label_set
    else:
        assert 0, 'Unavailable task mode, please set it as "gzsl", "zsl" or "il"! '

    ## re-define training set
    if task_id == 0:
        current_seen_label_set = ever_seen_label_set
        index_train = random_perm[:opt.base_class_num]
        train_label_set = data.seenclasses[index_train]
        train_smp_idx = []
        for i in index_train:
            icls = data.seenclasses[i]
            train_smp_idx.append(torch.where(data.train_label == icls)[0])

        train_smp_idx = torch.cat(train_smp_idx, 0)
        train_smp_idx = train_smp_idx[torch.randperm(train_smp_idx.size(0))]
        train_label = data.train_label[train_smp_idx]
        train_feature = data.train_feature[train_smp_idx]
    else:
        index_previous = random_perm[:opt.base_class_num+(task_id-1)*nclass_per_task]
        previous_seen_label_set = data.seenclasses[index_previous]  ## the label set of previous seen (except the current task)
        index_train = random_perm[opt.base_class_num + (task_id-1) * nclass_per_task : opt.base_class_num + task_id * nclass_per_task]
        current_seen_label_set = data.seenclasses[index_train]
        train_smp_idx = []
        for i in index_train:
            icls = data.seenclasses[i]
            train_smp_idx.append(torch.where(data.train_label == icls)[0])
        train_smp_idx = torch.cat(train_smp_idx, 0)
        current_train_label = data.train_label[train_smp_idx]
        current_train_feature = data.train_feature[train_smp_idx]


        ### task-agnostic generative replay
        netG_prev = model.MLP_G(opt)
        netG_prev.load_state_dict(torch.load(os.path.join(save_path, 'task_' + str(task_id-1) + '_generator' + '.t7')))
        netG_prev.eval()
        if opt.cuda:
            netG_prev.cuda()

        # generate the previous fake samples
        ever_seen_syn_feature, ever_seen_syn_label = generate_syn_feature(netG_prev, previous_seen_label_set, data.attribute, opt.syn_replay_num)
        # the train_feature for the current task include the previous fake samples and the current real samples
        train_feature = torch.cat([current_train_feature, ever_seen_syn_feature], 0)
        train_label = torch.cat([current_train_label, ever_seen_syn_label], 0)
        train_label_set = ever_seen_label_set

    ## re-define test seen set
    test_smp_idx = []
    for i in range(ever_seen_label_set.size(0)):
        icls = ever_seen_label_set[i]
        test_smp_idx.append(torch.where(data.test_seen_label == icls)[0])
    test_smp_idx = torch.cat(test_smp_idx, 0)
    test_seen_feature = data.test_seen_feature[test_smp_idx]
    test_seen_label = data.test_seen_label[test_smp_idx]


    ## Logit Adjustment
    num_s = data.seenclasses.size(0)
    num_u = data.unseenclasses.size(0)
    num_class = num_s + num_u
    ## seen-unseen prior
    log_p0_Y = torch.zeros(num_class).cuda()
    # log_p0_Y[data.seenclasses] = math.log(1-1/opt.gamma)
    log_p0_Y[current_seen_label_set] = math.log(1-1/opt.gamma)
    log_p0_Y[previous_seen_label_set] = math.log((1-1/opt.gamma)/opt.epsilon)
    log_p0_Y[data.unseenclasses] = math.log(1/opt.gamma)
    # for i in range(task_id):
    #     j = (i+1)/(task_id+1)
    #     if i == 0:
    #         index = random_perm[:opt.base_class_num]
    #     else:
    #         index = random_perm[opt.base_class_num+(i-1)*nclass_per_task: opt.base_class_num+i*nclass_per_task]
    #     class_set = data.seenclasses[index]
    #     log_p0_Y[class_set] += math.log(j)


    # initialize current generator and discriminator
    netG = model.MLP_G(opt)
    netD = model.MLP_CRITIC(opt)
    netDFE = model.Encoder_Proto(opt)
    if opt.refine_fea:
        refine_dim = opt.hSize + opt.resSize
        opt.proto_layer_sizes[-1] = refine_dim
    netCI = model.netP(opt.proto_layer_sizes, opt.attSize)

    if task_id == 0:
        center = nn.Parameter(torch.randn(opt.base_class_num, opt.protoSize))
        netG, netDFE = train_base_model(opt)
        test(opt, task_id, opt.nepoch_base, netG, netDFE)
    else:
        netG.load_state_dict(torch.load(os.path.join(save_path, 'task_' + str(task_id-1) + '_generator' + '.t7')))
        netD.load_state_dict(torch.load(os.path.join(save_path, 'task_' + str(task_id-1) + '_discriminator' + '.t7')))
        netDFE.load_state_dict(torch.load(os.path.join(save_path, 'task_' + str(task_id-1) + '_encoder_pr' + '.t7')))
        center_pre = torch.load(os.path.join(save_path, 'task_' + str(task_id - 1) + '_center' + '.t7')).cuda()
        netCI.load_state_dict(torch.load(os.path.join(save_path, 'task_' + str(task_id - 1) + '_classifier_netp' + '.t7')))

        # center_pre = []
        # for i in range(task_id):
        #     center_pre.append(torch.load(os.path.join(save_path, 'task_' + str(i) + '_center' + '.t7')).cuda())
        # center_pre = torch.cat(center_pre,0).cuda()
        center_current = torch.randn(nclass_per_task, opt.protoSize).cuda()

        center = nn.Parameter(torch.cat([center_pre, center_current], 0))
        netG, netDFE = train_incremental_model(opt)
        test(opt, task_id, opt.nepoch_incremental, netG, netDFE)






