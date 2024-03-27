import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

class MLP_G(nn.Module):
    def __init__(self, opt: object) -> object:
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class Encoder(nn.Module):

    def __init__(self, opt):

        super(Encoder,self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        self.fc1=nn.Linear(layer_sizes[0]+opt.latent_size, layer_sizes[-1])
        self.fc3=nn.Linear(layer_sizes[-1], opt.latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(opt.latent_size*2, opt.latent_size)
        self.linear_log_var = nn.Linear(opt.latent_size*2, opt.latent_size)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None: x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

class Generator(nn.Module):

    def __init__(self, opt):

        super(Generator,self).__init__()

        layer_sizes = opt.decoder_layer_sizes
        latent_size=opt.latent_size
        input_size = latent_size * 2
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid=nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h

class Encoder_Proto(nn.Module):
    def __init__(self, opt):
        super(Encoder_Proto, self).__init__()
        self.hidden = None
        self.lantent = None
        self.protoSize = opt.protoSize
        self.resSize = opt.resSize
        self.attSize = opt.attSize
        self.fc1 = nn.Linear(self.resSize, opt.hSize)
        self.fc_p = nn.Linear(opt.hSize, self.protoSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)
    def forward(self, feat):
        x = feat
        self.hidden = self.lrelu(self.fc1(x))
        normalized_h = F.normalize(self.hidden, dim=1)
        proto = self.fc_p(self.hidden)
        # recons = self.fc5(proto)
        return proto, normalized_h
    def getLayersOutDet(self):
        #used at synthesis time and feature transformation
        return self.hidden.detach()

class netP(nn.Module):
    def __init__(self, layer_sizes,att_size):
        super().__init__()
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate( zip([att_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes) :
                self.MLP.add_module(name="A%i"%(i), module=nn.LeakyReLU(0.2, True))
        self.apply(weights_init)
    def forward(self, x, c=None):
        x = self.MLP(x)
        return x

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

class FR(nn.Module):
    def __init__(self, opt):
        super(FR, self).__init__()
        self.embedSz = 0
        self.hidden = None

        self.attSize = opt.attSize
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.attSize * 2)
        self.discriminator = nn.Linear(opt.attSize, 1)
        self.classifier = nn.Linear(opt.attSize, opt.seen_class_num)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, feat, train_G=False):
        h = feat
        # if self.embedSz > 0:
        #   assert att is not None, 'Conditional Decoder requires attribute input'
        #    h = torch.cat((feat,att),1)
        self.hidden = self.lrelu(self.fc1(h))
        self.lantent = self.fc3(self.hidden)
        mus, stds = self.lantent[:, :self.attSize], self.lantent[:, self.attSize:]
        stds = self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)
        h = encoder_out
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred = self.logic(self.classifier(mus))
        if self.sigmoid is not None:
            h = self.sigmoid(h)
        else:
            h = h / h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h.size(1))
        return mus, stds, dis_out, pred, encoder_out, h

    def getLayersOutDet(self):
        # used at synthesis time and feature transformation
        return self.hidden.detach()

def reparameter(mu, sigma):
    return (torch.randn_like(mu) * sigma) + mu