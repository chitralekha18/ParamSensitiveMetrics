import sys
import collections as c
import os
import copy
import glob
import subprocess
import pdb
import torch
import numpy as np
import pickle
import soundfile as sf
import librosa
from torch.autograd import Variable
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings('ignore')

N_FREQ=257
N_FILTERS=512
IN_CHANNELS=N_FREQ

numStreams = 6
possible_kernels = [2,4,8,16,64,128,256,512,1024,2048]
hor_filters = [0]*numStreams
for j in range(numStreams):
    hor_filters[j]=possible_kernels[j]

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def log_scale(img):
    img = np.log1p(img)
    return img

def read_audio_spectrum(filename):
    x,fs = sf.read(filename)
    N_SAMPLES = len(x)
    N_FFT = 512
    K_HOP = 128
    R = np.abs(librosa.stft(x,n_fft=N_FFT, hop_length=K_HOP,win_length=N_FFT,center=False))
    return R,fs

def prepare_input(filename):
    R,fs = read_audio_spectrum(filename)
    a_style = log_scale(R)
    a_style = np.ascontiguousarray(a_style[None,None,:,:]) #[batch,channels,freq,samples]
    a_style = torch.from_numpy(a_style).permute(0,2,1,3) #pytorch:[batch,channels(freq),height(1),width(samples)]
    converted_img = Variable(a_style).type(dtype) #convert to pytorch variable
    return converted_img

def weights_init(m,hor_filter):
    std = np.sqrt(2)*np.sqrt(2.0/((N_FREQ+N_FILTERS)*hor_filter))
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1:
        m.weight.data.normal_(0.0,std)

class style_net(nn.Module):
    def __init__(self,hor_filter):
        super(style_net,self).__init__()
        self.layers = nn.Sequential(c.OrderedDict([
            ('conv1',nn.Conv2d(IN_CHANNELS,N_FILTERS,kernel_size=(1,hor_filter),bias=False)),
            ('relu1',nn.ReLU())
            ]))
    def forward(self, input):
        out = self.layers(input)
        return out

cnnlist = []
for j in range(numStreams):
    cnn = style_net(hor_filters[j])
    cnn.apply(lambda x, f=hor_filters[j]: weights_init(x,f))
    for param in cnn.parameters():
        param.requires_grad = False
    print(list(cnn.layers))
    if use_cuda:
        cnn = cnn.cuda()
    cnnlist.append(cnn)
content_layers_default = []
style_layers_default = ['relu_1']

class GramMatrix(nn.Module):
    def forward(self,input):
        a,b,c,d = input.size()
        features = input.view(b,a*c*d)
        features2 = features.unsqueeze(0)
        G = torch.matmul(features2,torch.transpose(features2,1,2))
        return G.div(a*c*d)

def get_gram(cnn,result,style_img, content_img=None,
                               style_weight=1, content_weight=0,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default, style_img2=None):
    cnn = copy.deepcopy(cnn)
    model = nn.Sequential()
    layer_list = list(cnn.layers)
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    i = 1
    for layer in layer_list:
        if isinstance(layer, nn.Conv2d): #if layer in vgg19 belong to class nn.Conv2d
            name = "conv_" + str(i)
            model.add_module(name, layer) #add that layer to our sequential model

            if name in style_layers: #at the right depth add the content loss "layer"
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                target_feature_gram = torch.flatten(target_feature_gram)
                target_feature_gram = target_feature_gram.numpy()
                target_feature_gram = target_feature_gram.reshape(512,512)
                target_feature_gram = StandardScaler().fit_transform(target_feature_gram)
                result.append((target_feature_gram))

        if isinstance(layer, nn.ReLU): #do the same for ReLUs
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in style_layers:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                target_feature_gram=torch.flatten(target_feature_gram)
                target_feature_gram=target_feature_gram.cpu().numpy()
                target_feature_gram=target_feature_gram.reshape(512,512)
                target_feature_gram= MinMaxScaler().fit_transform(target_feature_gram)
                result.append((target_feature_gram))

            i += 1
        if isinstance(layer, nn.MaxPool2d): #do the same for maxpool
            name = "pool_" + str(i)
            model.add_module(name, layer)
            
            if name in style_layers:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                target_feature_gram=torch.flatten(target_feature_gram)
                target_feature_gram=target_feature_gram.numpy()
                target_feature_gram=target_feature_gram.reshape(512,512)
                #print(target_feature_gram)
                target_feature_gram= StandardScaler().fit_transform(target_feature_gram)
                result.append((target_feature_gram))


    return result


def getGM(filename):
    input_img = prepare_input(filename)
    result = []
    for j in range(numStreams):
        gram = get_gram(cnnlist[j],result,input_img,None,1,0)
    return result

def compute_stats_GM(files):
    dim = 128 
    mu = np.zeros((dim, 1), dtype=np.float64)
    sigma = np.zeros((dim,dim),dtype=np.float64)
    samp_count = 0
    emb_array = []
    #print(len(files))
    for filename in files:
        #print(filename)
        GM_tensors = getGM(filename)
        #GM_tensors = pickle.load(open(filename,'rb')) #6x512x512
        cnn_num = 0
        for GM in GM_tensors:
            GM = np.squeeze(np.array(GM))
            for row in range(0,np.shape(GM)[0],dim):
                for col in range(0,np.shape(GM)[1],dim):
                    #print(row,col)
                    GM_sub = GM[row:row+dim,col:col+dim]
                    emb = np.array(GM_sub.mean(axis=0)).reshape((dim,1))
                    mu += emb
                    sigma +=emb*emb.T
                    emb_array+=list(emb)
            cnn_num += 1
        samp_count += 1
    

    mean = mu/samp_count
    std = (sigma/(samp_count-1)) - ((mu*mu.T*samp_count)/(samp_count-1))

    return np.squeeze(mean),std

def frechet_distance(mu_test, sigma_test, mu_train, sigma_train):
    """Fréchet distance calculation.

    From: D.C. Dowson & B.V. Landau The Fréchet distance between
    multivariate normal distributions
    https://doi.org/10.1016/0047-259X(82)90077-X

    The Fréchet distance between two multivariate gaussians,
    `X ~ N(mu_x, sigma_x)` and `Y ~ N(mu_y, sigma_y)`, is `d^2`.

    d^2 = (mu_x - mu_y)^2 + Tr(sigma_x + sigma_y - 2 * sqrt(sigma_x*sigma_y))
      = (mu_x - mu_y)^2 + Tr(sigma_x) + Tr(sigma_y)
                        - 2 * Tr(sqrt(sigma_x*sigma_y)))

    Args:
    mu_test: Mean of the test multivariate gaussian.
    sigma_test: Covariance matrix of the test multivariate gaussians.
    mu_train: Mean of the test multivariate gaussian.
    sigma_train: Covariance matrix of the test multivariate gaussians.

    Returns:
    The Fréchet distance.

    Raises:
    ValueError: If the input arrays do not have the expect shapes.
    """
    if len(mu_train.shape) != 1:
        raise ValueError('mu_train must be 1 dimensional.')
    if len(sigma_train.shape) != 2:
        raise ValueError('sigma_train must be 2 dimensional.')

    if mu_test.shape != mu_train.shape:
        raise ValueError('mu_test should have the same shape as mu_train')
    if sigma_test.shape != sigma_train.shape:
        raise ValueError('sigma_test should have the same shape as sigma_train')

    mu_diff = mu_test - mu_train
    #trace_sqrt_product = _stable_trace_sqrt_product(sigma_test, sigma_train) # --> Not being used for AGM computation

    return mu_diff.dot(mu_diff) #+ np.trace(sigma_test) + np.trace(sigma_train) - 2 * trace_sqrt_product # --> Not being used for AGM computation

def compute_FAD_GM(mu_bg,mu_test,sigma_bg=[],sigma_test=[]):
    if np.isnan(sigma_bg).any():
        sigma_bg = np.zeros(np.shape(sigma_bg))  # .tolist()
    if np.isnan(sigma_test).any():
        sigma_test = np.zeros(np.shape(sigma_test))  # .tolist()
    fad = frechet_distance(mu_bg, sigma_bg, mu_test, sigma_test)
    # print("FAD: %f" % fad)

    return fad

if __name__=='__main__':
    folder = sys.argv[1]
    filelist = glob.glob(folder+os.sep+'*.wav')
    filelist.sort()
    anchor = [filelist[0]]
    #create anchor GM vector
    mu_bg,sigma_bg = compute_stats_GM(anchor)
        
    testfiles = filelist[1:]
    agm_overall = np.zeros(len(testfiles))
    
    cnt=0
    for test in testfiles:
        #create test GM vector
        mu_test,sigma_test = compute_stats_GM([test])
        # compute FAD
        agm = compute_FAD_GM(mu_bg,mu_test,sigma_bg,sigma_test)
        print(agm)
        agm_overall[cnt] +=agm
        cnt+=1
        
    plt.figure()
    plt.plot(agm_overall,'o-')
    plt.xlabel('parameter variation')
    plt.ylabel('AGM')
    plt.title('Anchor to test comparison, where the test parameter varies over x-axis')
    plt.show()


