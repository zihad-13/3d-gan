# %% Importing
import torch
import numpy as np
from torch.autograd.variable import Variable
import os
import utils
from models.model import GeneratorModified, DiscriminatorModified, Generator_conv, Discriminator_conv
from torch.utils.tensorboard import SummaryWriter
import config
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from msdlib.msd.processing import get_time_estimation
import time
import sys
import json
# %% get arguments
args = config.gen_parser().parse_args()

# %% variables
cuda_device = "cuda:0"
inputDir='data/voxel'
object_name='chair'
gen_weight=''
disc_weight=''
rootdir=os.path.join(inputDir,object_name)
save_interval = 1 
show_sample_data =True
# %% set output dir
output_dir = 'outputs_%s_genLr_%s_discLr_%s_act_%s/'%(args.last_layer_name,args.gen_lr,args.disc_lr,args.last_layer_activation)+object_name #'outputs_linear_gen1e-4_disc1e-3/'+object_name
weight_dir=os.path.join(output_dir, 'weights')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(weight_dir, exist_ok=True)
# %% prepare datapath

voxel_folder_names = os.listdir(rootdir)
voxelLoc='models'
voxel_paths=[]
for name in voxel_folder_names:
    try:
        voxel_names=os.listdir(os.path.join(rootdir, name,voxelLoc))
        voxel_name=voxel_names[0]
    except: continue
    
    voxel_paths.append(os.path.join(rootdir, name,voxelLoc,voxel_name))
# %% voxel plotting
single_voxel=utils.data_loader(voxel_paths[0])
single_voxel=single_voxel.squeeze(0)


if show_sample_data:
    utils.plot_voxel(single_voxel,savefig=True,figname='image.png')
#sys.exit()

# %% model init function
def model_init(m,bias=args.bias):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.in_channels
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        if bias: m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        n = m.num_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        if bias: m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        if bias: m.bias.data.fill_(0)
        

def discriminator_init(m,bias=args.bias):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.in_channels
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        if bias: m.bias.data.uniform_(-y, y)
        #m.weight_u.data.uniform_(-y, y)
        #m.weight_v.data.uniform_(-y, y)
        #m.weight_bar.data.uniform_(-y, y)
    elif classname.find('BatchNorm') != -1:
        n = m.num_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        if bias: m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        if bias: m.bias.data.fill_(0)

# %%
def get_model_info(model, get='mean'):
    weights = {n: p for n, p in model.named_parameters()}
    total_param = 0
    for n in weights:
        _mult = 1
        for x in weights[n].shape:
            _mult *= x
        total_param += _mult
    
    grads = [weights[n].grad.mean() if get == 'mean' else weights[n].grad.sum() for n in weights if weights[n].grad is not None]
    nan_count = len(weights) - len(grads)
    grad, weight = None, None
    if get == 'sum':
        grad = sum(grads) if len(grads) > 0 else None
        weight = torch.sum(torch.tensor([weights[n].data.sum() for n in weights]))
    elif get == 'mean':
        grad = sum(grads) / len(grads) if len(grads) > 0 else None
        weight = torch.mean(torch.tensor([weights[n].data.sum() for n in weights]))
    
    return total_param, grad, weight, nan_count

# %% save parametes
def save_params(args,other_params,output_dir):
        with open(os.path.join(output_dir, 'arguments.json'), 'w') as f:

            json.dump({**vars(args), **other_params},f, sort_keys=True, indent=4)
# %% output plot function
def plotNsave(errG_arr, errD_arr, epoch_arr, epoch, noise,  generator, discriminator, bnum=None, save_weights=False):
    params = {'Generator loss': errG_arr,
             'Discriminator loss': errD_arr}
    bnum = '' if bnum is None else '-batch-%d'%bnum
    
    # Plotting loss curves
    fig, ax = plt.subplots(ncols=len(params), figsize=(30, 5))
    for i, name in enumerate(params):
        l, = ax[i].plot(epoch_arr, params[name])
        ax[i].set_xlabel('epoch')
        ax[i].set_ylabel('Loss')
        ax[i].legend([l], [name])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss-curve_epoch-%d'%epoch + bnum + '.png'), bbox_inches='tight')
    plt.close()
    _img=generator(noise)
    generated_img = _img.round()
    utils.plot_voxel(generated_img,savefig=True,figname=os.path.join(output_dir, 'generated_voxel_epoch-%d'%epoch + bnum + '.png'))
    # storing sample images
    #cv2.imwrite(os.path.join(output_dir, 'rd_xd_epoch-%d'%epoch + bnum + '.png'), rd_xd[0].detach().cpu().squeeze().numpy() * 255)
    #cv2.imwrite(os.path.join(output_dir, 'rphi_xd_epoch-%d'%epoch + bnum + '.png'), rphi_xd[0].detach().cpu().squeeze().numpy() * 255)
    
    if save_weights:
        # storing models
        torch.save(generator.state_dict(), os.path.join(weight_dir, 'generator_epoch-%d'%epoch + bnum + '.pt'))
        torch.save(discriminator.state_dict(), os.path.join(weight_dir, 'discriminator_epoch-%d'%epoch + bnum + '.pt'))

# %% Dataset class

class Dataset():
    
    def __init__(self, voxel_paths, device, dtype=torch.float32):
        self.dtype = dtype
        self.device = device
        self.voxel_paths=voxel_paths          
            
        #self.voxel_paths = [os.listdir(os.path.join(rootdir, name,voxelLoc))[0] for name in voxel_folder_names]
        
        #self.label_paths = [os.path.join(labeldir, name.replace('.binvox', '.png')) for name in voxel_names]
        self._len = len(self.voxel_paths)
        
    def __getitem__(self, index):
        voxel = utils.data_loader(self.voxel_paths[index]).to(dtype=self.dtype, device=self.device)
        #label = torch.tensor(cv2.imread(self.label_paths[index], cv2.IMREAD_UNCHANGED) / 255).to(dtype=self.dtype, device=self.device)
        label=torch.tensor([1]).to(dtype=self.dtype, device=self.device)
        return voxel, label.unsqueeze(0)
        
    def __len__(self):
        return self._len
    
# %%

device = torch.device(cuda_device if (torch.cuda.is_available()) else "cpu")
dataset = Dataset(voxel_paths, device)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
total_batch = len(loader)

# %%

noise_dim = args.z_size
in_channels = args.in_channels
dim = args.cube_volume  # cube volume
last_layer_activation=args.last_layer_activation

if args.last_layer_name.lower()=='conv':
    generator = Generator_conv(noise_dim, bias=args.bias,  activation=last_layer_activation).to(device)
    discriminator = Discriminator_conv(in_channels=1, dim=dim, out_conv_channels=in_channels,bias=args.bias).to(device)
elif args.last_layer_name.lower()=='linear' or args.last_layer_name.lower()!='conv':
    generator = GeneratorModified(in_channels=in_channels, out_dim=dim, out_channels=1, noise_dim=noise_dim, bias=args.bias,activation=last_layer_activation).to(device)
    discriminator = DiscriminatorModified(in_channels=1, dim=dim, out_conv_channels=in_channels,bias=args.bias).to(device)


# model layer initialization
if os.path.exists(gen_weight):
    generator.load_state_dict(torch.load(gen_weight, map_location=device))
    print('generator weights are loaded from %s...'%gen_weight)
else:
    generator.apply(model_init)
    print('generator is randomly initialized...')
if os.path.exists(disc_weight):
    discriminator.load_state_dict(torch.load(disc_weight, map_location=device))
    print('discriminator weights are loaded from %s...'%disc_weight)
else:
    discriminator.apply(discriminator_init)
    print('discriminator is randomly initialized...')

# %%
generator.train()
discriminator.train()
opt_disc = optim.Adam(discriminator.parameters(), lr=args.disc_lr)
opt_gen = optim.Adam(generator.parameters(), lr=args.gen_lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0
accuracy = 0
accuracy_thres = 0.8
t=time.time()
epoch_arr = []
errD_arr = []
errG_arr = []
other_params={'accuracy_threshold':accuracy_thres,'save_interval':save_interval,'voxel_dir':rootdir, 'discriminator_weight_dir':weight_dir,
'generator_weight_dir':weight_dir,'output_dir':output_dir}
save_params(args,other_params,output_dir)
#import sys; sys.exit()
torch.autograd.set_detect_anomaly(True)
for epoch in range(args.num_epochs):
    gl, dl = 0, 0
    for batch_idx, (real, _) in enumerate(loader):
        batch_size = real.shape[0]
        #print('batch_size ', batch_size)
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.rand(batch_size, noise_dim).to(device)
        #x = torch.zeros(batch_size, noise_dim).to(device)
        #noise = x + (0.1**0.5)*torch.randn(batch_size, noise_dim).to(device)
        #noise = torch.ones(args.batch_size, args.z_size, device=device).normal_(args.noise_mean, args.noise_std)
        #discriminator.train()
        disc_real = discriminator(real).view(-1) # D(x)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        # opt_disc.zero_grad()
        # lossD_real.backward()
        #generator.eval()
        #with torch.no_grad():
        fake = generator(noise) ### generate fake image
        
        disc_fake = discriminator(fake.detach()).view(-1) # D(G(z))
        
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake)
        if accuracy <= accuracy_thres:
            opt_disc.zero_grad()
            lossD.backward()
            #lossD_fake.backward()
            #print('grad and weigth for gen #1 ' ,get_model_info(generator))
            #print('grad and weigth disc #2 ' ,get_model_info(discriminator))
            opt_disc.step()
            
        disc_pred=torch.cat([disc_real,disc_fake])
        disc_label=torch.cat([torch.ones_like(disc_real),torch.zeros_like(disc_fake)])
        accuracy = (disc_pred.round() == disc_label).sum() / disc_label.shape[0]
        #print(disc_pred.shape,disc_label)
        #print('discriminator accuracy',accuracy)


        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))

        generator.train()
        #fake = generator(noise)  # G(z)
        #discriminator.eval()
        
        output = discriminator(fake).view(-1) # (D(G(z)))
        lossG = criterion(output, torch.ones_like(output))
        #generator.zero_grad()
        #print('grad and weigth for gen #3 ' ,get_model_info(generator))
        #print('grad and weigth disc #4 ' ,get_model_info(discriminator))
        opt_gen.zero_grad()
        #print('grad and weigth for gen #5 ' ,get_model_info(generator))
        #print('grad and weigth disc #6 ' ,get_model_info(discriminator))
        lossG.backward()
        #print('grad and weigth for gen #7 ' ,get_model_info(generator))
        #print('grad and weigth disc #8 ' ,get_model_info(discriminator))
        opt_gen.step()
        #print('grad and weigth for gen #9 ' ,get_model_info(generator))
        #print('grad and weigth disc #10 ' ,get_model_info(discriminator))
        
        #input()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{args.num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )
        
        gl += lossG.item()
        dl += lossD.item()
        tstring = get_time_estimation(t, current_ep=epoch, current_batch=batch_idx, total_ep=args.num_epochs, total_batch=total_batch)
        instring = f"[{epoch+1:04d}/{args.num_epochs:04d}][{batch_idx+1:04d}/{total_batch:04d}]  {tstring}"
        print(f"\r{instring}    Loss_D: {dl/(batch_idx+1):05.4f}   Loss_G: {gl/(batch_idx+1):05.4f}      ", end='', flush=True)

    epoch_arr.append(epoch)
    errD_arr.append(dl / total_batch)
    errG_arr.append(gl / total_batch)

    if epoch % save_interval == 0 or epoch + 1 == args.num_epochs:
        save_weights = True if epoch % 10 == 0 or epoch + 1 == args.num_epochs else False
        #test_noise = torch.ones(1, args.z_size, device=device).normal_(args.noise_mean, args.noise_std)
        test_noise=noise
        plotNsave(errG_arr, errD_arr, epoch_arr, epoch, test_noise, generator, discriminator, bnum=None, save_weights=save_weights)
