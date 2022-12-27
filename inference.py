import torch
import numpy as np
import os
import config
from models.model import GeneratorModified, DiscriminatorModified, Generator_conv, Discriminator_conv
import utils
# %%
def load_weight(model,gen_weight,device,epoch,modelname='generator'):
    if not os.path.isfile(gen_weight):
        filename='%s_epoch-%s.pt'%(modelname,epoch)
        gen_weight=os.path.join(gen_weight,filename)
        

    if os.path.exists(gen_weight):
        model.load_state_dict(torch.load(gen_weight, map_location=device)) # model layer initialization
        print('generator weights are loaded from %s...'%gen_weight)
        return model
    else:
        
        print('generator is weight file not found...')
        return -1

def generate_noise(device,batch_size=1, noise_dim=200):
    noise = torch.rand(batch_size, noise_dim).to(device)
    return noise

    
trained_model_output='outputs_linear_gen1e-4_disc1e-3'
common_path='/chair/weights/'
filename=''
weight_path=trained_model_output+common_path+filename

epoch=970
device = torch.device(cuda_device if (torch.cuda.is_available()) else "cpu")
args = config.gen_parser().parse_args()

noise_dim = args.z_size
in_channels = args.in_channels
dim = args.cube_volume  # cube volume
generator = GeneratorModified(in_channels=in_channels, out_dim=dim, out_channels=1, noise_dim=noise_dim, bias=True,activation="sigmoid").to(device)
generator=load_weight(generator,gen_weight,device,epoch)
generator.eval()
noise=generate_noise(device,batch_size=20, noise_dim=noise_dim)

generator_pred=generator(noise)



# %% plot the result
if not os.path.isfile(weight_path):
    output_dir='%s/output_plot_epoch-%s/'%(trained_model_output, epoch)
else:
    _filename=weight_path.split('/')[-1]
    epoch=''.join(c for c in _filename if c.isdigit())
    output_dir='%s/output_plot_epoch-%s/'%(trained_model_output, epoch)

for i in range((generator_pred.shape[0])):
    object_3d=generator_pred[i]
    if object_3d.dim()==4:
        object_3d=object_3d.squeeze(0)
    savepaath=os.path.join(output_dir,'generated_3d_object_%s_for_epoch_%s.png'%(i+1,epoch))
    utils.plot_voxel(object_3d,savefig=True,figname=savepaath)
