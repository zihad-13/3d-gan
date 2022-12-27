import argparse



def gen_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data")
    parser.add_argument('--dimension', type=int, default=64, help='dimension of Generator output 3D shape object')
    parser.add_argument('--viewpoint_dim', type=bool, default=128, help='dimension of Discriminator input image')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--gen_lr', type=float, default=1e-4) # default from paper =0.0025
    parser.add_argument('--disc_lr', type=float, default=1e-3) #  default from paper 1e-5
    parser.add_argument('--lr_reduce', type=float, default=.998)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--dom_lambda', type=float, default=100)
    parser.add_argument('--z_size', type=float, default=200)
    parser.add_argument('--noise_mean', type=float, default=0)
    parser.add_argument('--noise_std', type=float, default=1)
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--dropout_rate', type=float, default=0.25)
    parser.add_argument('--grad_clip', type=float, default=1)
    parser.add_argument('--is_grayscale', type=bool, default=True)
    parser.add_argument('--models', type=int, default=5)
    parser.add_argument('--in_channels', type=int, default=512)
    parser.add_argument('--cube_volume', type=int, default=64)
    parser.add_argument('--last_layer_name', type=str, default='conv')
    parser.add_argument('--last_layer_activation', type=str, default='sigmoid')
    return parser
