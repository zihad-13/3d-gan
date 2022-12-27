import numpy as np
import torch
import binvox_rw
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def show_3d(path, title='figure'):
    if isinstance(path, str):
        data = data_loader(path)
    else:
        data = path
    if isinstance(data, torch.Tensor):
        data = data.squeeze().detach().cpu().numpy()
    index = np.where(data)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title)
    ax.scatter(index[0], index[1], index[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot_vector(points, origin=None, title='Vector plot', margin=.05, show_coord=False):
    
    if origin is None:
        origin = np.zeros_like(points)
    
    fig = plt.figure()
    if points.shape[1] == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    ax.quiver(*(origin.T.tolist()), *(points.T.tolist()))
    
    if show_coord:
        for i in range(points.shape[0]):
            ax.text(points[i, 0], points[i, 1], points[i, 2], str(points[i]))
    
    _min = np.concatenate([origin, points], axis=0).min(axis=0)
    _max = np.concatenate([origin, points], axis=0).max(axis=0)
    _range = _max - _min
    _range[_range==0] = 1
    _ext = _range * margin
    _min = _min - _ext
    _max = _max + _ext
    
    ax.set_xlim(_min[0], _max[0])
    ax.set_ylim(_min[1], _max[0])
    if points.shape[1] == 3:
        ax.set_zlim(_min[2], _max[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def rotate_object(voxel, angles, rescale=False, rescale_margin=.1, wdim=3):
    # preparing rotation matrix
    rotmat = get_rotation_matrix(angles)
    
    original_dim = voxel.shape[0]
    
    if rescale:
        # adding margin to the voxel data
        voxel = add_margin(voxel, 1)
    
    # getting positions of the occupied space
    voxelpos = np.array(np.where(voxel==1)).T
    # shifting the object to the origin
    voxel_dim = voxel.shape[0]
    mid = np.array([[voxel_dim // 2, voxel_dim // 2, voxel_dim // 2]])
    voxelpos -= mid

    # rotating the object
    voxelpos_new = np.dot(rotmat, voxelpos.T).T

    if rescale:
        # rescaling the object
        _min, _max = voxelpos_new.min(axis=0, keepdims=True), voxelpos_new.max(axis=0, keepdims=True)
        _range = _max - _min
        _dev = _range * rescale_margin
        _min -= _dev
        _max += _dev
        voxelpos_new = (voxelpos_new - _min) / (_max - _min) * original_dim
    else:
        voxelpos_new += mid

    # creating new voxel with rotated object shape
    voxel_new = position2voxel(voxelpos_new, original_dim)

    # smooting the voxel grid
    # weight = torch.ones((1, 1, wdim, wdim, wdim), dtype=torch.float32) / (wdim ** 3)
    voxel_new = torch.nn.functional.max_pool3d(voxel_new.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                               kernel_size=wdim, padding=wdim//2, stride=1)
    voxel_new = voxel_new.round().to(dtype=torch.int).squeeze()

    return voxel_new


def position2voxel(voxel_pos, dim):
    voxel_pos = voxel_pos.round().astype(int)
    voxel = torch.zeros((dim, dim, dim))
    for v in voxel_pos:
        if all((v >= 0) & (v < dim)):
            voxel[v[0], v[1], v[2]] = 1
    return voxel


def add_margin(voxel, factor):
    big_box = torch.zeros((int(64 * (1 + factor)), int(64 * (1 + factor)), int(64 * (1 + factor))))
    mid = big_box.shape[0] // 2
    st = int(mid - 64 // 2)
    ed = st + 64
    big_box[st: ed, st: ed, st: ed] = voxel
    return big_box


def get_rotation_matrix(angles):

    angle_x, angle_y, angle_z = angles
    theta_x = np.deg2rad(angle_x)
    theta_y = np.deg2rad(angle_y)
    theta_z = np.deg2rad(angle_z)

    rotmat_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]])
    rotmat_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rotmat_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]])
    rotmat = np.dot(np.dot(rotmat_x, rotmat_y), rotmat_z)

    return rotmat


def data_loader(file_path):
    with open(file_path, 'rb') as f:
        voxels = binvox_rw.read_as_3d_array(f)
    return torch.tensor(voxels.data.round().astype(int)).unsqueeze(0)

def plot_voxel(voxelarray,savefig=False,figname=''):
    if voxelarray.dim()==5:
        voxelarray=voxelarray[0][0]
    elif voxelarray.dim()==4:
        voxelarray=voxelarray[0]
    
        
    voxelarray=voxelarray.detach().cpu().numpy()
    voxelarray=np.swapaxes(voxelarray,1,2)
    print("voxel shape ", voxelarray.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')

    ax.voxels(voxelarray, edgecolor="k")
    
    plt.show()
    if savefig:
        fig.savefig(figname, dpi=90)
    plt.close(fig)