import collections
import numpy as np
import scipy
from scipy.spatial import KDTree
import copy
import igl
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
import json
from json import JSONEncoder
from geomloss import SamplesLoss
import trimesh
import torch.nn.functional as F
np.seterr(divide='ignore', invalid='ignore')

import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import IterableDataset
from torch.func import functional_call
#from torch.nn.utils.stateless import functional_call 
import math


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

from building_sdf.viz_utils import *
from building_sdf.data_prep import *
from building_sdf.learning.network_viz import *

# From https://github.com/pytorch/pytorch/issues/104564
@torch.compile
def cosine_similarity(t1, t2, dim=-1, eps=1e-8):
    # get normalization value
    t1_div = torch.linalg.vector_norm(t1, dim=dim, keepdims=True)
    t2_div = torch.linalg.vector_norm(t2, dim=dim, keepdims=True)

    t2_div = t2_div.clone()
    with torch.no_grad():
        t1_div.clamp_(math.sqrt(eps))
        t2_div.clamp_(math.sqrt(eps))

    # normalize, avoiding division by 0
    t1_norm = t1 / t1_div
    t2_norm = t2 / t2_div

    return (t1_norm * t2_norm).sum(dim=dim)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
#from building_sdf.learning.global_cond import *

## Activation functions and initializations

# From https://github.com/vsitzmann/siren/blob/master/modules.py
########################
# Initialization methods

def init_weights_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
            
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)
    
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

# # From https://gist.github.com/rtqichen/b22a9c6bfc4f36e605a7b3ac1ab4122f
class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)
    
    
# Based on the implementation in https://github.com/ml-for-gp/jaxgptoolbox/blob/main/demos/lipschitz_mlp/model.py
class LipWeightNorm(nn.Module):
    """
    Lipschitz weight normalization based on the L-infinity norm (see Eq.9 in [Liu et al 2022])
    """

    def __init__(self, module):
        super(LipWeightNorm, self).__init__()
        self.module = module
        self.w = module.weight
        self.softplus = nn.Softplus()

    def _setweights(self):
        w = self.w
        absrowsum = torch.sum(torch.abs(w), axis = 1)
        c = torch.max(absrowsum)
        softplus_c = self.softplus(c)
        scale = torch.minimum(torch.tensor(1.0), softplus_c/absrowsum)
        w = w * torch.unsqueeze(scale, 1)
        setattr(self.module, 'weight', torch.nn.Parameter(w))

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)
    
# # Move to gpu
# def to_gpu(ob):
#     if isinstance(ob, collections.abc.Mapping):
#         return {k: to_gpu(v) for k, v in ob.items()}
#     elif isinstance(ob, tuple):
#         return tuple(to_gpu(k) for k in ob)
#     elif isinstance(ob, list):
#         return [to_gpu(k) for k in ob]
#     else:
#         try:
#             return ob.cuda()
#         except:
#             return ob


class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        #print('initial_weights',self.weight.data)
        #print('initial_bias',self.bias.data)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max() # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        #print('cur_weights',self.weight.data)
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        # print('c',self.c)
        # print('lipc',lipc)
        # print('abs_wgt',torch.abs(self.weight).sum(1))
        # print('scale',scale)
        #scale = torch.clamp(scale, max=1.0)
        scale = torch.minimum(scale, torch.tensor(1.0))
        # print('scale_afterCl',scale)
        # print('weight',self.weight)
        # print('weight_scaled',self.weight * scale.unsqueeze(1))
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)

        
# Move to gpu
def to_gpu(ob):
    if isinstance(ob, collections.abc.Mapping):
        return {k: v.cuda() if((not isinstance(v, list)) and (not isinstance(v[0], str))) else v for k, v in ob.items() }
    elif isinstance(ob, tuple):
        return tuple(k.cuda() for k in ob)
    elif isinstance(ob, list):
        return [k.cuda() for k in ob]
    else:
        try:
            return ob.cuda()
        except:
            return ob
        
# Print parameters
def print_params(module):
    for name, param in module.named_parameters():
        print(f"{name}: {tuple(param.shape)}")

# Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
# Print summary of training results
def trainSummary(step, loss, lr_cur, model_output, gt, models, loss_type_names = ['sdf', 'perf'], lr_type_names = ['model', 'latent'], rendering_dir = None, resolution = None, XYZ_coord = None, index = 0, act_index = 0, viz_dir = None, plotting_function = None, buffered_bbox = None, typeLoss='', det_losses = None, det_losses_types = None, det_losses_lambdas = None, perf_first_element=True):
    model_output = model_output[0:-1] if(len(models)==1) else model_output
    if(not isinstance(model_output, tuple)):
        model_output = [model_output]

    print(f"Step {step}: ")
    loss_string = f"{typeLoss} loss = {float(loss):.5f}, "
    for i in range(len(model_output)):
        if(perf_first_element):
            loss_string += f"{loss_type_names[i]} range = {float(np.min(model_output[i][..., :1].detach().cpu().numpy())):.5f} and {float(np.max(model_output[i][..., :1].detach().cpu().numpy())):.5f}, "
            loss_string += f"{loss_type_names[i]} GT range = {float(np.min(gt[i][..., :1].detach().cpu().numpy())):.5f} and {float(np.max(gt[i][..., :1].detach().cpu().numpy())):.5f}, "
        else:
            loss_string += f"{loss_type_names[i]} range = {float(np.min(model_output[i].detach().cpu().numpy())):.5f} and {float(np.max(model_output[i].detach().cpu().numpy())):.5f}, "
            loss_string += f"{loss_type_names[i]} GT range = {float(np.min(gt[i].detach().cpu().numpy())):.5f} and {float(np.max(gt[i].detach().cpu().numpy())):.5f}, "
    for i in range(len(lr_cur)):
        loss_string += f"{lr_type_names[i]} lr rate = {float(lr_cur[i]):.8f}." if(i == len(lr_cur)-1) else f"{lr_type_names[i]} lr rate = {float(lr_cur[i]):.8f}, "
    print(loss_string)
        
    if(det_losses is not None):
        losses_str = 'absolute: '
        losses_str_ld = 'weighted: '
        for i in range(len(det_losses)):
            losses_str += f"{det_losses_types[i]} = {float(det_losses[i]):.7f}"
            losses_str += ", " if(i < len(det_losses)-1) else "."
            losses_str_ld += f"{det_losses_types[i]} = {float(det_losses[i]*det_losses_lambdas[i]):.7f}"
            losses_str_ld += ", " if(i < len(det_losses)-1) else "."
        print(f"" + losses_str)
        print(f"" + losses_str_ld)

    if plotting_function is not None: 
         plotting_function(resolution, XYZ_coord, models, index, act_index, viz_dir, step, buffered_bbox, rendering_dir, typeP = typeLoss)
            
# Save and load model
def saveLoadModel(model, losses, save_toggle, load_toggle, experiments_dir, experiment_name):
    
    if not os.path.exists(experiments_dir + '/' + experiment_name):
        os.makedirs(experiments_dir + '/' + experiment_name)

    if(save_toggle):
        # Save model
        saved_ml = torch.save(model.state_dict(), experiments_dir + '/' + experiment_name + '/' + 'model.pth')
        # Save losses
        np.save(experiments_dir + '/' + experiment_name + '/' + 'losses.npy', losses)    

    if(load_toggle):
        # Load model
        model.load_state_dict(torch.load(experiments_dir + '/' + experiment_name + '/' + 'model.pth'))
        # Load losses
        losses = np.load(experiments_dir + '/' + experiment_name + '/' + 'losses.npy')

    return model, losses

def loadModelParam(model_dir, param_ar = ['mlp_param', 'loss_param', 'rep_param', 'dataset_train_param', 'dataset_test_param']):

    parameters_dir = model_dir + '/parameters/'
    param_sets = []
    # If parameter file exists, load it.
    for i in range(len(param_ar)):
        if(os.path.exists(parameters_dir + '/' + param_ar[i] + '.json')):
            param_sets.append(json.loads(open(parameters_dir + '/' + param_ar[i] + '.json').read()))
        else:
            param_sets.append(None)
    return param_sets

def getModelParametersCount(model):
    return sum(p.numel() for p in model.parameters())

def createModelDirs(model_dir):

    # Create model directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create checkpoints directory
    checkpoints_dir = model_dir + '/' + 'checkpoints'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Create training progress visualization directory
    training_viz_dir = model_dir + '/' + 'training_viz'
    if not os.path.exists(training_viz_dir):
        os.makedirs(training_viz_dir)

    # Get rendering directory
    rendering_dir = '/'.join(model_dir.replace('Experiments','Renders').split('/')[0:-2])
    
    return model_dir, checkpoints_dir, training_viz_dir, rendering_dir

def loadLosses(model_dir, load_type = 'final', epoch = None, model_type = 'train'):
    
    parameters_dir = model_dir + '/' + 'parameters'
    model_dir = model_dir +'/' + model_type if('test' in model_type) else model_dir
    checkpoints_dir = model_dir + '/' + 'checkpoints'

    # Read parameter files
    if(os.path.exists(parameters_dir + '/' + 'repParam.json')):
        total_ep = json.loads(open(parameters_dir + '/' + 'repParam.json').read())['total_epochs']

    # Load networks from saved model state dictionary
    if(load_type == 'current'):
        cur_ep = np.loadtxt(os.path.join(model_dir, 'epoch_current.txt'))
        losses = np.loadtxt(os.path.join(model_dir, 'train_losses_current.txt'))
        #losses = np.split(losses, cur_ep)
    elif(load_type == 'final'):
        losses = np.loadtxt(os.path.join(model_dir, 'train_losses_final.txt'))
        #losses = np.split(losses, total_ep)
    else:
        losses = np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch))
        #losses = np.split(losses, epoch_to_view)

    return losses

def loadDetailedLosses(model_dir, load_type = 'final', epoch = None, loss_type = 'train', model_type = 'train'):
    
    model_dir = model_dir +'/' + model_type if('test' in model_type) else model_dir
    checkpoints_dir = model_dir + '/' + 'checkpoints'

    # Load networks from saved model state dictionary
    if(load_type == 'current'):
        d_losses = json.loads(open(model_dir + '/' + loss_type + '_losses_detailed_current.json').read())
    elif(load_type == 'final'):
        d_losses = json.loads(open(model_dir + '/' + loss_type + '_losses_detailed_final.json').read())
    else:
        d_losses = json.loads(open(checkpoints_dir + '/' + loss_type + '_losses_detailed_epoch_%04d.json' % epoch).read())

    return d_losses


def modelfromLatents(model, param_to_replace): 

    # Create a duplicate of the model
    with torch.no_grad():
        # Create a duplicate copy of the model
        try:
            model_latent = copy.deepcopy(model)
        except:
            if(isinstance(model, nn.DataParallel)):
                weights = {}
                for module in model.modules():
                    for _, hook in module._forward_pre_hooks.items():
                        if isinstance(hook, WeightNorm):
                            weights[hook.name] = getattr(self, hook.name)
                            delattr(module, hook.name)
                model_latent = copy.deepcopy(model)
                for module in model_latent.modules():
                    for _, hook in module._forward_pre_hooks.items():
                        if isinstance(hook, WeightNorm):
                            hook(module, None)
                for name, value in weights.items():
                    setattr(model_latent, name, value)
            else:
                weights = {}
                for module in model.modules():
                    for _, hook in module._forward_pre_hooks.items():
                        if isinstance(hook, WeightNorm):
                            weights[hook.name] = getattr(self, hook.name)
                            delattr(module, hook.name)
                model_latent = copy.deepcopy(model)
                for module in model_latent.modules():
                    for _, hook in module._forward_pre_hooks.items():
                        if isinstance(hook, WeightNorm):
                            hook(module, None)
                for name, value in weights.items():
                    setattr(model_latent, name, value)

    # Create a latent embedding with the number of desired interpolation latents
    latents = nn.Embedding(num_embeddings = param_to_replace.shape[0], embedding_dim = param_to_replace.shape[1])
    latents.weight = torch.nn.parameter.Parameter(param_to_replace)
    
    # If model is wrapped with DataParallel - only keep the model modules
    if(isinstance(model_latent, nn.DataParallel)):
        model_latent.module.latents = latents
    else:
        model_latent.latents = latents
        
    # Return model with replaced parameters
    return model_latent


       
def loadCombinedMetrics(metrics_dir, load_type, model_type, epoch, mc_resolution, alt_resolution=None, sdf_only=False):
    #metrics_dir = metrics_dir if(model_type=='train') else metrics_dir.replace('metrics','test/metrics')
    if(os.path.exists(metrics_dir + '/' + 'metrics_'+ str(mc_resolution) + '_combined_' + load_type + '_' + str(epoch) + '.json')):
        comb_metrics = json.loads(open(metrics_dir + '/' + 'metrics_'+ str(mc_resolution) + '_combined_' + load_type + '_' + str(epoch) + '.json').read()) if(not sdf_only) else json.loads(open(metrics_dir + '/' + 'metrics_'+ str(mc_resolution) + '_combined_' + load_type + '_' + str(epoch) + '_sdflossOnly.json').read())
    elif((alt_resolution is not None) and os.path.exists(metrics_dir + '/' + 'metrics_'+ str(alt_resolution) + '_combined_' + load_type + '_' + str(epoch) + '.json')):
        comb_metrics = json.loads(open(metrics_dir + '/' + 'metrics_'+ str(alt_resolution) + '_combined_' + load_type + '_' + str(epoch) + '.json').read()) if(not sdf_only) else json.loads(open(metrics_dir + '/' + 'metrics_'+ str(alt_resolution) + '_combined_' + load_type + '_' + str(epoch) + '_sdflossOnly.json').read())
    else:
        print('The metrics file does not exist: ' + metrics_dir)
        comb_metrics = None
    return comb_metrics

def extractMetricsPerGeo(index, models, model_dir, xyz_sdf_dataset, dataset_param, load_type, model_type, epoch, resolution = 200, num_mesh_samples_chamfer = 30000, num_mesh_samples_wasserstein = 500, num_mesh_samples_mesh = 1000, num_mesh_samples_cossim = 2500, mesh_cp_param = 0.01, sdf_only=False):    
    
    # Directories
    parameters_dir = model_dir + '/parameters'
    metrics_dir = model_dir + '/metrics' if(model_type != 'test') else model_dir + '/test/metrics'
    reconstruction_dir = model_dir + '/reconstructions' if(model_type != 'test') else model_dir + '/test/reconstructions'
    buffered_bbox = np.array(json.loads(open(parameters_dir + '/' + 'rep_param.json').read())['buffered_bbox'])
    #print('dataset_param',dataset_param)
    
    # Get the data for the specific geometry index    
    model_input, ground_truth_sdf, ground_truth_normals, full_name = xyz_sdf_dataset.__getitem__(index, override_rot=None, returnNone=True)
    if(ground_truth_sdf is not None):
        g_name = '_'.join(full_name.split('_')[0:2])
        rot = int(full_name.split('_')[2])
        real_idx = int(g_name.split('_')[-1]) if(g_name.split('_')[0] in ['building','extrBuilding']) else int(g_name.split('_')[0])
        print('full_name',full_name)
        print('g_name',g_name)
        if(device.type != 'cpu'):
            model_input, ground_truth_sdf, ground_truth_normals = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_normals)

        # Predict the sdf using the trained model
        if(len(models)==1):
            model_output = models[0].to(device)({'xyz': model_input['xyz'].unsqueeze(0),'idx': torch.tensor([index]).to(device)})
        else:
            grid_output, param_latent = models[0].to(device)({'xyz': model_input['xyz'].unsqueeze(0),'idx': torch.tensor([index]).to(device)})
            model_output = models[1].to(device)({'xyz': model_input['xyz'].unsqueeze(0),'idx': torch.tensor([index]).to(device)}, grid_output)

        # Isolating the sdf results
        model_output = model_output[0:-1] if(len(models)==1) else model_output
        if(not isinstance(model_output, tuple)):
            model_output = [model_output]
        predicted_sdf = model_output[0]

        # Calculate sdf losses
        sdf_loss_l1 = (torch.abs(predicted_sdf - ground_truth_sdf)).mean().detach().numpy()
        #print('sdf_loss_l1',sdf_loss_l1)
        sdf_loss_l2 = ((predicted_sdf - ground_truth_sdf)**2).mean().detach().numpy()
        #print('sdf_loss_l2',sdf_loss_l2)

        # Get geometry metrics: chamfer distances, earth mover's distance
        rc_path = reconstruction_dir + '/' + 'reconstruction_' + str(resolution) + '_index_' + str(index) + '_' + load_type + '_' + str(epoch) +'.obj'
        set_fromIDX = "{:04d}".format(int(real_idx/500)*500) + '_' + "{:04d}".format(((int(real_idx/500)+1)*500)-1)
        gt_path = dataset_param['dataset_directory']+ '/Geometry/' + dataset_param['dataset_type'] + '/' + set_fromIDX + '/' + g_name + '.obj'

        chamfer_dist = None
        wasserstein_dist = None
        min_dist_accuracy = None
        mesh_comp = None
        cos_similarity = None

        if(os.path.exists(rc_path) and os.path.exists(gt_path)):
            #print('rc_path and gt_path exist.')
            if((not isinstance(trimesh.load_mesh(gt_path, skip_materials=True), trimesh.PointCloud)) and (not sdf_only)):
                #print('Mesh not a point cloud.')

                ground_truth_mesh = normalize_mesh(trimesh.load_mesh(gt_path, skip_materials=True, force='mesh'), buffered_bbox)
                gt_mesh_rot_vert = ground_truth_mesh.vertices + [-0.5,-0.5,-0.5]
                gt_mesh_rot_vert = np.concatenate([gt_mesh_rot_vert,np.ones((gt_mesh_rot_vert.shape[0],1))],1)
                gt_mesh_rot_vert = np.dot(rotation_matrix_y(rot),gt_mesh_rot_vert.T).T[:,:3].round(6)
                gt_mesh_rot_vert = gt_mesh_rot_vert + [0.5,0.5,0.5]
                ground_truth_mesh = trimesh.Trimesh(vertices=gt_mesh_rot_vert, faces=ground_truth_mesh.faces)

                reconstructed_mesh = trimesh.load(reconstruction_dir + '/' + 'reconstruction_' + str(resolution) + '_index_' + str(index) + '_' + load_type + '_' + str(epoch) +'.obj', force='mesh')
                print('reconstructed_mesh',reconstructed_mesh)
                if(isinstance(reconstructed_mesh, trimesh.Trimesh)):

                    # Chamfer distance (from DeepSDF supplementary)
                    # The sum of the nearest neighbor distances for each point to the nearest point in the other point set.
                    ground_truth_pc = trimesh.sample.sample_surface(ground_truth_mesh, num_mesh_samples_chamfer)[0]
                    reconstructed_pc = trimesh.sample.sample_surface(reconstructed_mesh, num_mesh_samples_chamfer)[0]
                    grd_kd_tree = KDTree(ground_truth_pc)
                    grd_distances, grd_vertex_ids = grd_kd_tree.query(reconstructed_pc)
                    grd_to_rec_chamfer = np.mean(np.square(grd_distances))
                    rec_kd_tree = KDTree(reconstructed_pc)
                    rec_distances, rec_vertex_ids = rec_kd_tree.query(ground_truth_pc)
                    rec_to_grd_chamfer = np.mean(np.square(rec_distances))
                    chamfer_dist = grd_to_rec_chamfer + rec_to_grd_chamfer
                    #print('chamfer_dist',chamfer_dist)

                    # Earth mover's distance (from DeepSDF supplementary)
                    # a.k.a. the Wasserstein distance, is another popular metric for measuring the difference between two discrete distributions.
                    ground_truth_pc = trimesh.sample.sample_surface(ground_truth_mesh, num_mesh_samples_wasserstein)[0]
                    reconstructed_pc = trimesh.sample.sample_surface(reconstructed_mesh, num_mesh_samples_wasserstein)[0]
                    #wasserstein_dist = wasserstein_distance(ground_truth_pc.T, reconstructed_pc.T)
                    sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
                    wasserstein_dist = sinkhorn_loss(torch.tensor(ground_truth_pc).unsqueeze(0), torch.tensor(reconstructed_pc).unsqueeze(0)).detach().numpy()[0]
                    #print('earth_movers_dist',wasserstein_dist)

                    # Mesh accuracy (from DeepSDF supplementary)
                    # The minimum distance d such that 90% of generated points are within d of the ground truth mesh.
                    reconstructed_pc = trimesh.sample.sample_surface(reconstructed_mesh, num_mesh_samples_mesh)[0]
                    dist_from_gt_mesh = np.absolute(igl.signed_distance(reconstructed_pc, ground_truth_mesh.vertices, ground_truth_mesh.faces)[0])
                    sorted_dist = np.sort(dist_from_gt_mesh)
                    min_dist_accuracy = sorted_dist[int(0.9*num_mesh_samples_mesh)]
                    #print('mesh_accuracy',min_dist_accuracy)

                    # Mesh completion  (from DeepSDF supplementary)
                    # fraction of points sampled from the ground truth mesh that are within some distance x (a parameter of the metric) to the generated mesh.
                    # Ideal mesh completion is 1.0, minimum is 0.0.
                    ground_truth_pc = trimesh.sample.sample_surface(ground_truth_mesh, num_mesh_samples_mesh)[0]
                    dist_from_rt_mesh = np.absolute(igl.signed_distance(ground_truth_pc, reconstructed_mesh.vertices, reconstructed_mesh.faces)[0])
                    mesh_comp = len(dist_from_rt_mesh[dist_from_rt_mesh<mesh_cp_param])/num_mesh_samples_mesh
                    #print('mesh_completion',mesh_comp)

                    # Mesh cosine similarity (from DeepSDF supplementary)
                    # The mean cosine similarity between the normals of points sampled from the ground truth mesh, and the normals of the nearest faces of the generated mesh
                    # Ideal cosine similarity is 1.0, minimum (given the allowed flip of the normal) is 0.0.
                    gt_points, gt_index = ground_truth_mesh.sample(num_mesh_samples_cossim, return_index=True)
                    gt_normals = ground_truth_mesh.face_normals[gt_index]
                    cl_pt, dist_pt, face_id = trimesh.proximity.closest_point(reconstructed_mesh, gt_points)
                    rt_normals = reconstructed_mesh.face_normals[face_id]
                    cos_similarity = np.minimum(np.absolute(cosine_similarity(torch.tensor(gt_normals), torch.tensor(rt_normals))), np.absolute(cosine_similarity(torch.tensor(gt_normals), torch.tensor(-1*rt_normals)))).mean()
                    #print('mesh_cosine_similarity',cos_similarity)
                    #print('Geometry metrics calculated for geometry with index ' + str(index))
                    file_name = 'metrics_'+ str(resolution) + '_index_' + str(index) + '_' + load_type + '_' + str(epoch)
                else:
                    file_name = None
                    os.remove(reconstruction_dir + '/' + 'reconstruction_' + str(resolution) + '_index_' + str(index) + '_' + load_type + '_' + str(epoch) +'.obj')
                    print('Reconstructed mesh is not a mesh for index ' + str(index) + '!')
            else:
                file_name = 'metrics_'+ str(resolution) + '_index_' + str(index) + '_' + load_type + '_' + str(epoch) + '_sdflossOnly'
                print('Reconstruction mesh file for geometry with index ' + str(index) + ' does not exist or could not be generated 0-vertices')
        else:
            print('Reconstruction geometry or ground truth geometry do not exist.')
            file_name = 'metrics_'+ str(resolution) + '_index_' + str(index) + '_' + load_type + '_' + str(epoch) + '_sdflossOnly'

        # Save metrics
        if(file_name is not None):
            np.save(metrics_dir + '/' + file_name, np.array([index, real_idx, sdf_loss_l1, sdf_loss_l2, chamfer_dist, wasserstein_dist, min_dist_accuracy, mesh_comp, cos_similarity]))
            print('Metrics calculations done for index ' + str(index) + '!')
    else:
        print('Geometry/Sampling does not exist in dataset for indec ' + str(index) + '!')
    
    
    
    
def logModelUpdates(models, model_names, train_losses, d_arr, d_arr_names, model_dir, log_type, epoch=None, val_losses = None, val_d_arr = None):
    epoch = epoch if(epoch is not None) else -1
    ext_str = {'current':'', 'final':'', 'epoch':'_%04d' % epoch}
    fol_base = {'current':model_dir, 'final':model_dir, 'epoch':model_dir + '/' + 'checkpoints'}
    # Save models
    [torch.save(models[i].state_dict(),os.path.join(fol_base[log_type], model_names[i]+'_'+log_type+ext_str[log_type]+'.pth')) for i in range(len(models))]
    # Save losses
    np.savetxt(os.path.join(fol_base[log_type], 'train_losses_' + log_type + ext_str[log_type] + '.txt'), train_losses)
    train_losses_detailed = {}
    for r in range(len(d_arr)):
        train_losses_detailed[d_arr_names[r]] = d_arr[r]
    with open(fol_base[log_type] + '/' + 'train_losses_detailed_'+log_type + ext_str[log_type]+'.json', 'w') as f:
        json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)
    # Save validation losses
    if(val_losses is not None):
        np.savetxt(os.path.join(fol_base[log_type], 'val_losses_' + log_type + ext_str[log_type] + '.txt'), val_losses)
        val_losses_detailed = {}
        for r in range(len(val_d_arr)):
            val_losses_detailed[d_arr_names[r]] = val_d_arr[r]
        with open(fol_base[log_type] + '/' + 'val_losses_detailed_'+log_type + ext_str[log_type]+'.json', 'w') as f:
            json.dump(val_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)
            
def loadfromCheckpoint(models, model_names, checkpoints_dir, d_arr, d_arr_names, epoch=None, d_arr_val=None):
    if(epoch is not None):
        for i in range(len(models)):
            file_to_load = os.path.join(checkpoints_dir, model_names[i]+'_epoch_%04d.pth'% epoch)
            try:
                models[i].load_state_dict(torch.load(file_to_load)) if(device != 'cpu') else models[i].load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu')))    
            except:
                try:
                    models[i] = torch.nn.DataParallel(models[i])
                    models[i].load_state_dict(torch.load(file_to_load)) if(device != 'cpu') else models[i].load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu'))) 
                except:
                    models[i] = torch.nn.DataParallel(models[i])
                    models[i] = torch.nn.DataParallel(models[i])
                    models[i].load_state_dict(torch.load(file_to_load)) if(device != 'cpu') else models[i].load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu'))) 
                    models[i] = models[i].module
        train_losses = list(np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch))).flatten().astype(float))
        train_check_det = json.loads(open(checkpoints_dir + '/' + 'train_losses_detailed_epoch_%04d.json' % epoch).read())
        for t in range(len(d_arr)):
            d_arr[t] = train_check_det[d_arr_names[t]]
        if(d_arr_val is not None):
            val_losses = list(np.array(np.loadtxt(os.path.join(checkpoints_dir, 'val_losses_epoch_%04d.txt' % epoch))).flatten().astype(float))
            val_check_det = json.loads(open(checkpoints_dir + '/' + 'val_losses_detailed_epoch_%04d.json' % epoch).read())
            for t in range(len(d_arr)):
                d_arr_val[t] = val_check_det[d_arr_names[t]]
            return models, train_losses, d_arr, val_losses, d_arr_val
        else:
            return models, train_losses, d_arr
    else:
        if(d_arr_val is not None):
            print('The checkpoint does not exist.')
            return None, None, None, None, None
        else:
            print('The checkpoint does not exist.')
            return None, None, None

def calculatePerfMetricsPerGeo(xyz_sdf_perf, g_names, perf_metric, model, i, metrics_file, override = False, pc_list = [5,10,25,50,75,90,95], space_types = ['srf','grd','field'], pt_sets = {'srf':'srf_20', 'grd':'XYgrid_512_30_15'},crop_min = 0.1464,crop_max = 0.8536): #crop_min = 0.1464,crop_max = 0.8536
    b_name = g_names[i]
    idx = int(b_name.split('_')[1])
    geo_type = b_name.split('_')[0]
    orien_val = int(b_name.split('_')[2])
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    cur_spaces = []
    factor = 20 if(perf_metric=='U') else 1
    thr = [0.001,100] if(perf_metric=='U') else [0,1]
    # Calculate metrics
    count = 0
    if((not os.path.exists(metrics_file)) or override):
        all_stats = []
        for s in space_types:
            if(s != 'field'):
                p_set = pt_sets[s]
                outputs = xyz_sdf_perf.__getSelPtSet__(idx, geo_type, perf_metric, p_set, orien_val, fil_bBox = False, return_fact = False)
                xyz, xyz_org, sdf, perf = outputs
            else:
                outputs = xyz_sdf_perf.__getitem__(idx, geo_type, perf_metric, orien_val, return_fact = False)
                xyz, xyz_org, sdf, perf = outputs
            if(perf is not None):
                # print((xyz[:, 0] >= crop_min).shape)
                # print((sdf > 0).squeeze(-1).shape)
                xyz_fil = (
                            (xyz[:, 0] >= crop_min) & (xyz[:, 0] <= crop_max) &
                            (xyz[:, 1] >= 0) & (xyz[:, 1] <= 1) &
                            (xyz[:, 2] >= crop_min) & (xyz[:, 2] <= crop_max) &
                            (sdf > 0).squeeze(-1)
                        )
                # print('s',s)
                # print('x.min.0',xyz[:, 0].min())
                # print('x.min.0',xyz[:, 0].max())
                # print('x.min.1',xyz[:, 1].min())
                # print('x.min.1',xyz[:, 1].max())
                # print('x.min.2',xyz[:, 0].min())
                # print('x.min.2',xyz[:, 0].max())
                xyz = xyz[xyz_fil]
                sdf = sdf[xyz_fil]
                perf = perf[xyz_fil] 
                model_input = {'xyz': xyz.unsqueeze(0),
                        'idx': torch.tensor([idx]).unsqueeze(0),
                        'rot': torch.tensor([orien_val]).unsqueeze(0), # if(self.perf_metric in ['U','P']) else torch.tensor([0]),
                        'item_n': geo_type,
                        'geo_n': geo_type + '_' + str(idx),
                        'geo_code': torch.tensor([0,idx]).unsqueeze(0) if(geo_type=='building') else torch.tensor([1,idx]).unsqueeze(0)}
                del xyz, xyz_org, sdf
                ground_truth = perf
                geo_name = model_input['item_n'][0]
                model_output = model(model_input)
                output, ctxt_latents = model_output
                #xyz_fil = ((model_input['xyz'][..., 0]>=0)&(model_input['xyz'][..., 0]<=1)&(model_input['xyz'][..., 1]>=0)&(model_input['xyz'][..., 1]<=1)&(model_input['xyz'][..., 2]>=0)&(model_input['xyz'][..., 2]<=1)).detach().numpy().flatten()
                # xyz_fil = ((model_input['xyz'][..., 0]>=crop_min)&(model_input['xyz'][..., 0]<=crop_max)&(model_input['xyz'][..., 1]>=0)&(model_input['xyz'][..., 1]<=1)&(model_input['xyz'][..., 2]>=crop_min)&(model_input['xyz'][..., 2]<=crop_max)&(sdf.unsqueeze(0).squeeze(-1)>0)).detach().numpy().flatten()
                # del sdf
                perf = output[..., :1].detach().numpy().flatten()*factor
                gt_perf = ground_truth[..., :1].detach().numpy().flatten()*factor
                #perf = output[..., :1].detach().numpy().flatten()[xyz_fil]*factor
                #gt_perf = ground_truth[..., :1].detach().numpy().flatten()[xyz_fil]*factor
                #print('gt_perf',gt_perf.shape)
                #perf_fil = (((gt_perf >= thr[0]) & (gt_perf < thr[1]) & (perf >= thr[0]) & (perf < thr[1])) | ((perf <= thr[0]) & (gt_perf <= thr[0])))
                #perf_fil = (((gt_perf > 0.000000000000000001) & (perf > 0.00001)) | (gt_perf >= 0.000000000000000001))#(((gt_perf >= thr[0]) & (gt_perf < thr[1]) & (perf >= thr[0]) & (perf < thr[1])) | ((perf <= thr[0]) & (gt_perf <= thr[0])))
                # gt_perf = gt_perf[perf_fil]
                # perf = perf[perf_fil]
                # inside_mask = gt_perf<0.00001
                # perf = perf[inside_mask]
                # gt_perf = gt_perf[inside_mask]
                # print('gt_perf.size',gt_perf.shape)
                # print('Pre-calcs')
                if gt_perf.size > 0 and perf.size > 0:
                    abs_diff = np.absolute(gt_perf-perf)
                    sq_diff = abs_diff**2
                    absP_diff = np.clip(abs_diff/(gt_perf+0.0000000000001), -1e4, 1e4)
                    sqP_diff = np.clip(absP_diff**2, -1e4, 1e4)
                    # print('Pre-KL')
                    kl_loss_val = kl_loss(F.log_softmax(output[..., :1].flatten(), dim=-1), F.softmax(ground_truth[..., :1].flatten(), dim=0)).detach().cpu().numpy() 
                    # print('Post-KL')
                    diff_stats = np.array([[x for xs in [[np.mean(d),np.std(d),np.min(d),np.max(d)],[np.percentile(d, p) for p in pc_list]] for x in xs] for d in [abs_diff,sq_diff,absP_diff,sqP_diff]]).flatten()
                    diff_stats = np.append(diff_stats, kl_loss_val)
                    diff_stats = np.append(diff_stats, s)
                else:
                    print(f"[Warning] Empty perf arrays after filtering in {b_name}, space: {s}")
                    count += 1
                    diff_stats = np.repeat(None, ((4+len(pc_list))*4)+1)
                    diff_stats = np.append(diff_stats, s)
            else:
                count += 1
                diff_stats = np.repeat(None, ((4+len(pc_list))*4)+1)
                diff_stats = np.append(diff_stats, s)
            all_stats.append(diff_stats)
        all_stats = np.array(all_stats)
        if(all_stats.shape[0]==len(space_types)):
            if(count == 0):
                np.save(metrics_file, all_stats)
                print('Metrics for geometry: ' + str(b_name) + ' have been calculated and saved. ' + str(all_stats.shape))
            else:
                np.save(metrics_file.replace('metrics_','partialmetrics_'), all_stats)
                print('One or more point sets do not exist.')
                print('Partial metrics for geometry: ' + str(b_name) + ' have been calculated and saved. ' + str(all_stats.shape))
        else:
            print('All sets do not exist.')
            
# def calculatePerfMetricsPerGeo(xyz_sdf_perf, g_names, perf_metric, model, i, metrics_file, override = False, pc_list = [5,10,25,50,75,90,95], space_types = ['srf','grd','field'], pt_sets = {'srf':'srf_20', 'grd':'XYgrid_512_30_15'},crop_min = 0.1464,crop_max = 0.8536): #crop_min = 0.1464,crop_max = 0.8536
#     b_name = g_names[i]
#     idx = int(b_name.split('_')[1])
#     geo_type = b_name.split('_')[0]
#     orien_val = int(b_name.split('_')[2])
#     kl_loss = nn.KLDivLoss(reduction="batchmean")
#     cur_spaces = []
#     factor = 20 if(perf_metric=='U') else 1
#     thr = [0.001,100] if(perf_metric=='U') else [0,1]
#     # Calculate metrics
#     count = 0
#     if((not os.path.exists(metrics_file)) or override):
#         all_stats = []
#         for s in space_types:
#             if(s != 'field'):
#                 p_set = pt_sets[s]
#                 outputs = xyz_sdf_perf.__getSelPtSet__(idx, geo_type, perf_metric, p_set, orien_val, fil_bBox = False, return_fact = False)
#                 xyz, xyz_org, sdf, perf = outputs
#             else:
#                 outputs = xyz_sdf_perf.__getitem__(idx, geo_type, perf_metric, orien_val, return_fact = False)
#                 xyz, xyz_org, sdf, perf = outputs
#             if(perf is not None):
#                 model_input = {'xyz': xyz.unsqueeze(0),
#                         'idx': torch.tensor([idx]).unsqueeze(0),
#                         'rot': torch.tensor([orien_val]).unsqueeze(0), # if(self.perf_metric in ['U','P']) else torch.tensor([0]),
#                         'item_n': geo_type,
#                         'geo_n': geo_type + '_' + str(idx),
#                         'geo_code': torch.tensor([0,idx]).unsqueeze(0) if(geo_type=='building') else torch.tensor([1,idx]).unsqueeze(0)}
#                 del xyz, xyz_org
#                 ground_truth = perf
#                 geo_name = model_input['item_n'][0]
#                 model_output = model(model_input)
#                 output, ctxt_latents = model_output
#                 #xyz_fil = ((model_input['xyz'][..., 0]>=0)&(model_input['xyz'][..., 0]<=1)&(model_input['xyz'][..., 1]>=0)&(model_input['xyz'][..., 1]<=1)&(model_input['xyz'][..., 2]>=0)&(model_input['xyz'][..., 2]<=1)).detach().numpy().flatten()
#                 xyz_fil = ((model_input['xyz'][..., 0]>=crop_min)&(model_input['xyz'][..., 0]<=crop_max)&(model_input['xyz'][..., 1]>=0)&(model_input['xyz'][..., 1]<=1)&(model_input['xyz'][..., 2]>=crop_min)&(model_input['xyz'][..., 2]<=crop_max)&(sdf.unsqueeze(0).squeeze(-1)>0)).detach().numpy().flatten()
#                 del sdf
#                 perf = output[..., :1].detach().numpy().flatten()[xyz_fil]*factor
#                 gt_perf = ground_truth[..., :1].detach().numpy().flatten()[xyz_fil]*factor
#                 #print('gt_perf',gt_perf.shape)
#                 #perf_fil = (((gt_perf >= thr[0]) & (gt_perf < thr[1]) & (perf >= thr[0]) & (perf < thr[1])) | ((perf <= thr[0]) & (gt_perf <= thr[0])))
#                 #perf_fil = (((gt_perf > 0.000000000000000001) & (perf > 0.00001)) | (gt_perf >= 0.000000000000000001))#(((gt_perf >= thr[0]) & (gt_perf < thr[1]) & (perf >= thr[0]) & (perf < thr[1])) | ((perf <= thr[0]) & (gt_perf <= thr[0])))
#                 # gt_perf = gt_perf[perf_fil]
#                 # perf = perf[perf_fil]
#                 # inside_mask = gt_perf<0.00001
#                 # perf = perf[inside_mask]
#                 # gt_perf = gt_perf[inside_mask]
#                 print('gt_perf.size',gt_perf.shape)
#                 print('Pre-calcs')
#                 if gt_perf.size > 0 and perf.size > 0:
#                     abs_diff = np.absolute(gt_perf-perf)
#                     sq_diff = abs_diff**2
#                     absP_diff = np.clip(abs_diff/(gt_perf+0.0000000000001), -1e4, 1e4)
#                     sqP_diff = np.clip(absP_diff**2, -1e4, 1e4)
#                     print('Pre-KL')
#                     kl_loss_val = kl_loss(F.log_softmax(output[..., :1].flatten(), dim=-1), F.softmax(ground_truth[..., :1].flatten(), dim=0)).detach().cpu().numpy() 
#                     print('Post-KL')
#                     diff_stats = np.array([[x for xs in [[np.mean(d),np.std(d),np.min(d),np.max(d)],[np.percentile(d, p) for p in pc_list]] for x in xs] for d in [abs_diff,sq_diff,absP_diff,sqP_diff]]).flatten()
#                     diff_stats = np.append(diff_stats, kl_loss_val)
#                     diff_stats = np.append(diff_stats, s)
#                 else:
#                     print(f"[Warning] Empty perf arrays after filtering in {b_name}, space: {s}")
#                     count += 1
#                     diff_stats = np.repeat(None, ((4+len(pc_list))*4)+1)
#                     diff_stats = np.append(diff_stats, s)
#             else:
#                 count += 1
#                 diff_stats = np.repeat(None, ((4+len(pc_list))*4)+1)
#                 diff_stats = np.append(diff_stats, s)
#             all_stats.append(diff_stats)
#         all_stats = np.array(all_stats)
#         if(all_stats.shape[0]==len(space_types)):
#             if(count == 0):
#                 np.save(metrics_file, all_stats)
#                 print('Metrics for geometry: ' + str(b_name) + ' have been calculated and saved. ' + str(all_stats.shape))
#             else:
#                 np.save(metrics_file.replace('metrics_','partialmetrics_'), all_stats)
#                 print('One or more point sets do not exist.')
#                 print('Partial metrics for geometry: ' + str(b_name) + ' have been calculated and saved. ' + str(all_stats.shape))
#         else:
#             print('All sets do not exist.')
        
# def calculatePerfMetricsPerGeo(xyz_sdf_perf, g_names, perf_metric, model, i, metrics_file, override = False, pc_list = [5,10,25,50,75,90,95], space_types = ['srf','grd','field'], pt_sets = {'srf':'srf_20', 'grd':'XYgrid_512_30_15'}):
#     b_name = g_names[i]
#     idx = int(b_name.split('_')[1])
#     geo_type = b_name.split('_')[0]
#     orien_val = int(b_name.split('_')[2])
#     kl_loss = nn.KLDivLoss(reduction="batchmean")
#     cur_spaces = []
#     # Calculate metrics
#     count = 0
#     if((not os.path.exists(metrics_file)) or override):
#         all_stats = []
#         for s in space_types:
#             if(s != 'field'):
#                 p_set = pt_sets[s]
#                 outputs = xyz_sdf_perf.__getSelPtSet__(idx, geo_type, perf_metric, p_set, orien_val, fil_bBox = True, return_fact = False)
#                 xyz, xyz_org, sdf, perf = outputs
#             else:
#                 outputs = xyz_sdf_perf.__getitem__(idx, geo_type, perf_metric, orien_val, return_fact = False)
#                 xyz, xyz_org, sdf, perf = outputs
#             if(perf is not None):
#                 model_input = {'xyz': xyz.unsqueeze(0),
#                         'idx': torch.tensor([idx]).unsqueeze(0),
#                         'rot': torch.tensor([orien_val]).unsqueeze(0), # if(self.perf_metric in ['U','P']) else torch.tensor([0]),
#                         'item_n': geo_type,
#                         'geo_n': geo_type + '_' + str(idx),
#                         'geo_code': torch.tensor([0,idx]).unsqueeze(0) if(geo_type=='building') else torch.tensor([1,idx]).unsqueeze(0)}
#                 ground_truth = perf
#                 geo_name = model_input['item_n'][0]
#                 model_output = model(model_input)
#                 output, ctxt_latents = model_output
#                 xyz_fil = ((model_input['xyz'][..., 0]>=0)&(model_input['xyz'][..., 0]<=1)&(model_input['xyz'][..., 1]>=0)&(model_input['xyz'][..., 1]<=1)&(model_input['xyz'][..., 2]>=0)&(model_input['xyz'][..., 2]<=1)).detach().numpy().flatten()
#                 #print('perf',output[..., :1].detach().numpy().flatten().shape)
#                 perf = output[..., :1].detach().numpy().flatten()[xyz_fil]
#                 #print('perf',perf.shape)
#                 gt_perf = ground_truth[..., :1].detach().numpy().flatten()[xyz_fil]
#                 abs_diff = np.absolute(gt_perf-perf)
#                 sq_diff = abs_diff**2
#                 absP_diff = np.clip(abs_diff/(gt_perf+0.0000000000001), -1e10, 1e10)
#                 sqP_diff = np.clip(absP_diff**2, -1e10, 1e10)
#                 kl_loss_val = kl_loss(F.log_softmax(output[..., :1].flatten(), dim=-1), F.softmax(ground_truth[..., :1].flatten(), dim=0)).detach().cpu().numpy() 
#                 diff_stats = np.array([[x for xs in [[np.mean(d),np.std(d),np.min(d),np.max(d)],[np.percentile(d, p) for p in pc_list]] for x in xs] for d in [abs_diff,sq_diff,absP_diff,sqP_diff]]).flatten()
#                 diff_stats = np.append(diff_stats, kl_loss_val)
#                 diff_stats = np.append(diff_stats, s)
#             else:
#                 count += 1
#                 diff_stats = np.repeat(None, ((4+len(pc_list))*4)+1)
#                 diff_stats = np.append(diff_stats, s)
#             all_stats.append(diff_stats)
#         all_stats = np.array(all_stats)
#         if(all_stats.shape[0]==len(space_types)):
#             if(count == 0):
#                 np.save(metrics_file, all_stats)
#                 print('Metrics for geometry: ' + str(b_name) + ' have been calculated and saved. ' + str(all_stats.shape))
#             else:
#                 np.save(metrics_file.replace('metrics_','partialmetrics_'), all_stats)
#                 print('One or more point sets do not exist.')
#                 print('Partial metrics for geometry: ' + str(b_name) + ' have been calculated and saved. ' + str(all_stats.shape))
#         else:
#             print('All sets do not exist.')
            
def calculatePerfMetricsPerGeoIMG(xyz_sdf_perf, g_names, perf_metric, model, i, metrics_file, override = False, pc_list = [5,10,25,50,75,90,95], space_types = ['grd'], pt_sets = {'grd':'XYgrid_512_30_15'}, inc_localMap=True,crop_min = 0.1464,crop_max = 0.8536):
    b_name = g_names[i]
    idx = int(b_name.split('_')[1])
    geo_type = b_name.split('_')[0]
    orien_val = int(b_name.split('_')[2])
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    cur_spaces = []
    factor = 20 if(perf_metric=='U') else 1
    thr = [0.001,100] if(perf_metric=='U') else [0,1]
    # Calculate metrics
    count = 0
    if((not os.path.exists(metrics_file)) or override):
        all_stats = []
        for s in space_types:
            p_set = pt_sets[s]
            outputs = xyz_sdf_perf.__getSelPtSetHgtMap__(idx, geo_type, perf_metric, p_set, orien_val, return_fact = False)
            xyz, xyz_org, sdf, perf, hgtmap = outputs
            if(perf is not None):
                model_input = {'xyz': xyz.unsqueeze(0),
                        'idx': torch.tensor([idx]).unsqueeze(0),
                        'hgtmap':hgtmap.unsqueeze(0) if(inc_localMap) else hgtmap[:,:1].unsqueeze(0),
                        'rot': torch.tensor([orien_val]).unsqueeze(0), # if(self.perf_metric in ['U','P']) else torch.tensor([0]),
                        'item_n': geo_type,
                        'geo_n': geo_type + '_' + str(idx),
                        'geo_code': torch.tensor([0,idx]).unsqueeze(0) if(geo_type=='building') else torch.tensor([1,idx]).unsqueeze(0)}
                ground_truth = perf
                geo_name = model_input['item_n'][0]
                model_output = model(model_input)
                output, ctxt_latents = model_output
                #xyz_fil = ((model_input['xyz'][..., 0]>=0)&(model_input['xyz'][..., 0]<=1)&(model_input['xyz'][..., 1]>=0)&(model_input['xyz'][..., 1]<=1)&(model_input['xyz'][..., 2]>=0)&(model_input['xyz'][..., 2]<=1)).detach().numpy().flatten()
                xyz_fil = ((model_input['xyz'][..., 0]>=crop_min)&(model_input['xyz'][..., 0]<=crop_max)&(model_input['xyz'][..., 1]>=0)&(model_input['xyz'][..., 1]<=1)&(model_input['xyz'][..., 2]>=crop_min)&(model_input['xyz'][..., 2]<=crop_max)&(sdf.unsqueeze(0).squeeze(-1)>0)).detach().numpy().flatten()
                #print('perf',output[..., :1].detach().numpy().flatten().shape)
                perf = output[..., :1].detach().numpy().flatten()[xyz_fil]*factor
                #print('perf',perf.shape)
                gt_perf = ground_truth[..., :1].detach().numpy().flatten()[xyz_fil]*factor
                # perf_fil = (((gt_perf < 0.000000000000000001) & (perf > 0.00001)) | (gt_perf >= 0.000000000000000001))#(((gt_perf >= thr[0]) & (gt_perf < thr[1]) & (perf >= thr[0]) & (perf < thr[1])) | ((perf <= thr[0]) & (gt_perf <= thr[0])))
                # gt_perf = gt_perf[perf_fil]
                # perf = perf[perf_fil]
                # inside_mask = gt_perf<0.00001
                # perf = perf[inside_mask]
                # gt_perf = gt_perf[inside_mask]
                if gt_perf.size > 0 and perf.size > 0:
                    abs_diff = np.absolute(gt_perf-perf)
                    sq_diff = abs_diff**2
                    absP_diff = np.clip(abs_diff/(gt_perf+0.0000000000001), -1e4, 1e4)
                    sqP_diff = np.clip(absP_diff**2, -1e4, 1e4)
                    kl_loss_val = kl_loss(F.log_softmax(output[..., :1].flatten(), dim=-1), F.softmax(ground_truth[..., :1].flatten(), dim=0)).detach().cpu().numpy() 
                    diff_stats = np.array([[x for xs in [[np.mean(d),np.std(d),np.min(d),np.max(d)],[np.percentile(d, p) for p in pc_list]] for x in xs] for d in [abs_diff,sq_diff,absP_diff,sqP_diff]]).flatten()
                    diff_stats = np.append(diff_stats, kl_loss_val)
                    diff_stats = np.append(diff_stats, s)
                else:
                    print(f"[Warning] Empty perf arrays after filtering in {b_name}, space: {s}")
                    count += 1
                    diff_stats = np.repeat(None, ((4+len(pc_list))*4)+1)
                    diff_stats = np.append(diff_stats, s)
            else:
                count += 1
                diff_stats = np.repeat(None, ((4+len(pc_list))*4)+1)
                diff_stats = np.append(diff_stats, s)
            all_stats.append(diff_stats)
        all_stats = np.array(all_stats)
        if(all_stats.shape[0]==len(space_types)):
            if(count == 0):
                np.save(metrics_file, all_stats)
                print('Metrics for geometry: ' + str(b_name) + ' have been calculated and saved. ' + str(all_stats.shape))
            else:
                np.save(metrics_file.replace('metrics_','partialmetrics_'), all_stats)
                print('One or more point sets do not exist.')
                print('Partial metrics for geometry: ' + str(b_name) + ' have been calculated and saved. ' + str(all_stats.shape))
        else:
            print('All sets do not exist.')

           
    
# def calculatePerfMetricsPerGeo(xyz_sdf_perf_dataset, model, i, point_sets, point_set_names, perf_metric, metrics_dir, load_type, epoch, override = False):
#     kl_loss = nn.KLDivLoss(reduction="batchmean")
#     model_input, perf = xyz_sdf_perf_dataset.__getitembySet__(i, point_sets[0][0], fil_bBox = False, include_sdf = False, formatforModel = True)
#     metrics_file = metrics_dir + '/' + 'metrics_index_' + str(i) + '_name_' + model_input['item_n'][0]  + '_' + load_type + '_' + str(epoch) + '.npy'
#     if(((not os.path.exists(metrics_file)) or override)and(perf is not None)):
#         l_types = ['d','abD'] if(perf_metric in ['P','SVF']) else ['d','abD','cS']
#         p_cols = list(np.array([[tp + '_' + d for d in ['mean','std','min','p25','p50','p75','max']] for tp in l_types]).flatten())
#         l_plus = ['perf_l1','perf_l2','perf_kl'] if(perf_metric in ['P','SVF']) else ['perf_l1','perf_l2','perf_kl','perf_cS']
#         [p_cols.append(l) for l in l_plus]
#         cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
#         p_diff_ar, p_abdiff_ar, cos_sim_ar, l1_ar, l2_ar, cos_ar, kl_ar = [], [], [], [], [], [], []
#         for point_set in point_sets:
#             perf_pred_vals, perf_gt_vals, perf_diff, abs_perf_dif, cos_sim, kl = [], [], [], [], [], []
#             for p in point_set:
#                 # Load geometry and center the mesh
#                 model_input, perf = xyz_sdf_perf_dataset.__getitembySet__(i, p, fil_bBox = False, include_sdf = False, formatforModel = True)
#                 if(model_input['xyz'] is not None):
#                     if(device.type != 'cpu'):
#                         model_input = to_gpu(model_input)
#                     #print(model)
#                     model_output = model(model_input)
#                     #print(model_output[0].shape)
#                     perf_gt = perf[..., :1].cpu().numpy().flatten()
#                     pred_p = model_output[0][..., :1].detach().numpy().flatten()
#                     perf_diff.append(list(pred_p-perf_gt)) 
#                     abs_perf_dif.append(list(np.absolute(pred_p-perf_gt)))
#                     perf_pred_vals.append(model_output[0][..., :1].detach().cpu().numpy().flatten())
#                     perf_gt_vals.append(perf[..., :1].detach().cpu().numpy().flatten())
#                     if(perf_metric in ['U']):
#                         vec = model_output[0][..., 1:].detach().cpu()
#                         gt_vec = torch.unsqueeze(perf[..., 1:],0).detach().cpu()
#                         # print('vec',vec.shape)
#                         # print('gt_vec',gt_vec.shape)
#                         cos_sim_loss = (cosine_similarity(vec, gt_vec)+1)/2
#                         cos_sim.append(cos_sim_loss.detach().cpu().numpy().flatten())

#             #print(torch.tensor(perf_pred_vals).shape)
#                     perf_pred_cat = torch.from_numpy(np.concatenate([perf_pred_vals]))
#                     perf_gt_cat = torch.from_numpy(np.concatenate([perf_gt_vals]))
#                     kl_loss_val = kl_loss(F.log_softmax(perf_pred_cat, dim=-1), F.softmax(perf_gt_cat, dim=1)).detach().cpu().numpy() #if(perf_pred_cat.shape[0]>0) else 0
#                     # print(kl_loss_val)
#                     kl.append(kl_loss_val)
#             #print('kl',kl)
#             perf_diff = np.array(perf_diff).flatten()
#             abs_perf_dif = np.array(abs_perf_dif).flatten()
#             p_diff_ar.append(list(pd.DataFrame(perf_diff).describe().fillna(0).values.flatten())[1:]) 
#             p_abdiff_ar.append(list(pd.DataFrame(abs_perf_dif).describe().fillna(0).values.flatten())[1:]) 
#             l1_ar.append(abs_perf_dif.mean()) if(len(abs_perf_dif)>0) else l1_ar.append(0)
#             l2_ar.append((abs_perf_dif**2).mean()) if(len(abs_perf_dif)>0) else l2_ar.append(0)
#             kl_ar.append(np.array(kl).mean()) if(len(kl)>0) else kl_ar.append(0)
#             if(perf_metric in ['U']):
#                 cos_sim = np.array(cos_sim).flatten()
#                 cos_sim_ar.append(list(pd.DataFrame(cos_sim).describe().fillna(0).values.flatten())[1:])
#                 cos_ar.append(cos_sim.mean()) if(len(cos_sim)>0) else cos_ar.append(0)
#         p_diff_ar = np.array(p_diff_ar)
#         p_abdiff_ar = np.array(p_abdiff_ar)
#         cos_sim_ar = np.array(cos_sim_ar)
#         l1_ar = np.expand_dims(np.array(l1_ar),-1) 
#         l2_ar = np.expand_dims(np.array(l2_ar),-1)
#         kl_ar = np.expand_dims(np.array(kl_ar),-1)
#         if(perf_metric in ['U']):
#             cos_sim_ar = np.array(cos_sim_ar)
#             cos_ar = np.expand_dims(np.array(cos_ar),-1)
#         all_m = np.concatenate([p_diff_ar, p_abdiff_ar, cos_sim_ar, np.array(l1_ar), np.array(l2_ar), kl_ar, cos_ar],-1) if(perf_metric in ['U']) else np.concatenate([p_diff_ar, p_abdiff_ar, np.array(l1_ar), np.array(l2_ar), kl_ar],-1)
#         np.save(metrics_file, all_m)
#         print('Metrics for geometry: ' + str(model_input['item_n'][0]) + ' have been calculated and saved. ' + str(all_m.shape) + ', m_types: ' + str(len(p_cols)))
#         np.save(metrics_dir + '/' + 'perf_cols.npy', p_cols) 
#         np.save(metrics_dir + '/' + 'point_set_names.npy', point_set_names)

def combineMetricResults(g_names, metrics_dir, load_type, epoch, override=True, inc_partials=True):
    
    perf_cols = np.load(metrics_dir + '/' + 'perf_cols.npy') 
    space_types = np.load(metrics_dir + '/' + 'space_types.npy') 

    c_files_exist = all([os.path.exists(metrics_dir + '/' + 'metrics_combined_' + c + '_' +  load_type + '_' + str(epoch) + '.csv') for c in space_types])

    if((not c_files_exist) or override):
        comb_ar = [[] for i in range(len(space_types))]
        for i in range(len(g_names)):
            metrics_file = metrics_dir + '/' + 'metrics_index_' + str(i) + '_' + load_type + '_' + str(epoch) + '.npy'
            metrics_file = metrics_file if(os.path.exists(metrics_file)) else metrics_file.replace('metrics_','partialmetrics_') if(os.path.exists(metrics_file.replace('metrics_','partialmetrics_')) and inc_partials) else None
            if(metrics_file is not None):
                m_vals = np.load(metrics_file, allow_pickle=True)
                [comb_ar[j].append(np.append(m_vals[j],g_names[i])) for j in range(len(space_types))]
        comb_ar = np.array(comb_ar)
    else:
        comb_ar = []

    if(len(comb_ar)>0):
        for j in range(len(space_types)):
            comb_ar_v = pd.DataFrame(comb_ar[j])
            comb_ar_v.columns =  np.append(np.append(perf_cols, 'sp_type'),'item_n')
            comb_ar_v[perf_cols] = comb_ar_v[perf_cols].astype(float)
            comb_ar_v.to_csv(metrics_dir + '/' + 'metrics_combined_' + space_types[j] + '_' +  load_type + '_' + str(epoch) + '.csv', index=None)
            
def loadCombinedPerfMetrics(metrics_dir, load_type, epoch, space_type='srf'):
    comb_file = metrics_dir + '/' + 'metrics_combined_' + space_type + '_' +  load_type + '_' + str(epoch) + '.csv'
    if(os.path.exists(comb_file)):
        comb_metrics = pd.read_csv(comb_file)
    else:
        print('The metrics file does not exist: ' + comb_file)
        comb_metrics = None
    return comb_metrics
        
# def combineMetricResults(xyz_sdf_perf_dataset, metrics_dir, load_type, epoch, override=False):
#     if((not os.path.exists(metrics_dir + '/' + 'metrics_combined_' + load_type + '_' + str(epoch) + '.json')) or override):
#         # Create a combined metrics results file
#         metric_types = np.load(metrics_dir + '/' + 'perf_cols.npy')
#         print('metric_types',metric_types)
#         p_sets = np.load( metrics_dir + '/' + 'point_set_names.npy')
#         perf_metrics = [p for p in ['perf_l1', 'perf_l2', 'perf_cS', 'perf_kl'] if(p in metric_types)]
#         print('perf_metrics',perf_metrics)
#         metric_idx = [list(metric_types).index(l) for l in perf_metrics]
#         print('metric_idx',metric_idx)
#         met_cols = ['idx','g_name'] + [y for x in [[p + '_' + t for t in perf_metrics] for p in p_sets] for y in x]
#         print('met_cols',met_cols)
#         metrics_all = []
#         for i in range(len(xyz_sdf_perf_dataset)):
#             model_input, perf = xyz_sdf_perf_dataset.__getitemIncNone__(i, include_sdf = False)
#             g_name = model_input['item_n'][0]
#             if(os.path.exists(metrics_dir + '/' + 'metrics_index_' + str(i) + '_name_' + g_name  + '_' + load_type + '_' + str(epoch) + '.npy')):
#                 print('f_shape',np.load(metrics_dir + '/' + 'metrics_index_' + str(i) + '_name_' + g_name  + '_' + load_type + '_' + str(epoch) + '.npy').shape)
#                 m_loaded = np.load(metrics_dir + '/' + 'metrics_index_' + str(i) + '_name_' + g_name  + '_' + load_type + '_' + str(epoch) + '.npy')[:,metric_idx].flatten()
#                 m_vals = [i,g_name] + list(m_loaded)
#                 metrics_all.append(list(m_vals))
#                 #print('Metric file for geometry of index '+str(i) + ' appended to list.')
#             else:
#                 print('Metric file for geometry of index '+str(i) + ' does not exist.')

#         metrics_all = np.array(metrics_all).T
#         comb_metrics =  {}
#         for i in range(len(met_cols)):
#             comb_metrics[met_cols[i]] = metrics_all[i].astype(int) if(i == 0) else metrics_all[i].astype(str) if(i == 1) else metrics_all[i].astype(float)
#         with open(metrics_dir + '/' + 'metrics_combined_' + load_type + '_' + str(epoch) + '.json', 'w') as f:
#             json.dump(comb_metrics, f, indent=2, cls=NumpyArrayEncoder)        

#         print('All metrics calculations combined for the entire set.')
#     else:
#         print('The combined metrics file already exists: ' + metrics_dir + '/' + 'metrics_combined_' + load_type + '_' + str(epoch) + '.json')
    
# def loadCombinedPerfMetrics(metrics_dir, load_type, model_type, epoch):
#     metrics_dir = metrics_dir if(model_type=='train') else metrics_dir.replace('metrics',model_type+'/metrics')
#     if(os.path.exists(metrics_dir + '/' + 'metrics_combined_' + load_type + '_' + str(epoch) + '.json')):
#         comb_metrics = json.loads(open(metrics_dir + '/' + 'metrics_combined_' + load_type + '_' + str(epoch) + '.json').read())
#     else:
#         print('The metrics file does not exist: ' + metrics_dir + '/' + 'metrics_combined_' + load_type + '_' + str(epoch) + '.json')
#         comb_metrics = None
#     return comb_metrics


def getChamferDist(mesh_1,mesh_2,num_samples=30000):
    mesh_1_pc = trimesh.sample.sample_surface(mesh_1, num_samples)[0]
    mesh_2_pc = trimesh.sample.sample_surface(mesh_2, num_samples)[0]
    mesh_1_kd_tree = KDTree(mesh_1_pc)
    mesh_1_distances, mesh_1_vertex_ids = mesh_1_kd_tree.query(mesh_2_pc)
    mesh_1_to_mesh_2_chamfer = np.mean(np.square(mesh_1_distances))
    mesh_2_kd_tree = KDTree(mesh_2_pc)
    mesh_2_distances, mesh_2_vertex_ids = mesh_2_kd_tree.query(mesh_1_pc)
    mesh_2_to_mesh_1_chamfer = np.mean(np.square(mesh_2_distances))
    chamfer_dist = mesh_1_to_mesh_2_chamfer + mesh_2_to_mesh_1_chamfer
    return chamfer_dist

def getWassersteinDist(mesh_1,mesh_2,num_samples=500):
    mesh_1_pc = trimesh.sample.sample_surface(mesh_1, num_samples)[0]
    mesh_2_pc = trimesh.sample.sample_surface(mesh_2, num_samples)[0]
    sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
    wasserstein_dist = sinkhorn_loss(torch.tensor(mesh_1_pc).unsqueeze(0), torch.tensor(mesh_2_pc).unsqueeze(0)).detach().numpy()[0]
    return wasserstein_dist


def predictPerf_custIn(perf_model, xyz, latent):
    
    if(perf_model.apply_ctxt):
        ctxt_latent = perf_model.ctxtmod_autoenc(latent)

    g_latents_sub = torch.unsqueeze(latent, 1)
    g_latents_sub = torch.tile(g_latents_sub, (1, xyz.shape[1], 1))
    if(perf_model.apply_ctxt):
        ctxt_latent_t = torch.unsqueeze(ctxt_latent, 1)
        ctxt_latent_t = torch.tile(ctxt_latent_t, (1, xyz.shape[1], 1))

    dec_input = torch.cat([g_latents_sub, ctxt_latent_t, xyz], -1).float() if(perf_model.apply_ctxt) else torch.cat([g_latents_sub, xyz], -1).float()

    outputs = perf_model.perf_dec(dec_input)
    
    return outputs

def getGTMeshfromUnName(item_n, geo_directory='/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF/Geometry', dataset_type='01_Buildings', bbox_name = 'bboxParam_500_10'):
    geo_name = '_'.join(item_n.split('_')[0:2])
    geo_id = int(item_n.split('_')[1])
    geo_rot = int(item_n.split('_')[2])
    idx_s = int(geo_id/500)
    set_fromIDX = "{:04d}".format(idx_s*500) + '_' + "{:04d}".format(((idx_s+1)*500)-1)
    g_dir = geo_directory + '/'+ dataset_type + '/' + set_fromIDX
    if((os.path.exists(g_dir + '/' + geo_name + '.obj')) and (geo_rot%45 == 0) and (geo_rot<360)):
        mesh_f = trimesh.load(g_dir + '/' + geo_name + '.obj')
        buffered_bbox = np.array(json.loads(open(geo_directory.replace('Geometry','Parameters') + '/' + bbox_name+'.json').read())['bbox'])
        norm_mesh = normalize_mesh(mesh_f, buffered_bbox)
        norm_mesh_vertices_rot = norm_mesh.vertices + [-0.5,-0.5,-0.5]
        norm_mesh_vertices_rot = np.concatenate([norm_mesh_vertices_rot,np.ones((norm_mesh_vertices_rot.shape[0],1))],1)
        norm_mesh_vertices_rot = np.dot(rotation_matrix_y(int(geo_rot)),norm_mesh_vertices_rot.T).T[:,:3].round(6)
        norm_mesh_vertices_rot = norm_mesh_vertices_rot + [0.5,0.5,0.5]
        norm_mesh = trimesh.Trimesh(vertices=norm_mesh_vertices_rot, faces=norm_mesh.faces)
    else:
        norm_mesh = None
    return norm_mesh



