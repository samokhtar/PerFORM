import numpy as np
import os
import torch
import pandas as pd
import cv2
import pyvista as pv

import sys
sys.path.insert(0, '/home/gridsan/smokhtar/building_urban_sdf_project')

from building_sdf.viz_utils import *
from building_sdf.sampling import *
from building_sdf.data_prep import *
from building_sdf.learning import global_cond, hybrid_cond
from building_sdf.learning.network_loss import *
from building_sdf.learning.network_modules import *
from building_sdf.learning.network_training import *
from building_sdf.learning.network_utils import *
from building_sdf.learning.network_viz import *
from sklearn.metrics.pairwise import pairwise_distances, paired_distances

import torch
print(f"Installed Torch version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

print(device)






def getAvTimeprEpoch(m_dir, n_ext='model'):
    if((os.path.exists(m_dir + '/checkpoints')) and (len([o for o in os.listdir(m_dir + '/checkpoints') if(o.startswith(n_ext+'_epoch_'))])>0)):
        ord_list = np.sort([int(o.split('.')[0].split('_')[-1]) for o in os.listdir(m_dir + '/checkpoints') if(o.startswith(n_ext+'_epoch_'))])
        file_names_list = np.array([n_ext+'_epoch_' + "{:04d}".format(n) + '.pth' for n in ord_list])
        e_diff = (ord_list-np.roll(ord_list,1))[1:]
        t_ep = np.mean([(os.path.getmtime(m_dir + '/checkpoints' + '/' + file_names_list[i+1])-os.path.getmtime(m_dir + '/checkpoints' + '/' + np.roll(file_names_list, 1)[i+1]))/(e_diff[i]) for i in range(len(file_names_list)-1)])
    else:
        t_ep = 0
    return t_ep

def getGeoModelParam(m_dir, grid_type='global', model_type='train', include_val=True, include_rot=True):
    val_add = 'val' if(include_val) else ''
    rot_add = 'rot' if(include_rot) else ''
    d_name = 'dataset_train'+val_add+rot_add+'_param' if(model_type == 'train') else 'dataset_test_param'
    if(grid_type=='global'):
        mlp_param, dataset_param = loadModelParam(m_dir, param_ar = ['mlp_param', d_name])
        max_instances = dataset_param['max_num_instances'] if(dataset_param['max_num_instances'] != -1) else dataset_param['total_instances']
        model = global_cond.LatentNeuralField(num_latents=max_instances, mlp_param = mlp_param).to(device)
        sub_models = [model]
    else:
        grid_param, field_param, dataset_param = loadModelParam(m_dir, param_ar = ['grid_param', 'field_param', d_name])
        max_instances = dataset_param['max_num_instances'] if(dataset_param['max_num_instances'] != -1) else dataset_param['total_instances']
        model_grid = hybrid_cond.LatenttoHybridGroundLatentGrid_ConvDecoder(grid_param=grid_param).to(device) if(grid_param['grid_type'] == 'ground') else hybrid_cond.LatenttoHybridVoxelLatentGrid_ConvDecoder(grid_param=grid_param).to(device)
        grid_shape = (1, grid_param['feature_dim'], grid_param['grid_size'], grid_param['grid_size']) if(grid_param['grid_type'] == 'ground') else (1, grid_param['feature_dim'], grid_param['grid_size'], grid_param['grid_size'], grid_param['grid_size'])
        model_grid_ad = AutoDecoderWrapper(max_instances, model_grid, param_name='grid.latent', in_wgt=grid_param['latent_init']).to(device)
        model_field = hybrid_cond.LatentField(grid_param=grid_param, field_param=field_param).to(device)
        sub_models = [model_grid_ad, model_field]
    p_par = [int(getModelParametersCount(p)) for p in sub_models]
    return p_par

def checkModelUpdates(model_category, model_name, n_ext='model', model_type='train', experiments_dir='/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF/Experiments/01_Buildings'):
    m_dir = experiments_dir + '/' + model_category + '/' + model_name 
    r_dir = m_dir if(model_type == 'train') else m_dir + '/' + model_type
    #print('m_dir',m_dir)
    grid_type = model_category.split('_')[0]
    n_ext = 'model' if(grid_type == 'global') else 'model_grid'
    if(os.path.exists(r_dir + '/' + 'checkpoints')):
        m_stat = 1 if(os.path.exists(r_dir + '/'+n_ext+'_final.pth')) else np.sort([int(o.split('.')[0].split('_')[-1]) for o in os.listdir(r_dir + '/checkpoints/') if((''+n_ext+'_epoch_') in o)])[::-1][0] if(len([o for o in os.listdir(r_dir + '/checkpoints/') if((''+n_ext+'_epoch_') in o)])) else 0
        #print('m_stat',m_stat)
    else:
        m_stat = 0
    #print('grid_type',grid_type)
    av_time_epoch = getAvTimeprEpoch(r_dir, n_ext=n_ext)
    #print('av_time_epoch',av_time_epoch)
    m_par = getGeoModelParam(m_dir, grid_type=grid_type, model_type=model_type)
    #print('m_par',m_par)
    return m_stat, av_time_epoch, m_par

def checkPPUpdates(model_category, model_name, load_type, epoch = None, mc_res = 128, n_ext='model', model_type='train', include_val=True, include_rot=True, experiments_dir='/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF/Experiments/01_Buildings', pairs_type='all'):
    m_dir = experiments_dir + '/' + model_category + '/' + model_name
    #print('m_dir',m_dir)
    r_dir = m_dir if(model_type == 'train') else m_dir + '/' + model_type
    #print('r_dir',r_dir)
    val_add = 'val' if(include_val) else ''
    rot_add = 'rot' if(include_rot) else ''
    d_name = 'dataset_train'+val_add+rot_add+'_param' if(model_type == 'train') else 'dataset_test_param'
    #print('d_name',d_name)
    dataset_param = loadModelParam(m_dir, param_ar = [d_name])[0]
    #print('dataset_param',dataset_param)
    b_exp = '240612_full5000_incRot_incVal_lipNormUpd'
    latent_par_dir = '/'.join(m_dir.split('/')[0:-2]) + '/global_building/' + b_exp + '/' + 'latent_param' if('test' not in model_type) else '/'.join(m_dir.split('/')[0:-2]) + '/global_building/' + b_exp + '/'+ model_type +'/' + 'latent_param'
    l_pairs =  len(np.load(latent_par_dir + '/' + 'sel_pairs_all_500B.npy')) if(pairs_type!='all') else len(np.load(latent_par_dir + '/' + 'sel_pairs_all.npy'))
    rec_stat = 0 if((not os.path.exists(r_dir + '/reconstructions')) or (len([o for o in os.listdir(r_dir + '/reconstructions') if(o.endswith('_'+load_type+'_'+str(epoch)+'.obj'))])==0)) else 1 if(os.path.exists(r_dir + '/reconstructions/reconstruction_'+str(mc_res)+'_index_'+str(dataset_param['max_num_instances']-1)+'_'+load_type+'_'+str(epoch)+'.obj')) else len([o for o in os.listdir(r_dir + '/reconstructions') if(o.endswith('_'+load_type+'_'+str(epoch)+'.obj'))])
    met_stat = 0 if((not os.path.exists(r_dir + '/metrics')) or (len([o for o in os.listdir(r_dir + '/metrics') if(o.endswith('_'+load_type+'_'+str(epoch)+'.npy'))])==0)) else 1 if(os.path.exists(r_dir + '/metrics/metrics_'+str(mc_res)+'_index_'+str(dataset_param['max_num_instances']-1)+'_'+load_type+'_'+str(epoch)+'.npy')) else len([o for o in os.listdir(r_dir + '/metrics') if(o.endswith('_'+load_type+'_'+str(epoch)+'.npy'))])
    if(model_type == 'test'):
        int_stat = 0 if((not os.path.exists(r_dir + '/interpolations')) or (len([o for o in os.listdir(r_dir + '/interpolations') if(o.endswith('_'+load_type+'.obj'))])==0)) else 1 if(len([o for o in os.listdir(r_dir + '/interpolations') if(o.endswith('_'+load_type+'.obj'))])==(l_pairs*6)) else (len([o for o in os.listdir(r_dir + '/interpolations') if(o.endswith('_'+load_type+'.obj'))])/6)
    else:
        int_stat = 0 if((not os.path.exists(r_dir + '/interpolations')) or (len([o for o in os.listdir(r_dir + '/interpolations') if(o.endswith('_'+load_type+'_'+str(epoch)+'.obj'))])==0)) else 1 if(len([o for o in os.listdir(r_dir + '/interpolations') if(o.endswith('_'+load_type+'_'+str(epoch)+'.obj'))])==(l_pairs*6)) else (len([o for o in os.listdir(r_dir + '/interpolations') if(o.endswith('_'+load_type+'_'+str(epoch)+'.obj'))])/6)
    return rec_stat, met_stat, int_stat

def loadModelLatbyTyp(model_category, model_name, load_type, epoch = None, model_type='train', include_val=True, include_rot=True, experiments_dir='/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF/Experiments/01_Buildings'):
    
    m_dir = experiments_dir + '/' + model_category + '/' + model_name 
    grid_type = model_category.split('_')[0]
    
    if(grid_type == 'global'):
        model = global_cond.loadModel(m_dir, model_type = model_type, load_type = load_type, epoch = epoch, device = device.type, include_val=include_val, include_rot=include_rot)
        model.eval()
        models = [model]
    else:
        model_grid, model_field = hybrid_cond.loadModels(m_dir, model_type = model_type, load_type = load_type, epoch = epoch, device = device.type, include_val=include_val, include_rot=include_rot)
        model_grid.eval()
        model_field.eval()
        models = [model_grid, model_field]
    
    model_latents = models[0].module.latents.weight.detach().cpu().numpy() if(isinstance(models[0], nn.DataParallel)) else models[0].latents.weight.detach().cpu().numpy()
    num_latents = model_latents.shape[0]
    
    return models, model_latents, num_latents

def getDatasetIDs(model_category, model_name, model_type='train', include_val=True, include_rot=True, experiments_dir='/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF/Experiments/01_Buildings'):
    m_dir = experiments_dir + '/' + model_category + '/' + model_name 
    val_add = 'val' if(include_val) else ''
    rot_add = 'rot' if(include_rot) else ''
    d_name = 'dataset_train'+val_add+rot_add+'_param' if(model_type == 'train') else 'dataset_test_param'
    dataset_param = loadModelParam(m_dir, param_ar = [d_name])[0]
    dataset_ids = np.array(dataset_param['dataset_ids'])
    return dataset_ids

def getLatentIntPairs(model_category, model_name, load_type, epoch = None, model_type='train', include_val=True, include_rot=True, experiments_dir='/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF/Experiments/01_Buildings', pairs_type='all'):

    dataset_ids = getDatasetIDs(model_category, model_name, model_type=model_type, include_val=include_val, include_rot=include_rot)
    models, model_latents, num_latents = loadModelLatbyTyp(model_category, model_name, load_type, epoch, model_type, include_val=include_val, include_rot=include_rot)
    
    m_dir = experiments_dir + '/' + model_category + '/' + model_name
    latent_par_dir = experiments_dir + '/' + 'global_building/240612_full5000_incRot_incVal_lipNormUpd/latent_param' if('test' not in model_type) else experiments_dir + '/' + 'global_building/240612_full5000_incRot_incVal_lipNormUpd/test/latent_param'
    sel_pairs = {'all': np.load(latent_par_dir + '/' + 'sel_pairs_all.npy'), 'cond':np.load(latent_par_dir + '/' + 'sel_pairs_all_500B.npy')}
    cur_pairs = sel_pairs[pairs_type]
    
    b_1_all = ['_'.join(b.split('_')[:3]) for b in cur_pairs]
    b_2_all = ['_'.join(b.split('_')[3:]) for b in cur_pairs]
    b_idx_1_all = np.array([list(dataset_ids).index(b_1_all[i]) for i in range(len(b_1_all))])
    b_idx_2_all = np.array([list(dataset_ids).index(b_2_all[i]) for i in range(len(b_2_all))])
    latents_1_all = model_latents[b_idx_1_all]
    latents_2_all = model_latents[b_idx_2_all]
    euc_distances = paired_distances(latents_1_all, latents_2_all, metric='euclidean')
    
    return b_1_all, b_2_all, b_idx_1_all, b_idx_2_all, latents_1_all, latents_2_all, euc_distances

