# Visualization libraries
import numpy as np
from plotly.subplots import make_subplots
import plotly
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import ipywidgets as widgets
from mpl_toolkits.axes_grid1 import ImageGrid
from IPython.display import display, clear_output
import seaborn as sns
# Pytorch
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
# Other
import mcubes
import trimesh
from building_sdf.sampling import *
from building_sdf.viz_utils import *

# Plot losses
def plotLosses(losses, loss_type='Loss', x_range=None):
    plt.figure(figsize=(30,6))
    if(len(losses) == 1):
        plt.plot(losses.flatten(), label=loss_type)
        if(x_range is not None):
            plt.xlim(x_range)
        plt.xlabel('Epochs')
        plt.ylabel(loss_type)
        plt.title(loss_type + ' Evolution')
        plt.legend()
        plt.show()
    else:
        for i in range(len(losses)):
            plt.plot(losses[i].flatten(), label=loss_type)
            if(x_range is not None):
                plt.xlim(x_range)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Evolution')
        plt.legend()
        plt.show()

# Plot detailed losses
def plotDetailedLosses(losses_df_byepoch, loss_lambdas, weighted = True, y_range=None, x_range=None, colors=None):

    plt.figure(figsize=(30,6))
    cols = losses_df_byepoch.columns
    epochs_x = np.array(losses_df_byepoch.index)

    for i in range(len(cols)):
        if(weighted):
            if(colors is not None):
                plt.plot(epochs_x, losses_df_byepoch[cols[i]]*loss_lambdas[i], label=cols[i], color=colors[i])
            else:
                plt.plot(epochs_x, losses_df_byepoch[cols[i]]*loss_lambdas[i], label=cols[i])
            l_type = 'weighted'   
        else:
            if(colors is not None):
                plt.plot(epochs_x, losses_df_byepoch[cols[i]], label=cols[i], color=colors[i])
            else:
                plt.plot(epochs_x, losses_df_byepoch[cols[i]], label=cols[i])
            l_type = 'absolute'

    plt.xlabel('Epochs')
    #plt.xticks(epochs_x)
    plt.ylabel('Loss')
    if(y_range is not None):
        plt.ylim(y_range)
    if(x_range is not None):
        plt.xlim(x_range)
    plt.title('Losses across epochs - ' + l_type)
    plt.legend()
    plt.show()
    
# Plot detailed losses
def plotMtpLosses(loss_values, loss_names, y_range=None, x_range=None, colors=None, title = 'Losses across epochs - comparative', legend_right = False, file_path = None, figsize=(30,6), legend_loc='upper left', legend_bbox_to_anchor=(1.01, 1.0)):

    plt.figure(figsize=figsize)
    epochs_x = np.array(loss_values[0])

    for i in range(len(loss_values)):
        if(colors is not None):
            plt.plot(np.arange(len(loss_values[i])), loss_values[i], label=loss_names[i], color = colors[i])
        else:
            plt.plot(np.arange(len(loss_values[i])), loss_values[i], label=loss_names[i])
    plt.xlabel('Epochs')
    #plt.xticks(epochs_x)
    plt.ylabel('Loss')
    if(y_range is not None):
        plt.ylim(y_range)
    if(x_range is not None):
        plt.xlim(x_range)
    plt.title(title)
    if(legend_right):
        plt.legend(bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc)
    else:
        plt.legend()
    if(file_path is not None):
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() 
    clear_output(wait = True)
    plt.pause(0.5)
    
# Plot reconstructions while training
def plotRandReconstruction(resolution, XYZ_coord, models, index, act_index, viz_dir, step, buffered_bbox, rendering_dir, typeP = 'Train'):
    
    reconstruct_mesh = reconstructMCMesh(resolution, XYZ_coord, models, extents = None, min_v = None, index=index)
    idx_s = int(act_index/500)
    set_fromIDX = "{:04d}".format(idx_s*500) + '_' + "{:04d}".format(((idx_s+1)*500)-1)
    
    if(reconstruct_mesh is not None):
        print('Mesh was reconstructed.')
        reconstruct_mesh = unnormalize_mesh(reconstruct_mesh, buffered_bbox)
        renderGeo(reconstruct_mesh, viz_dir + '/building_' + str(act_index) + '_45' + '_' + str(step) + '.png', azim_angle = 45, ptLightLoc = [[100, 100, 100]], show = False)
        print('Mesh reconstruction was rendered.')
        if(os.path.exists(rendering_dir + '/' + set_fromIDX +  '/single/building_' + str(act_index) + '_45.png')):
            print('Rendered ground truth image exists.')
            images = np.array([viz_dir + '/building_' + str(act_index) + '_45' + '_' + str(step) + '.png', rendering_dir + '/' + set_fromIDX + '/single/building_' + str(act_index) + '_45.png'])
            im_titles = np.array(['predicted_' + str(index), 'groundtruth_' + str(index)])
            print(typeP + " sample:")
            plotImageMatrix(images, im_titles, fig_title = '', text_size = 6, title_size = 12, col_n = 2, save_dir = None, show_titles = True, plot_size = 5)
        else:
            print('Rendered ground truth image does not exists in location: ' + rendering_dir + '/' + set_fromIDX +  '/single/building_' + str(act_index) + '_45.png')
    else:
        print('Mesh could not be reconstructed - 0-vertices.')
        fig = plt.figure(figsize=(18,18))
        plt.savefig(viz_dir + '/building_' + str(act_index) + '_45' + '_' + str(step) + '.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)


    # Use marching cubes to reconstruct SDF surface from model sdf prediction
# def reconstructMCMesh(resolution, XYZ_coord, model, index=None):
#     outputs = model.to("cpu")({'xyz': XYZ_coord,'idx': torch.tensor([index])})
#     if(len(outputs) == 3):
#         SDF_predicted, perf_predicted, latent = outputs
#     else:
#         SDF_predicted, latent = outputs
#     #SDF_predicted_cl = torch.clamp(SDF_predicted, min=torch.tensor(-0.99999999), max=torch.tensor(0.99999999))
#     SDF_predicted = SDF_predicted.detach().cpu().numpy().reshape((resolution, resolution, resolution))
#     verts, faces = mcubes.marching_cubes(SDF_predicted, 0)
#     verts = verts/resolution
#     # verts, faces, normals, values = skimage.measure.marching_cubes(
#     #         SDF_predicted_cl, level=0.0, spacing=[resolution] * 3
#     #     )
#     if(len(verts) != 0):
#         reconstructed_mesh = trimesh.Trimesh(verts, faces)
#     else:
#         reconstructed_mesh = None
#         print('Mesh for geometry of index ' + str(index) + ' could not be reconstructed - 0-vertices.')
#     return reconstructed_mesh

# Use marching cubes to reconstruct SDF surface from model sdf prediction
def reconstructMCMeshJoint(resolution, XYZ_coord, model_sdf, model_grid, index=None, smooth=True):
    grid_predicted, latent = model_grid.to("cpu")({'xyz': XYZ_coord,'idx': torch.tensor([index])})
    SDF_predicted = model_sdf.to("cpu")({'xyz': XYZ_coord,'idx': torch.tensor([index])},grid_predicted)
    SDF_predicted_cl = torch.clamp(SDF_predicted, min=torch.tensor(-0.99999999), max=torch.tensor(0.99999999))
    SDF_predicted_cl = SDF_predicted_cl.detach().cpu().numpy().reshape((resolution, resolution, resolution))
    if(smooth):
        smoothed_SDF = mcubes.smooth(SDF_predicted)
        verts, faces = mcubes.marching_cubes(smoothed_SDF, 0)
    else:
        verts, faces = mcubes.marching_cubes(SDF_predicted_cl, 0)
    verts = verts/resolution
    # verts, faces, normals, values = skimage.measure.marching_cubes(
    #         SDF_predicted_cl, level=0.0, spacing=[resolution] * 3
    #     )
    if(len(verts) != 0):
        reconstructed_mesh = trimesh.Trimesh(verts, faces)
    else:
        reconstructed_mesh = None
        print('Mesh for geometry of index ' + str(index) + ' could not be reconstructed - 0-vertices.')
    print(reconstructed_mesh)
    return reconstructed_mesh

# Get 3D grid coordinates
def get3DGrid(resolution):
    grid = np.array([[x, y, z] for x in np.linspace(0, 1, resolution) for y in np.linspace(0, 1, resolution) for z in np.linspace(0, 1, resolution)])
    XYZ_coord = torch.tensor(np.expand_dims(grid, axis=0)).float()
    return XYZ_coord

# Normalize and unnormalize meshes
def normalize_mesh_calcs(mesh, bbox):
    norm_vertices = (mesh.vertices)/(bbox[:,1]-bbox[:,0])
    norm_mesh = trimesh.Trimesh(vertices=norm_vertices, faces=mesh.faces, process=False)
    return norm_mesh

def normalize_mesh_calcs(mesh, bbox):
    unnorm_vertices = (mesh.vertices*(bbox[:,1]-bbox[:,0]))
    unnorm_mesh = trimesh.Trimesh(vertices=unnorm_vertices, faces=mesh.faces, process=False)
    return unnorm_mesh

# Normalize training set meshes by bounding box dimensions
def normalize_mesh(mesh, bbox):
    norm_vertices = (mesh.vertices-bbox[:,0])/(bbox[:,1]-bbox[:,0])
    norm_mesh = trimesh.Trimesh(vertices=norm_vertices, faces=mesh.faces)
    return norm_mesh

# Normalize training set meshes by bounding box dimensions
def unnormalize_mesh(mesh_norm, bbox):
    unnorm_vertices = (mesh_norm.vertices*(bbox[:,1]-bbox[:,0]))+bbox[:,0]
    unnorm_mesh = trimesh.Trimesh(vertices=unnorm_vertices, faces=mesh_norm.faces)
    return unnorm_mesh

# Normalize training set meshes by bounding box dimensions
def normalize_mesh_consistent(mesh, bbox):
    norm_vertices = (mesh.vertices-bbox[:,0])/(bbox[:,1]-bbox[:,0])
    norm_mesh = trimesh.Trimesh(vertices=norm_vertices, faces=mesh.faces, process=False)
    return norm_mesh

# Normalize training set meshes by bounding box dimensions
def unnormalize_mesh_consistent(mesh_norm, bbox):
    unnorm_vertices = (mesh_norm.vertices*(bbox[:,1]-bbox[:,0]))+bbox[:,0]
    unnorm_mesh = trimesh.Trimesh(vertices=unnorm_vertices, faces=mesh_norm.faces, process=False)
    return unnorm_mesh

# Normalize and unnormalize points
def normalize_points(points, bbox):
    norm_points = (points-bbox[:,0])/(bbox[:,1]-bbox[:,0])
    return norm_points

def unnormalize_points(points, bbox):
    unnorm_points = ((points*(bbox[:,1]-bbox[:,0])))+bbox[:,0]
    return unnorm_points

def generateLatentInterpolations(interpolation_dir, latents, sel_1, sel_2, sub_divisions, ltype_ext=None):
    interp_n = np.linspace(0,1,sub_divisions)
    f_ext = '' if(ltype_ext is None) else ('_' + ltype_ext)
    interpolations, interpNames = [], []

    for j in range(len(interp_n)):
        l_vec_1 = latents[sel_1]
        l_vec_2 = latents[sel_2]
        intP = l_vec_1*(1-interp_n[j]) + l_vec_2*(interp_n[j])
        l_interp = 'latent_interp_' + str(sel_1) + '_' + str(sel_2) + '_' + str(int(interp_n[j]*100))
        interpNames.append(l_interp)
        np.savez(interpolation_dir + '/' + l_interp + f_ext + '.npz', lVec = intP)
        interpolations.append(intP)
    
    return torch.nn.Parameter(torch.tensor(np.array(interpolations)).clone().detach()), np.array(interpNames)

def generate2DLatentInterpolations(interpolation_dir, latents, sel_1, sel_2, sel_3, sel_4, sub_divisions, ltype_ext=None):
    interp_n = np.linspace(0,1,sub_divisions)
    f_ext = '' if(ltype_ext is None) else ('_' + ltype_ext)
    interpolations, interpNames = [], []

    l_vec_1 = latents[sel_1]
    l_vec_2 = latents[sel_2]
    l_vec_3 = latents[sel_3]
    l_vec_4 = latents[sel_4]

    interpolated_grid = np.zeros((sub_divisions, sub_divisions, latents.shape[1]))
    
    # Perform bilinear interpolation over the grid
    for i, a in enumerate(interp_n):
        for j, b in enumerate(interp_n):
            interpolated_grid[i, j] = (
                (1 - a) * (1 - b) * l_vec_1 +
                a * (1 - b) * l_vec_2 +
                (1 - a) * b * l_vec_3 +
                a * b * l_vec_4
            )
    
    all_interp_ar = interpolated_grid.reshape(-1,latents.shape[1])

    for j in range(len(all_interp_ar)):
        intP = all_interp_ar[j]
        l_interp = 'latent_interp_' + str(sel_1) + '_' + str(sel_2) + '_' + str(sel_3) + '_' + str(sel_4) + '_' + str(j)
        interpNames.append(l_interp)
        np.savez(interpolation_dir + '/' + l_interp + f_ext + '_' + str(sub_divisions) + '.npz', lVec = intP)
        interpolations.append(intP)
        
    return torch.nn.Parameter(torch.tensor(np.array(interpolations)).clone().detach()), np.array(interpNames)

# # Use marching cubes to reconstruct SDF surface from model sdf prediction
# def reconstructMCMesh(resolution, XYZ_coord, model, index=None):
#     outputs = model.to("cpu")({'xyz': XYZ_coord,'idx': torch.tensor([index])})
#     if(len(outputs) == 3):
#         SDF_predicted, perf_predicted, latent = outputs
#     else:
#         SDF_predicted, latent = outputs
        
#     #SDF_predicted_cl = torch.clamp(SDF_predicted, min=torch.tensor(-0.99999999), max=torch.tensor(0.99999999))
#     SDF_predicted = SDF_predicted.detach().cpu().numpy().reshape((resolution, resolution, resolution))
#     verts, faces = mcubes.marching_cubes(SDF_predicted, 0)
#     verts = verts/resolution
#     # verts, faces, normals, values = skimage.measure.marching_cubes(
#     #         SDF_predicted_cl, level=0.0, spacing=[resolution] * 3
#     #     )
#     if(len(verts) != 0):
#         reconstructed_mesh = trimesh.Trimesh(verts, faces)
#     else:
#         reconstructed_mesh = None
#         print('Mesh for geometry of index ' + str(index) + ' could not be reconstructed - 0-vertices.')
#     return reconstructed_mesh

# def getLowResMCBounds(resolution, XYZ_coord, models, index=None):
#     if(len(models)==1):
#         outputs = models[0]({'xyz': XYZ_coord.to(device),'idx': torch.tensor([index]).to(device)})
#         if(len(outputs) == 3):
#             SDF_predicted, perf_predicted, latent = outputs
#         else:
#             SDF_predicted, latent = outputs
#     else:
#         grid_output, param_latent = models[0]({'xyz': XYZ_coord.to(device),'idx': torch.tensor([index]).to(device)})
#         outputs = models[1]({'xyz': XYZ_coord.to(device),'idx': torch.tensor([index]).to(device)}, grid_output)
#         if(len(outputs) == 2):
#             SDF_predicted, perf_predicted = outputs
#         else:
#             SDF_predicted = outputs
    
#     XYZ_coord_to_use = get3DGrid(resolution)[0]
#     outputs_pos = XYZ_coord_to_use[(SDF_predicted.detach().cpu()<0).flatten().detach().numpy()].detach().numpy()
#     if(outputs_pos.shape[0]>0):
#         buffer = (1/resolution)*2
#         extents = np.max(outputs_pos, axis=0) + buffer - (np.min(outputs_pos, axis=0)-buffer)
#         min_v = (np.min(outputs_pos, axis=0)-buffer)
#         return extents, min_v
#     else:
#         return None, None 


# # Use marching cubes to reconstruct SDF surface from model sdf prediction
# def reconstructMCMesh(resolution, XYZ_coord, models, extents = None, min_v = None, index=None):
#     if(XYZ_coord.min()<0):
#         XYZ_coord_cur = ((((((XYZ_coord+1)/2)*extents)+min_v)*2)-1).to(device) if((extents is not None) and (min_v is not None)) else XYZ_coord
#     else:
#         XYZ_coord_cur = ((XYZ_coord*extents)+min_v).to(device) if((extents is not None) and (min_v is not None)) else XYZ_coord
#     if(len(models)==1):
#         outputs = models[0]({'xyz': XYZ_coord_cur,'idx': torch.tensor([index]).to(device)})
#         if(len(outputs) == 3):
#             SDF_predicted, perf_predicted, latent = outputs
#         else:
#             SDF_predicted, latent = outputs
#     else:
#         grid_output, param_latent = models[0]({'xyz': XYZ_coord_cur,'idx': torch.tensor([index]).to(device)})
#         outputs = models[1]({'xyz': XYZ_coord_cur,'idx': torch.tensor([index]).to(device)}, grid_output)
#         if(len(outputs) == 2):
#             SDF_predicted, perf_predicted = outputs
#         else:
#             SDF_predicted = outputs

#     SDF_predicted = SDF_predicted.detach().cpu().numpy().reshape((resolution, resolution, resolution))
#     verts, faces = mcubes.marching_cubes(SDF_predicted, 0)
#     verts = verts/resolution
#     if((extents is not None) and (min_v is not None)):
#         verts = verts*extents
#         verts = (verts+min_v)

#     if(len(verts) != 0):
#         reconstructed_mesh = trimesh.Trimesh(verts, faces)
#     else:
#         reconstructed_mesh = None
#         print('Mesh for geometry of index ' + str(index) + ' could not be reconstructed - 0-vertices.')

#     return reconstructed_mesh

def getLowResMCBounds(resolution, XYZ_coord, models, index=None):
    if(len(models)==1):
        outputs = models[0]({'xyz': XYZ_coord.to(device),'idx': torch.tensor([index]).to(device)})
        if(len(outputs) == 3):
            SDF_predicted, perf_predicted, latent = outputs
        else:
            SDF_predicted, latent = outputs
    else:
        grid_output, param_latent = models[0]({'xyz': XYZ_coord.to(device),'idx': torch.tensor([index]).to(device)})
        outputs = models[1]({'xyz': XYZ_coord.to(device),'idx': torch.tensor([index]).to(device)}, grid_output)
        if(len(outputs) == 2):
            SDF_predicted, perf_predicted = outputs
        else:
            SDF_predicted = outputs
    
    XYZ_coord_to_use = get3DGrid(resolution)[0]
    # print('mean:,',(SDF_predicted).mean())
    # print('min:,',(SDF_predicted).min())
    # print('max:,',(SDF_predicted).max())
    outputs_pos = XYZ_coord_to_use[(SDF_predicted.detach().cpu()<0).flatten().detach().numpy()].detach().numpy()
    if(outputs_pos.shape[0]>0):
        buffer = (1/resolution)*2
        extents = np.max(outputs_pos, axis=0) + buffer - (np.min(outputs_pos, axis=0)-buffer)
        min_v = (np.min(outputs_pos, axis=0)-buffer)
        return extents, min_v
    else:
        return None, None 


# Use marching cubes to reconstruct SDF surface from model sdf prediction
def reconstructMCMesh(resolution, XYZ_coord, models, extents = None, min_v = None, index=None, smooth=False):
    if(XYZ_coord.min()<0):
        XYZ_coord_cur = ((((((XYZ_coord+1)/2)*extents)+min_v)*2)-1).to(device) if((extents is not None) and (min_v is not None)) else XYZ_coord
    else:
        XYZ_coord_cur = ((XYZ_coord*extents)+min_v).to(device) if((extents is not None) and (min_v is not None)) else XYZ_coord
    #print(XYZ_coord_cur)
    if(len(models)==1):
        outputs = models[0]({'xyz': XYZ_coord_cur,'idx': torch.tensor([index]).to(device)})
        if(len(outputs) == 3):
            SDF_predicted, perf_predicted, latent = outputs
        else:
            SDF_predicted, latent = outputs
    else:
        grid_output, param_latent = models[0]({'xyz': XYZ_coord_cur,'idx': torch.tensor([index]).to(device)})
        outputs = models[1]({'xyz': XYZ_coord_cur,'idx': torch.tensor([index]).to(device)}, grid_output)
        if(len(outputs) == 2):
            SDF_predicted, perf_predicted = outputs
        else:
            SDF_predicted = outputs

    SDF_predicted = SDF_predicted.detach().cpu().numpy().reshape((resolution, resolution, resolution))
    if(smooth):
        smoothed_SDF = mcubes.smooth(SDF_predicted)
        verts, faces = mcubes.marching_cubes(smoothed_SDF, 0)
    else:
        verts, faces = mcubes.marching_cubes(SDF_predicted, 0)
    # print('mean:,',np.mean(SDF_predicted))
    # print('min:,',np.min(SDF_predicted))
    # print('max:,',np.max(SDF_predicted))
    
    #verts, faces = mcubes.marching_cubes(SDF_predicted, 0)
    verts = verts/resolution
    if((extents is not None) and (min_v is not None)):
        verts = verts*extents
        verts = (verts+min_v)

    if(len(verts) != 0):
        reconstructed_mesh = trimesh.Trimesh(verts, faces)
    else:
        reconstructed_mesh = None
        print('Mesh for geometry of index ' + str(index) + ' could not be reconstructed - 0-vertices.')

    return reconstructed_mesh

def getSDFPred(models,xyz,index):
    return models[0].to(device)({'xyz': torch.tensor([xyz]).to(device),'idx': torch.tensor([index]).to(device)})[0]

# # Use marching cubes to reconstruct SDF surface from model sdf prediction
# def reconstructMCMesh(resolution, XYZ_coord, models, extents, min_v, index=None):
#     if((extents is not None) and (min_v is not None)):
        
#         if(len(models) == 1):
#             outputs = models[0]({'xyz': ((XYZ_coord*extents)+min_v).to(device),'idx': torch.tensor([index]).to(device)})
#             if(len(outputs) == 3):
#                 SDF_predicted, perf_predicted, latent = outputs
#             else:
#                 SDF_predicted, latent = outputs
#         else:
#             model_grid, model_field =  models
#             grid_predicted, latent = model_grid({'xyz': ((XYZ_coord*extents)+min_v).to(device),'idx': torch.tensor([index]).to(device)})
#             SDF_predicted = model_field({'xyz': ((XYZ_coord*extents)+min_v).to(device),'idx': torch.tensor([index]).to(device)},grid_predicted)
#             SDF_predicted = torch.clamp(SDF_predicted, min=torch.tensor(-0.99999999), max=torch.tensor(0.99999999))
        
#         SDF_predicted = SDF_predicted.detach().cpu().numpy().reshape((resolution, resolution, resolution))
#         verts, faces = mcubes.marching_cubes(SDF_predicted, 0)
#         verts = verts/resolution
#         verts = verts*extents
#         verts = (verts+min_v)
        
#         if(len(verts) != 0):
#             reconstructed_mesh = trimesh.Trimesh(verts, faces)
#         else:
#             reconstructed_mesh = None
#             print('Mesh for geometry of index ' + str(index) + ' could not be reconstructed - 0-vertices.')
#     else:
#         reconstructed_mesh = None
#         print('Mesh for geometry of index ' + str(index) + ' could not be reconstructed - 0-vertices.')
#     return reconstructed_mesh


# def getLowResMCBounds(resolution, XYZ_coord, models, index=None):
#     print(len(models))
#     print(models[0])
#     if(len(models) == 1):
#         outputs = models[0]({'xyz': XYZ_coord.to(device),'idx': torch.tensor([index]).to(device)})
#         if(len(outputs) == 3):
#             SDF_predicted, perf_predicted, latent = outputs
#         else:
#             SDF_predicted, latent = outputs
#     else:
#         model_grid, model_field =  models
#         grid_predicted, latent = model_grid({'xyz': XYZ_coord.to(device),'idx': torch.tensor([index]).to(device)})
#         SDF_predicted = model_field({'xyz': XYZ_coord.to(device),'idx': torch.tensor([index]).to(device)},grid_predicted)
#         SDF_predicted = torch.clamp(SDF_predicted, min=torch.tensor(-0.99999999), max=torch.tensor(0.99999999))
           
#     outputs_pos = XYZ_coord[0][(SDF_predicted.detach().cpu()<0).flatten().detach().numpy()].detach().numpy()
#     if(outputs_pos.shape[0]>0):
#         buffer = (1/resolution)*2
#         extents = np.max(outputs_pos, axis=0) + buffer - (np.min(outputs_pos, axis=0)-buffer)
#         min_v = (np.min(outputs_pos, axis=0)-buffer)
#         return extents, min_v
#     else:
#         return None, None 



# # Use marching cubes to reconstruct SDF surface from model sdf prediction
# def reconstructMCMeshJoint(resolution, XYZ_coord, model_sdf, model_grid, index=None):
#     grid_predicted, latent = model_grid.to("cpu")({'xyz': XYZ_coord,'idx': torch.tensor([index])})
#     SDF_predicted = model_sdf.to("cpu")({'xyz': XYZ_coord,'idx': torch.tensor([index])},grid_predicted)
#     SDF_predicted_cl = torch.clamp(SDF_predicted, min=torch.tensor(-0.99999999), max=torch.tensor(0.99999999))
#     SDF_predicted_cl = SDF_predicted_cl.detach().cpu().numpy().reshape((resolution, resolution, resolution))
#     verts, faces = mcubes.marching_cubes(SDF_predicted_cl, 0)
#     verts = verts/resolution
#     # verts, faces, normals, values = skimage.measure.marching_cubes(
#     #         SDF_predicted_cl, level=0.0, spacing=[resolution] * 3
#     #     )
#     if(len(verts) != 0):
#         reconstructed_mesh = trimesh.Trimesh(verts, faces)
#     else:
#         reconstructed_mesh = None
#         print('Mesh for geometry of index ' + str(index) + ' could not be reconstructed - 0-vertices.')
#     return reconstructed_mesh






# Plot losses
def plotLosses(losses, loss_type='Loss', x_range=None):
    losses_u = np.array([np.mean(a) for a in np.array_split(losses,2000)])
    plt.figure(figsize=(30,6))
    plt.plot(losses_u, label=loss_type)
    if(x_range is not None):
        plt.xlim(x_range)
    plt.xlabel('Epochs')
    plt.ylabel(loss_type)
    plt.title(loss_type + ' Evolution')
    plt.legend()
    plt.show()
    
# Function to plot multiple meshes side-by-side
def plotGeobyPerfRank(data_df, perf_n, geo_directory, reconstruction_dir, buffered_bbox, load_type, epoch_to_view, show_grdmesh, MCresolution = 32, descending = True, num = 5, flatshading: bool = False, height=800, width=800, color="white"):

    if(descending):
        sorted_real_ids = data_df.sort_values(by=[perf_n])['real_idx'].values[0:num]
        sorted_set_ids = data_df.sort_values(by=[perf_n])['idx'].values[0:num]
        sorted_perf = data_df.sort_values(by=[perf_n])[perf_n].values[0:num]
    else:
        sorted_real_ids = data_df.sort_values(by=[perf_n])['real_idx'][::-1].values[0:num]
        sorted_set_ids = data_df.sort_values(by=[perf_n])['idx'][::-1].values[0:num]
        sorted_perf = data_df.sort_values(by=[perf_n])[perf_n][::-1].values[0:num]

    fig = make_subplots(rows=1, cols=num, subplot_titles=(['ID :' + str(sorted_real_ids[i]) + ' : ' + str('{:.4f}'.format(sorted_perf[i])) for i in range(len(sorted_real_ids))]), specs=[list(np.repeat({"type": "scene"}, num))])

    mesh_plots = []
    for p in range(num):
        real_idx = int(sorted_real_ids[p])
        set_fromIDX = "{:04d}".format(int(real_idx/500)*500) + '_' + "{:04d}".format(((int(real_idx/500)+1)*500)-1)
        ground_truth_mesh = normalize_mesh(trimesh.load_mesh(geo_directory + '/' + set_fromIDX + '/' + 'building_' + str(real_idx) + '.obj', skip_materials=True), buffered_bbox)
        rec_mesh = trimesh.load(reconstruction_dir + '/' + 'reconstruction_' + str(MCresolution) + '_index_' + str(int(sorted_set_ids[p])) + '_' + load_type + '_' + str(epoch_to_view) +'.obj')
        if(show_grdmesh):
            mesh = ground_truth_mesh
        else:
            mesh = rec_mesh
        x, y, z = np.array(mesh.vertices).T
        i, j, k = np.array(mesh.faces).T
        mesh_plot = go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                color=color,
                opacity=1
            )
        mesh_plots.append(mesh_plot)
        fig.add_trace(mesh_plot, row=1, col=(p+1))

    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=5, y=5, z=5)
                  )
    fig.update_scenes(
                      xaxis_showbackground=False,
                      yaxis_showbackground=False,
                      zaxis_showbackground=False,
                      xaxis_showspikes=False,
                      yaxis_showspikes=False,
                      zaxis_showspikes=False,
                      xaxis=dict(visible=False, showticklabels=False),
                      yaxis=dict(visible=False, showticklabels=False),
                      zaxis=dict(visible=False, showticklabels=False),
                      camera=camera
                      )
    fig.update_traces(
        flatshading=flatshading, lighting=dict(specular=1.0), selector=dict(type="mesh3d")
    )
    fig.update_layout(margin=dict(r=25, l=25, b=25, t=25))
    fig.update_layout(scene_aspectmode="data")
    fig.update_layout(
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=False, showticklabels=False),
            yaxis=dict(visible=False, showticklabels=False),
            zaxis=dict(visible=False, showticklabels=False),
        ),
        width=width, height=height
    )
    fig.update_annotations(font_size=8)

    return fig
    