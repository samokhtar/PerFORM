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

from building_sdf.learning.network_utils import LipschitzLinear

## Loss functions and regularizations

# From https://github.com/pytorch/pytorch/issues/104564
@torch.compile
def cosine_similarity(t1, t2, dim=-1, eps=1e-8):
    # get normalization value
    t1_div = torch.linalg.vector_norm(t1, dim=dim, keepdims=True)
    t2_div = torch.linalg.vector_norm(t2, dim=dim, keepdims=True)

    t1_div = t1_div.clone()
    t2_div = t2_div.clone()
    with torch.no_grad():
        t1_div.clamp_(math.sqrt(eps))
        t2_div.clamp_(math.sqrt(eps))

    # normalize, avoiding division by 0
    t1_norm = t1 / t1_div
    t2_norm = t2 / t2_div

    return (t1_norm * t2_norm).sum(dim=dim)


# # Based on the implementation in https://github.com/ml-for-gp/jaxgptoolbox/blob/main/demos/lipschitz_mlp/model.py
# def get_lipschitz_loss(models):
#     """
#     This function computes the Lipschitz regularization Eq.7 in the [Liu et al 2022] 
#     """
#     loss_lip = 1.0
#     softplus = nn.Softplus()
#     for m in models:
#         for name, param in m.named_parameters():
#             if 'weight' in name:
#                 c = torch.max(torch.sum(torch.abs(param), axis = 1))
#                 loss_lip = loss_lip * softplus(c)      
#     return loss_lip

# Based on the implementation in https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/lipmlp.py
def get_lipschitz_loss(models):
    loss_lipc = 1.0
    for m in models:
        for name, layer in m.named_modules():#[n for n in m.named_modules()]:
            if(isinstance(layer, LipschitzLinear)):
                loss_lipc = (loss_lipc * layer.get_lipschitz_constant())#.mean()
    return loss_lipc

# # Based on the implementation in https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/lipmlp.py
# def get_lipschitz_loss(models):
#     loss_lipc = 1.0
#     for m in models:
#         for ii in range(len(m.layers)):
#             loss_lipc = loss_lipc * m.layers[ii].get_lipschitz_constant()
#         loss_lipc = loss_lipc *  m.layer_output.get_lipschitz_constant()
#     return loss_lipc



# Calculate the loss function including regularizations
def loss_fn(model_output, ground_truth, models, loss_fn_param, xyz, normals):

    sdf, latents = model_output

    # Clamp the loss terms
    sdf_cl = torch.clamp(sdf, -1*loss_fn_param['sdf_loss_clamp_val'], loss_fn_param['sdf_loss_clamp_val']) if(loss_fn_param['sdf_loss_clamp']) else torch.clamp(sdf,-1,1)
    gt_cl = torch.clamp(ground_truth, -1*loss_fn_param['sdf_loss_clamp_val'], loss_fn_param['sdf_loss_clamp_val']) if(loss_fn_param['sdf_loss_clamp']) else torch.clamp(ground_truth,-1,1)
    sdf_nr = torch.clamp(sdf,-1,1)
    gt_nr = torch.clamp(ground_truth,-1,1)
    
    # Calculate the sdf loss
    if(loss_fn_param['sdf_loss'] == 'l2'):
        sdf_loss = ((sdf_nr - gt_nr)**2).mean()
    else:
        sdf_loss = (torch.abs(sdf_nr-gt_nr)).mean()
        
    # Calculate the clamped sdf loss
    if(loss_fn_param['sdf_loss'] == 'l2'):
        sdf_loss_cl = ((sdf_cl - gt_cl)**2).mean()
    else:
        sdf_loss_cl = (torch.abs(sdf_cl-gt_cl)).mean()
        
    
    # From https://github.com/vsitzmann/siren/blob/master/diff_operators.py
    if((loss_fn_param['normal_loss_lambda'] != 0) or (loss_fn_param['gradient_loss_lambda'] != 0)):
        grad_outputs = torch.ones_like(sdf)
        gradient = torch.autograd.grad(sdf, [xyz], grad_outputs=grad_outputs, create_graph=True)[0]
        
    # Calculate the normal loss/constraint
    normal_loss = torch.tensor(0).float().to(sdf.device)
    if(loss_fn_param['normal_loss_lambda'] != 0):
        normal_loss = torch.where(ground_truth != 0., 1 - F.cosine_similarity(gradient, normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient.to(sdf.device)[..., :1])).mean()
    # Calculate the gradient loss/constraint
    # From https://github.com/vsitzmann/siren/blob/master/diff_operators.py
    gradient_loss = torch.tensor(0).float().to(sdf.device)
    if(loss_fn_param['gradient_loss_lambda'] != 0):
        gradient_loss = torch.abs(gradient.norm(dim=-1) - 1).mean()
    #print('gradient_loss',gradient_loss)
    #print('gradient_loss.shape',gradient_loss.shape)

    # Calculate the latent regularization loss
    latent_loss = torch.tensor(0).float().to(sdf.device)
    if(loss_fn_param['latent_reg_lambda']!= 0):
        latent_loss = (latents**2).mean()

    # Calculate the weight regularization loss
    param_loss = torch.tensor(0).float().to(sdf.device)
    if((loss_fn_param['weight_reg_lambda'] != 0)):
        for m in models:
            for name, param in m.named_parameters():
                if ('grid.latent' in name) or ('latent' not in name):
                    # print('name',name)
                    param_loss += (param**2).mean()
            #         print('param_loss',param_loss)
            # print('param_loss',param_loss)

    # Calculate the lipschitz regularization loss
    lip_loss = torch.tensor(0).float().to(sdf.device)
    if(loss_fn_param['lipschitz_reg_lambda'] != 0):
        lip_loss = get_lipschitz_loss(models)

    # Calculate the combined loss
    comb_loss = (sdf_loss*(1-loss_fn_param['sdf_loss_clamp_ratio'])*loss_fn_param['sdf_loss_lambda']) + ((loss_fn_param['sdf_loss_clamp_ratio'])*sdf_loss_cl*loss_fn_param['sdf_loss_lambda']) + (loss_fn_param['latent_reg_lambda']*latent_loss) + (loss_fn_param['weight_reg_lambda']*param_loss) + (loss_fn_param['lipschitz_reg_lambda']*lip_loss) + (loss_fn_param['gradient_loss_lambda']*gradient_loss) + (loss_fn_param['normal_loss_lambda']*normal_loss)

    return comb_loss, sdf_loss, sdf_loss_cl, latent_loss, param_loss, lip_loss, gradient_loss, normal_loss

# Calculate the loss function including regularizations
def loss_fn_rot(model_output, ground_truth, models, loss_param):

    sdf, latents = model_output

    # Clamp the loss terms
    sdf_cl = torch.clamp(sdf, -1*loss_param['sdf_loss_clamp_val'], loss_param['sdf_loss_clamp_val']) if(loss_param['sdf_loss_clamp']) else torch.clamp(sdf,-1,1)
    gt_cl = torch.clamp(ground_truth, -1*loss_param['sdf_loss_clamp_val'], loss_param['sdf_loss_clamp_val']) if(loss_param['sdf_loss_clamp']) else torch.clamp(ground_truth,-1,1)
    sdf_nr = torch.clamp(sdf,-1,1)
    gt_nr = torch.clamp(ground_truth,-1,1)
    
    # Calculate the sdf loss
    if(loss_param['sdf_loss'] == 'l2'):
        sdf_loss = ((sdf_nr - gt_nr)**2).mean()
    else:
        sdf_loss = (torch.abs(sdf_nr-gt_nr)).mean()
        
    # Calculate the clamped sdf loss
    if(loss_param['sdf_loss'] == 'l2'):
        sdf_loss_cl = ((sdf_cl - gt_cl)**2).mean()
    else:
        sdf_loss_cl = (torch.abs(sdf_cl-gt_cl)).mean()
        
    # Calculate the weight regularization loss
    param_loss = torch.tensor(0).float().to(sdf.device)
    if((loss_param['weight_reg_lambda'] != 0)):
        for m in models:
            for name, param in m.named_parameters():
                if 'latent' not in name:
                    param_loss += (param**2).mean()

    # Calculate the combined loss
    comb_loss = (sdf_loss*(1-loss_param['sdf_loss_clamp_ratio'])*loss_param['sdf_loss_lambda']) + ((loss_param['sdf_loss_clamp_ratio'])*sdf_loss_cl*loss_param['sdf_loss_lambda']) + (loss_param['weight_reg_lambda']*param_loss)

    return comb_loss, sdf_loss, sdf_loss_cl, param_loss
    
# Calculate the loss function including regularizations for latent only
def loss_fn_latent(model_output, ground_truth, model, loss_fn_param, xyz, normals):   

    sdf, latents = model_output
    
    # Clamp the loss terms
    sdf_cl = torch.clamp(sdf, -1*loss_fn_param['sdf_loss_clamp_val'], loss_fn_param['sdf_loss_clamp_val']) if(loss_fn_param['sdf_loss_clamp']) else torch.clamp(sdf,-1,1)
    gt_cl = torch.clamp(ground_truth, -1*loss_fn_param['sdf_loss_clamp_val'], loss_fn_param['sdf_loss_clamp_val']) if(loss_fn_param['sdf_loss_clamp']) else torch.clamp(ground_truth,-1,1)
    sdf_nr = torch.clamp(sdf,-1,1)
    gt_nr = torch.clamp(ground_truth,-1,1)

    # Calculate the sdf loss
    if(loss_fn_param['sdf_loss'] == 'l2'):
        sdf_loss = ((sdf_nr - gt_nr)**2).mean()
    else:
        sdf_loss = (torch.abs(sdf_nr-gt_nr)).mean()
        
    # Calculate the clamped sdf loss
    if(loss_fn_param['sdf_loss'] == 'l2'):
        sdf_loss_cl = ((sdf_cl - gt_cl)**2).mean()
    else:
        sdf_loss_cl = (torch.abs(sdf_cl-gt_cl)).mean()
        
    # Calculate the normal loss/constraint
    # From https://github.com/vsitzmann/siren/blob/master/diff_operators.py
    
    if((loss_fn_param['normal_loss_lambda'] != 0) or (loss_fn_param['gradient_loss_lambda'] != 0)):
        grad_outputs = torch.ones_like(sdf)
        gradient = torch.autograd.grad(sdf, [xyz], grad_outputs=grad_outputs, create_graph=True)[0]
    
    normal_loss = torch.tensor(0).float().to(sdf.device)
    if(loss_fn_param['normal_loss_lambda'] != 0):
        normal_loss = torch.where(ground_truth != 0., 1 - F.cosine_similarity(gradient, normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1])).mean()
    
    # Calculate the gradient loss/constraint
    # From https://github.com/vsitzmann/siren/blob/master/diff_operators.py
    gradient_loss = torch.tensor(0).float().to(sdf.device)
    if(loss_fn_param['gradient_loss_lambda'] != 0):
        gradient_loss = torch.abs(gradient.norm(dim=-1) - 1).mean()
    #print('gradient_loss',gradient_loss)
    #print('gradient_loss.shape',gradient_loss.shape)

    # Calculate the latent regularization loss
    latent_loss = torch.tensor(0).float().to(sdf.device)
    if(loss_fn_param['latent_reg_lambda']!= 0):
        latent_loss = (latents**2).mean()

    # Calculate the combined loss
    comb_loss = sdf_loss*(1-loss_fn_param['sdf_loss_clamp_ratio']) + (loss_fn_param['sdf_loss_clamp_ratio'])*sdf_loss_cl + (loss_fn_param['latent_reg_lambda']*latent_loss) + (loss_fn_param['gradient_loss_lambda']*gradient_loss) + (loss_fn_param['normal_loss_lambda']*normal_loss)

    return comb_loss, sdf_loss, sdf_loss_cl, (loss_fn_param['latent_reg_lambda']*latent_loss), torch.tensor(0).float().to(sdf.device), torch.tensor(0).float().to(sdf.device), gradient_loss, normal_loss



# Calculate the loss function including regularizations
def loss_fn_hybrid(model_output, ground_truth, param_latent, models, loss_fn_param):

    sdf = model_output
    latents = param_latent

    # Clamp the loss terms
    sdf_cl = torch.clamp(sdf, -1*loss_fn_param['sdf_loss_clamp_val'], loss_fn_param['sdf_loss_clamp_val']) if(loss_fn_param['sdf_loss_clamp']) else sdf
    gt_cl = torch.clamp(ground_truth, -1*loss_fn_param['sdf_loss_clamp_val'], loss_fn_param['sdf_loss_clamp_val']) if(loss_fn_param['sdf_loss_clamp']) else ground_truth
    sdf_nr = sdf
    gt_nr = ground_truth
    
    # Calculate the sdf loss
    if(loss_fn_param['sdf_loss'] == 'l2'):
        sdf_loss = ((sdf_nr - gt_nr)**2).mean()
    else:
        sdf_loss = (torch.abs(sdf_nr-gt_nr)).mean()
        
    # Calculate the clamped sdf loss
    if(loss_fn_param['sdf_loss'] == 'l2'):
        sdf_loss_cl = ((sdf_cl - gt_cl)**2).mean()
    else:
        sdf_loss_cl = (torch.abs(sdf_cl-gt_cl)).mean()

    # Calculate the latent regularization loss
    latent_loss = 0
    if(loss_fn_param['latent_reg_loss']):
        latent_loss = (latents**2).mean()

    # Calculate the weight regularization loss
    param_loss = 0
    if((not loss_fn_param['lipschitz_reg_loss'])):
        for m in models:
            for name, param in m.named_parameters():
                if 'latent' not in name:
                    param_loss += (param**2).mean()

    # Calculate the lipschitz regularization loss
    lip_loss = 0
    if(loss_fn_param['lipschitz_reg_loss']):
        lip_loss = get_lipschitz_loss(model)

    # Calculate the combined loss
    comb_loss = sdf_loss*(1-loss_fn_param['sdf_loss_clamp_ratio']) + (loss_fn_param['sdf_loss_clamp_ratio'])*sdf_loss_cl + (loss_fn_param['latent_reg_lambda']*latent_loss) + (loss_fn_param['weight_reg_lambda']*param_loss) + (loss_fn_param['lipschitz_reg_lambda']*lip_loss)

    return comb_loss, sdf_loss, (loss_fn_param['latent_reg_lambda']*latent_loss), (loss_fn_param['weight_reg_lambda']*param_loss),(loss_fn_param['lipschitz_reg_lambda']*lip_loss)



# Calculate the loss function including regularizations
def loss_fn_perf(model_output, ground_truth, ground_truth_sdf, model, loss_param, xyz):
    
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    output, ctxt_latents = model_output
    perf = output[..., :1]
    gt_perf = ground_truth[..., :1]
    
    # Define biasing term - closer to surface and outside "posSDF" - take larger weight
    if(loss_param['sdf_bias_term'] != 0):
        bias_sdf = ground_truth_sdf
        bias_term = loss_param['sdf_bias_term']  # Smaller means more emphasis closer to surface (between 0 and 1)
        bias_sdf[bias_sdf<0] = 0.9
        bias_sdf = torch.pow(bias_term,bias_sdf).to(perf.device)
    else:
        bias_sdf = torch.ones(gt_perf.shape).to(perf.device)
        
    # Define biasing term for heights
    if(loss_param['z_bias_term'] != 0):
        bias_Z = torch.pow(loss_param['z_bias_term'],xyz[..., 1].unsqueeze(-1)).to(perf.device)    # Smaller means more emphasis closer to the ground
        #print('bias_Z',bias_Z.shape)
    else:
        bias_Z = torch.ones(gt_perf.shape).to(perf.device)
        
    # Define sdf clamp - do not consider points way too far from surface or inside
    if(loss_param['sdf_clamp_term'] != 0):
        perf = torch.where((ground_truth_sdf < loss_param['sdf_clamp_term'])and(ground_truth_sdf>0), perf, 0)
        gt_perf = torch.where((ground_truth_sdf < loss_param['sdf_clamp_term'])and(ground_truth_sdf>0), gt_perf, 0)

    # Calculate cosine similarity if there is a vector involved - Higher values means higher similarity between vectors
    cos_sim_loss = torch.tensor(0).float().to(perf.device)
    unbias_cos_sim_loss = torch.tensor(0).float().to(perf.device)
    if((output.shape[-1]!= 1) and (loss_param['cos_sim_lambda'] != 0)):
        vec = output[..., 1:]
        gt_vec = ground_truth[..., 1:]
        cos_sim_val = (cosine_similarity(vec, gt_vec)+1)/2
        cos_sim_loss = (cos_sim_val*bias_sdf[:,:,0]*bias_Z[:,:,0]).mean()
        unbias_cos_sim_loss = cos_sim_val.mean()
    
    # Calculate the perf loss
    if(loss_param['perf_loss'] == 'l2'):
        perf_loss = (((perf - gt_perf)*bias_sdf*bias_Z)**2).mean()
        unbias_perf_loss = (((perf - gt_perf))**2).mean()
    else:
        perf_loss = ((torch.abs(perf-gt_perf))*bias_sdf*bias_Z).mean()
        unbias_perf_loss = ((torch.abs(perf-gt_perf))).mean()
        
    # Calculate the weight regularization loss
    param_loss = torch.tensor(0).float().to(perf.device)
    if((loss_param['weight_reg_lambda'] != 0)):
        for name, param in model[0].named_parameters():
            if 'latent' not in name:
                param_loss += (param**2).mean()

    # Calculate the combined loss
    cos_sim_loss_div = (loss_param['cos_sim_lambda']*cos_sim_loss) if((loss_param['cos_sim_lambda'] != 0) and (output.shape[-1]!= 1)) else torch.tensor(1).float().to(perf.device)
    comb_loss = ((perf_loss*loss_param['perf_loss_lambda']) + (loss_param['weight_reg_lambda']*param_loss)) / cos_sim_loss_div

    return comb_loss, unbias_perf_loss, unbias_cos_sim_loss, param_loss


# Calculate the loss function including regularizations
def loss_fn_perf_aggCDF(model_output, ground_truth, model, loss_param):
    
    output, ctxt_latents = model_output
    #print('output',output.shape)
    output_cdf = torch.cumsum(output,-1)
    #print('output_cdf',output_cdf[5][0])
    diff = torch.diff(ground_truth, prepend=torch.zeros(ground_truth.shape[0],ground_truth.shape[1],1))
    gt_diff = diff.reshape(diff.shape[0], diff.shape[1], output_cdf.shape[-1], -1).sum(axis=-1)
    gt_cdf = torch.cumsum(gt_diff,-1)
    #print('gt_cdf',gt_cdf[5][0])
    
    # Define the distribution diff loss
    dist_loss_weights = loss_param['dist_loss_weights'] if(loss_param['dist_loss_weights'] is not None) else torch.ones(output.shape[-1]).to(output.device)

    if(loss_param['dist_loss'] == 'l2'):
        dist_loss = (((torch.abs(gt_cdf - output_cdf))*dist_loss_weights)**2).mean()
    elif(loss_param['dist_loss'] == 'l1'):
        dist_loss = ((torch.abs(gt_cdf - output_cdf)*dist_loss_weights)).mean()
    else:
        dist_loss = torch.max(torch.max(torch.abs(gt_cdf - output_cdf)*dist_loss_weights,-1)[0],0)[0].mean()
        
    # Calculate the weight regularization loss
    param_loss = torch.tensor(0).float().to(ground_truth.device)
    if((loss_param['weight_reg_lambda'] != 0)):
        for name, param in model[0].named_parameters():
            if 'latent' not in name:
                param_loss += (param**2).mean()

    # Calculate the combined loss
    comb_loss = ((dist_loss*loss_param['dist_loss_lambda']) + (loss_param['weight_reg_lambda']*param_loss)) 

    return comb_loss, dist_loss, param_loss


# Calculate the loss function including regularizations
def calculateAllLossesperGeo_perf_aggCDF(model_output, ground_truth, dist_loss_weights=None):
    
    output, ctxt_latents = model_output
    #print('output',output.shape)
    output_cdf = torch.cumsum(output,-1)
    #print('output_cdf',output_cdf[5][0])
    diff = torch.diff(ground_truth, prepend=torch.zeros(ground_truth.shape[0],ground_truth.shape[1],1))
    gt_diff = diff.reshape(diff.shape[0], diff.shape[1], output_cdf.shape[-1], -1).sum(axis=-1)
    gt_cdf = torch.cumsum(gt_diff,-1)
    #print('gt_cdf',gt_cdf[5][0])
    
    # Define the distribution diff loss
    dist_loss_weights = dist_loss_weights if(dist_loss_weights is not None) else torch.ones(output.shape[-1]).to(output.device)

    l2_dist_loss, l1_dist_loss, max_dist_loss = [], [], []
    for i in range(output.shape[1]):
        l2_dist_loss.append((((torch.abs(gt_cdf[:,i] - output_cdf[:,i]))*dist_loss_weights)**2).mean(1).detach().numpy())
        l1_dist_loss.append(((torch.abs(gt_cdf[:,i] - output_cdf[:,i])*dist_loss_weights)).mean(1).detach().numpy())
        max_dist_loss.append(torch.max(torch.abs(gt_cdf[:,i] - output_cdf[:,i])*dist_loss_weights,-1)[0].detach().numpy())

    losses = [l2_dist_loss[0], l1_dist_loss[0], max_dist_loss[0], l2_dist_loss[1], l1_dist_loss[1], max_dist_loss[1]]
    losses_names = ['l2_dist_loss_srf', 'l1_dist_loss_srf', 'max_dist_loss_srf', 'l2_dist_loss_grd', 'l1_dist_loss_grd', 'max_dist_loss_grd']
    return losses, losses_names




# # Calculate the loss function including regularizations
# def loss_fn_parallel(model_output, ground_truth_sdf, ground_truth_perf, model, loss_fn_param):

#     outputs_sdf, outputs_perf, latents = model_output
#     #print('outputs.shape',outputs.shape)
#     sdf = outputs_sdf
#     perf = outputs_perf

#     # Clamp the loss terms
#     sdf_cl = torch.clamp(sdf, -1*loss_fn_param['sdf_loss_clamp_val'], loss_fn_param['sdf_loss_clamp_val']) if(loss_fn_param['sdf_loss_clamp']) else sdf
#     gt_cl = torch.clamp(ground_truth_sdf, -1*loss_fn_param['sdf_loss_clamp_val'], loss_fn_param['sdf_loss_clamp_val']) if(loss_fn_param['sdf_loss_clamp']) else ground_truth
#     sdf_nr = sdf
#     gt_nr = ground_truth_sdf

#     # Calculate the sdf loss
#     if(loss_fn_param['sdf_loss'] == 'l2'):
#         sdf_loss = ((sdf_nr - gt_nr)**2).mean()
#     else:
#         sdf_loss = (torch.abs(sdf_nr-gt_nr)).mean()
        
#     # Calculate the clamped sdf loss
#     if(loss_fn_param['sdf_loss'] == 'l2'):
#         sdf_loss_cl = ((sdf_cl - gt_cl)**2).mean()
#     else:
#         sdf_loss_cl = (torch.abs(sdf_cl-gt_cl)).mean()
        
#     # Calculate the perf loss
#     if(loss_fn_param['perf_loss'] == 'l2'):
#         perf_loss = ((perf - ground_truth_perf)**2).mean()
#     else:
#         perf_loss = (torch.abs(perf-ground_truth_perf)).mean()

#     # Calculate the latent regularization loss
#     latent_loss = 0
#     if(loss_fn_param['latent_reg_loss']):
#         latent_loss = (latents**2).mean()

#     # Calculate the weight regularization loss
#     param_loss = 0
#     if((not loss_fn_param['lipschitz_reg_loss'])):
#         for name, param in model.named_parameters():
#             if 'latent' not in name:
#                 param_loss += (param**2).mean()

#     # Calculate the lipschitz regularization loss
#     lip_loss = 0
#     if(loss_fn_param['lipschitz_reg_loss']):
#         lip_loss = get_lipschitz_loss(model)

#     # Calculate the combined loss
#     comb_loss_sdf = sdf_loss*(1-loss_fn_param['sdf_loss_clamp_ratio']) + (loss_fn_param['sdf_loss_clamp_ratio'])*sdf_loss_cl + (loss_fn_param['latent_reg_lambda']*latent_loss) + (loss_fn_param['weight_reg_lambda']*param_loss) + (loss_fn_param['lipschitz_reg_lambda']*lip_loss)
#     comb_loss_all = comb_loss_sdf*loss_fn_param['geoToPerfRatio'] + perf_loss*(1-loss_fn_param['geoToPerfRatio'])/10

#     return comb_loss_all, comb_loss_sdf, perf_loss, sdf_loss, (loss_fn_param['latent_reg_lambda']*latent_loss), (loss_fn_param['weight_reg_lambda']*param_loss),(loss_fn_param['lipschitz_reg_lambda']*lip_loss)






# # Calculate the loss function including regularizations
# def loss_fn_joint(model_output, ground_truth_sdf, ground_truth_perf, model, loss_fn_param):

#     outputs, latents = model_output
#     #print('outputs.shape',outputs.shape)
#     sdf = outputs[0]
#     perf = outputs[1]

#     # Clamp the loss terms
#     sdf_cl = torch.clamp(sdf, -1*loss_fn_param['sdf_loss_clamp_val'], loss_fn_param['sdf_loss_clamp_val']) if(loss_fn_param['sdf_loss_clamp']) else sdf
#     gt_cl = torch.clamp(ground_truth_sdf, -1*loss_fn_param['sdf_loss_clamp_val'], loss_fn_param['sdf_loss_clamp_val']) if(loss_fn_param['sdf_loss_clamp']) else ground_truth
#     sdf_nr = sdf
#     gt_nr = ground_truth_sdf

#     # Calculate the sdf loss
#     if(loss_fn_param['sdf_loss'] == 'l2'):
#         sdf_loss = ((sdf_nr - gt_nr)**2).mean()
#     else:
#         sdf_loss = (torch.abs(sdf_nr-gt_nr)).mean()
        
#     # Calculate the clamped sdf loss
#     if(loss_fn_param['sdf_loss'] == 'l2'):
#         sdf_loss_cl = ((sdf_cl - gt_cl)**2).mean()
#     else:
#         sdf_loss_cl = (torch.abs(sdf_cl-gt_cl)).mean()
        
#     # Calculate the perf loss
#     if(loss_fn_param['perf_loss'] == 'l2'):
#         perf_loss = ((perf - ground_truth_perf)**2).mean()
#     else:
#         perf_loss = (torch.abs(perf-ground_truth_perf)).mean()

#     # Calculate the latent regularization loss
#     latent_loss = 0
#     if(loss_fn_param['latent_reg_loss']):
#         latent_loss = (latents**2).mean()

#     # Calculate the weight regularization loss
#     param_loss = 0
#     if((not loss_fn_param['lipschitz_reg_loss'])):
#         for name, param in model.named_parameters():
#             if 'latent' not in name:
#                 param_loss += (param**2).mean()

#     # Calculate the lipschitz regularization loss
#     lip_loss = 0
#     if(loss_fn_param['lipschitz_reg_loss']):
#         lip_loss = get_lipschitz_loss(model)

#     # Calculate the combined loss
#     comb_loss_sdf = sdf_loss*(1-loss_fn_param['sdf_loss_clamp_ratio']) + (loss_fn_param['sdf_loss_clamp_ratio'])*sdf_loss_cl + (loss_fn_param['latent_reg_lambda']*latent_loss) + (loss_fn_param['weight_reg_lambda']*param_loss) + (loss_fn_param['lipschitz_reg_lambda']*lip_loss)
#     comb_loss_all = comb_loss_sdf*loss_fn_param['geoToPerfRatio'] + perf_loss*(1-loss_fn_param['geoToPerfRatio'])/10

#     return comb_loss_all, comb_loss_sdf, perf_loss, sdf_loss, (loss_fn_param['latent_reg_lambda']*latent_loss), (loss_fn_param['weight_reg_lambda']*param_loss),(loss_fn_param['lipschitz_reg_lambda']*lip_loss)


