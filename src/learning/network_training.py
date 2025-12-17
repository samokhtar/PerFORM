import os
import time
import collections

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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
#torch.manual_seed(0)

from building_sdf.learning.network_viz import *
from building_sdf.learning.network_modules import *

torch.autograd.set_detect_anomaly(True)

# Fit ML function

def fit(
    model: nn.Module,
    model_dir,
    train_dataloader,
    loss_fn,
    summary_fn,
    plotting_function = None,
    optLatentOnly = False,
    fromCheckPt = None,
    dataset_override = None
   ):

    # Define data arrays
    train_losses = []   
    idx_ar, epoch_ar, step_ar, loss_ar, sdf_loss_ar, sdf_clamped_loss_ar, latent_loss_ar, param_loss_ar, lip_loss_ar, gradient_loss_ar, normal_loss_ar = [], [], [], [], [], [], [], [], [], [], []
    d_arr = [idx_ar, epoch_ar, step_ar, loss_ar, sdf_loss_ar, sdf_clamped_loss_ar, latent_loss_ar, param_loss_ar, lip_loss_ar, gradient_loss_ar, normal_loss_ar]
    d_arr_names = ['idx', 'epoch', 'step', 'loss', 'sdf_loss', 'sdf_clamped_loss', 'latent_loss', 'param_loss', 'lip_loss', 'gradient_loss', 'normal_loss']
    
    # Create model sub-directories
    model_dir, checkpoints_dir, training_viz_dir, rendering_dir = createModelDirs(model_dir)
    #print('rendering_dir',rendering_dir)
    
    # Create model sub-directories
    if(not optLatentOnly):
        mlp_param, loss_param, rep_param, dataset_param = loadModelParam(model_dir, param_ar = ['mlp_param', 'loss_param', 'rep_param', 'dataset_train_param'])
        dataset_param = dataset_param if(dataset_override is None) else dataset_override
    if(optLatentOnly): # if model is test, do not use the latent and model regularizations

        m_b_dir = '/'.join(model_dir.split('/')[0:-1]) if(model_dir.split('/')[-2]!='test_sets') else '/'.join(model_dir.split('/')[0:-2])
        print('/'.join(model_dir.split('/')[0:-1]))
        print('/'.join(model_dir.split('/')[0:-2]))
        print(model_dir.split('/')[-2])
        print(m_b_dir)
        mlp_param, loss_param, rep_param, dataset_param = loadModelParam(m_b_dir, param_ar = ['mlp_param', 'loss_param', 'rep_param', 'dataset_test_param'])
        dataset_param = dataset_param if(dataset_override is None) else dataset_override
        loss_param['weight_reg_lambda'] = 0
        loss_param['lipschitz_reg_lambda'] = 0
        loss_param['latent_reg_lambda'] = 0
    
    print(int(len(dataset_param['dataset_ids'])))
    print(model.latents.weight.shape[0])
    if(int(len(dataset_param['dataset_ids']))!=model.latents.weight.shape[0]):
        print('Number of model latents does not match the saved dataset list!')
    
    # Setup parallel model if GPUs>1
    if (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)
        print("This training will use", torch.cuda.device_count(), "GPUs.")
    
    # Disable dropout if in test phase
    if(optLatentOnly):
        dropout_modules = [module for module in model.modules() if isinstance(module,torch.nn.Dropout)]
        [module.eval() for module in dropout_modules] # disable dropout
    
    # Load model parameters
    det_losses_lambdas = [loss_param['sdf_loss_lambda']*(1-loss_param['sdf_loss_clamp_ratio']), loss_param['sdf_loss_lambda']*loss_param['sdf_loss_clamp_ratio'], loss_param['latent_reg_lambda'], loss_param['weight_reg_lambda'], loss_param['lipschitz_reg_lambda'], loss_param['gradient_loss_lambda'], loss_param['normal_loss_lambda']]

    # Get lr rates based on starting epoch
    if ((fromCheckPt is None) or (int(fromCheckPt/loss_param['lr_sch_interval'])==0)):
        lr_model = loss_param['lr_sch_initial_model']
        lr_latent = loss_param['lr_sch_initial_latent']
    else:
        lr_model = loss_param['lr_sch_initial_model']*np.power(loss_param['lr_sch_factor'],(int(np.floor(fromCheckPt/loss_param['lr_sch_interval']))))
        lr_latent = loss_param['lr_sch_initial_latent']*np.power(loss_param['lr_sch_factor'],(int(np.floor(fromCheckPt/loss_param['lr_sch_interval']))))
    
    # Define optimizer
    if(not optLatentOnly):
        if(torch.cuda.device_count() > 1):
            optimizer = torch.optim.Adam(
                    [
                        {
                            "params": model.module.mlp.parameters(),
                            "lr": lr_model,
                        },
                        {
                            "params": model.module.latents.parameters(),
                            "lr": lr_latent,
                        },
                    ]
                )
        else:
                        optimizer = torch.optim.Adam(
                    [
                        {
                            "params": model.mlp.parameters(),
                            "lr": lr_model,
                        },
                        {
                            "params": model.latents.parameters(),
                            "lr": lr_latent,
                        },
                    ]
                )
    else:
        for name, param in model.named_parameters():
            if param.requires_grad and ('latent' not in name):
                param.requires_grad = False
        if(torch.cuda.device_count() > 1):
            optimizer = torch.optim.Adam(
                    [
                        {
                            "params": model.module.latents.parameters(),
                            "lr": lr_latent,
                        },
                    ]
                )
        else:
                        optimizer = torch.optim.Adam(
                    [
                        {
                            "params": model.latents.parameters(),
                            "lr": lr_latent,
                        },
                    ]
                )
    
    
    

    # Loading from checkpoint
    if(fromCheckPt is not None):
        models, train_losses, d_arr = loadfromCheckpoint([model], ['model'], checkpoints_dir, d_arr, d_arr_names, fromCheckPt)
        model = models[0]
        
    # Get XYZ for reconstruction
    if(plotting_function is not None):
        XYZ_coord = (get3DGrid(rep_param['mc_resolution'])*2)-1 if(dataset_param['s_range'] == 'N1_1') else get3DGrid(rep_param['mc_resolution'])
        XYZ_coord = XYZ_coord if(plotting_function is not None) else None
    
    # Get bounding box
    buffered_bbox = np.array(rep_param['buffered_bbox'])
    
    # Define the number of batch steps needed to complete one epoch
    num_steps_per_epoch = len(train_dataloader) 
    print('The number of steps per epoch is ' + str(num_steps_per_epoch))
    total_steps = 0

    starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1
    total_epochs = rep_param['total_epochs_train'] if(not optLatentOnly) else rep_param['total_epochs_test']
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    loss = 0
    
    # Set the model to train
    model.train()
    
    # Weight updates checks
    model_weights = [0,0]
    model_w_types = ['latent', 'mlp']
    
    #torch.backends.cudnn.benchmark = True

    for epoch in range(total_epochs):

        if(epoch >= starting_epoch):

            start_time_epoch = time.time()

            # Create train_dataloader_iterator
            train_dataloader_iter = iter(train_dataloader)
            
            for i in range(len(train_dataloader)):
                
                # Get the next batch of data and move it to the GPU
                model_input, ground_truth_sdf, ground_truth_normals, g_name = next(train_dataloader_iter)
                if(torch.cuda.device_count()>0):
                    model_input, ground_truth_sdf, ground_truth_normals = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_normals)

                # Compute the MLP output for the given input data and compute the loss
                if((loss_param['normal_loss_lambda'] != 0) or (loss_param['gradient_loss_lambda'] != 0)):
                    model_input['xyz'].requires_grad = True  # to calculate normal gradients
                model_output = model(model_input)
                sdf, latents = model_output
                
                if torch.norm(latents).item() > 50:
                    print(f"[!] Latent norm too high at step {i*epoch}")
                
                # # Weight updates checks
                # cur_latents_weight = torch.sum(model.module.latents.weight) if(torch.cuda.device_count() > 1) else torch.sum(model.latents.weight)
                # cur_mlp_weight = torch.sum(model.module.mlp.layer_in[0].weight)+torch.sum(model.module.mlp.layer_out[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.mlp.layer_in[0].weight)+torch.sum(model.mlp.layer_out[0].weight)
                # cur_model_weights = [cur_latents_weight, cur_mlp_weight]
                # for i in range(len(model_w_types)):
                #     if(cur_model_weights[i]==model_weights[i]):
                #         if(not((optLatentOnly) and (model_w_types[i] == 'mlp'))):
                #             print('The ' + model_w_types[i] + ' model weights are not updating!')
                # model_weights = cur_model_weights
                #print('st_wgt', time.time()-start_time_epoch)

                # Implement a loss function between the ground truth and model output 
                loss, sdf_loss, sdf_loss_cl, latent_loss, param_loss, lip_loss, gradient_loss, normal_loss = loss_fn(model_output, ground_truth_sdf, [model], loss_param, model_input['xyz'], ground_truth_normals) 
                #print('loss',loss)
                detailed_losses = [sdf_loss, sdf_loss_cl, latent_loss, param_loss, lip_loss, gradient_loss, normal_loss]
                
                # Append loss and detailed loss information per geometry
                train_losses.append(loss.detach().cpu().numpy())
                l_tr = [model_input['idx'].detach().cpu().numpy(), epoch, i, loss.detach().cpu().numpy()]
                [d_arr[d].append(l_tr[d]) for d in range(len(l_tr))]
                [d_arr[d+4].append(detailed_losses[d].detach().cpu().numpy()) for d in range(len(detailed_losses))]
                
                # Write summary
                if not total_steps % rep_param['steps_til_summary']:
                    logModelUpdates([model], ['model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'current')
                    summary_fn(total_steps, np.mean(train_losses), [scheduler.get_last_lr()[0]] if(optLatentOnly) else [scheduler.get_last_lr()[0],scheduler.get_last_lr()[1]], model_output, [ground_truth_sdf], [model], rendering_dir, det_losses = detailed_losses, det_losses_types = d_arr_names[4:], det_losses_lambdas = det_losses_lambdas) 

                # One step of optimization
                optimizer.zero_grad()
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"[!] NaN or Inf detected at step {i*epoch}, skipping update.")
                    print(f"loss = {loss}")
                    continue
                
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                total_steps += 1
            
            print("Before scheduler:", [group['lr'] for group in optimizer.param_groups])
            scheduler.step()
            print("After scheduler:", [group['lr'] for group in optimizer.param_groups])
            
            # Clamp learning rate manually
            min_lr = 1e-6
            for param_group in optimizer.param_groups:
                if param_group['lr'] < min_lr:
                    param_group['lr'] = min_lr

            lr_latent = scheduler.get_last_lr()[1] if(not optLatentOnly) else scheduler.get_last_lr()[0]
            lr_model = scheduler.get_last_lr()[0] if(not optLatentOnly) else 0 
            print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, model learning rate %0.8f, latent learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, lr_model, lr_latent))

            if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):
                # Save the model and losses at checkpoints
                logModelUpdates([model], ['model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'epoch', epoch = epoch)

    # Save the final model and detailed stats
    logModelUpdates([model], ['model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'final')
        

# Fit ML function
def fit_hybrid(
    model_grid: nn.Module,
    model_field: nn.Module,
    model_dir,
    train_dataloader,
    loss_fn,
    summary_fn,
    plotting_function = None,
    optLatentOnly = False,
    fromCheckPt = None,
    dataset_override = None
   ):

    # Define data arrays
    train_losses = []   
    idx_ar, epoch_ar, step_ar, loss_ar, sdf_loss_ar, sdf_clamped_loss_ar, latent_loss_ar, param_loss_ar, lip_loss_ar, gradient_loss_ar, normal_loss_ar = [], [], [], [], [], [], [], [], [], [], []
    d_arr = [idx_ar, epoch_ar, step_ar, loss_ar, sdf_loss_ar, sdf_clamped_loss_ar, latent_loss_ar, param_loss_ar, lip_loss_ar, gradient_loss_ar, normal_loss_ar]
    d_arr_names = ['idx', 'epoch', 'step', 'loss', 'sdf_loss', 'sdf_clamped_loss', 'latent_loss', 'param_loss', 'lip_loss', 'gradient_loss', 'normal_loss']
    
    # Setup parallel model if GPUs>1
    if (torch.cuda.device_count() > 1):
        model_grid = torch.nn.DataParallel(model_grid)
        model_field = torch.nn.DataParallel(model_field)
        print("This training will use", torch.cuda.device_count(), "GPUs.")
    elif(torch.cuda.device_count() == 1):
        print("This training will use 1 GPU.")
    else:
        print("This training will run on CPU.")
    
    # Create model sub-directories
    model_dir, checkpoints_dir, training_viz_dir, rendering_dir = createModelDirs(model_dir)
    
    # Get model parameters
    if(not optLatentOnly):
        grid_param, field_param, loss_param, rep_param, dataset_param = loadModelParam(model_dir, param_ar = ['grid_param', 'field_param', 'loss_param', 'rep_param', 'dataset_train_param'])
        dataset_param = dataset_param if(dataset_override is None) else dataset_override
    if(optLatentOnly): # if model is test, do not use the latent and model regularizations
        
        m_b_dir = '/'.join(model_dir.split('/')[0:-1]) if(model_dir.split('/')[-2]!='test_sets') else '/'.join(model_dir.split('/')[0:-2])

        grid_param, field_param, loss_param, rep_param, dataset_param = loadModelParam(m_b_dir, param_ar = ['grid_param', 'field_param', 'loss_param', 'rep_param', 'dataset_test_param']) if('/'.join(model_dir.split('/')[0:-1])!='test_sets') else loadModelParam('/'.join(model_dir.split('/')[0:-2]), param_ar = ['grid_param', 'field_param', 'loss_param', 'rep_param', 'dataset_test_param'])
        dataset_param = dataset_param if(dataset_override is None) else dataset_override
        loss_param['weight_reg_lambda'] = 0
        loss_param['lipschitz_reg_lambda'] = 0
        loss_param['latent_reg_lambda'] = 0
    
    # Load model parameters
    det_losses_lambdas = [loss_param['sdf_loss_lambda']*(1-loss_param['sdf_loss_clamp_ratio']), loss_param['sdf_loss_lambda']*loss_param['sdf_loss_clamp_ratio'], loss_param['latent_reg_lambda'], loss_param['weight_reg_lambda'], loss_param['lipschitz_reg_lambda'], loss_param['gradient_loss_lambda'], loss_param['normal_loss_lambda']]

    # Get lr rates based on starting epoch
    if ((fromCheckPt is None) or (np.floor(fromCheckPt/loss_param['lr_sch_interval'])==0)):
        lr_model = loss_param['lr_sch_initial_model']
        lr_latent = loss_param['lr_sch_initial_latent']
    else:
        lr_model = loss_param['lr_sch_initial_model']*np.power(loss_param['lr_sch_factor'],(int(np.floor(fromCheckPt/loss_param['lr_sch_interval']))))
        lr_latent = loss_param['lr_sch_initial_latent']*np.power(loss_param['lr_sch_factor'],(int(np.floor(fromCheckPt/loss_param['lr_sch_interval']))))
 
    # Define optimizer
    if(not optLatentOnly):
        if(torch.cuda.device_count() > 1):
            optimizer = torch.optim.Adam(
                    [
                        {
                            "params": model_grid.module.submodule.parameters(),
                            "lr": lr_model,
                        },
                        {
                            "params": model_field.module.parameters(),
                            "lr": lr_model,
                        },
                        {
                            "params": model_grid.module.latents.parameters(),
                            "lr": lr_latent,
                        },
                    ]
                )
        else:
            optimizer = torch.optim.Adam(
                    [
                        {
                            "params": model_grid.submodule.parameters(),
                            "lr": lr_model,
                        },
                        {
                            "params": model_field.parameters(),
                            "lr": lr_model,
                        },
                        {
                            "params": model_grid.latents.parameters(),
                            "lr": lr_latent,
                        },
                    ]
                )
    else:
        for m in [model_grid, model_field]:
            for name, param in m.named_parameters():
                if param.requires_grad and ('latent' not in name):
                    param.requires_grad = False
        if(torch.cuda.device_count() > 1):
            optimizer = torch.optim.Adam(
                    [
                        {
                            "params": model_grid.module.latents.parameters(),
                            "lr": lr_latent,
                        },
                    ]
                )
        else:
            optimizer = torch.optim.Adam(
                    [
                        {
                            "params": model_grid.latents.parameters(),
                            "lr": lr_latent,
                        },
                    ]
                )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
    loss = 0

    # Loading from checkpoint
    if(fromCheckPt is not None):
        model_grid.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_grid_epoch_%04d.pth' % fromCheckPt)))
        model_field.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_field_epoch_%04d.pth' % fromCheckPt)))
        train_losses = list(np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt))).flatten().astype(float))
        train_check_det = json.loads(open(checkpoints_dir + '/' + 'train_losses_detailed_epoch_%04d.json' % fromCheckPt).read())
        for t in range(len(d_arr)):
            d_arr[t] = train_check_det[d_arr_names[t]]
    
    # Get XYZ for reconstruction
    XYZ_coord = (get3DGrid(rep_param['mc_resolution'])*2)-1 if(dataset_param['s_range'] == 'N1_1') else get3DGrid(rep_param['mc_resolution'])
    XYZ_coord = XYZ_coord if(plotting_function is not None) else None
    
    # Get bounding box
    buffered_bbox = np.array(rep_param['buffered_bbox'])
    
    # Define the number of batch steps needed to complete one epoch
    num_steps_per_epoch = len(train_dataloader)
    print('The number of steps per epoch is ' + str(num_steps_per_epoch))
    total_steps = 0

    starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1
    total_epochs = rep_param['total_epochs_train'] if(not optLatentOnly) else rep_param['total_epochs_test']
    
    # Set the models to train
    model_grid.train()
    model_field.train()
    
    # Weight updates checks
    model_weights = [0,0,0]
    model_w_types = ['grid latent', 'grid decoder', 'field']

    for epoch in range(total_epochs):

        if(epoch >= starting_epoch):

            start_time_epoch = time.time()
            
            # Create train_dataloader_iterator
            train_dataloader_iter = iter(train_dataloader)
                
            for i in range(len(train_dataloader)):
                start_time = time.time()
                
                # Get the next batch of data and move it to the GPU
                model_input, ground_truth_sdf, ground_truth_normals, g_name = next(train_dataloader_iter)
                if(i == 0):
                    print(f"epoch:{epoch} g_name={g_name} | SDF min/max: {ground_truth_sdf.min():.4f}, {ground_truth_sdf.max():.4f}")
                    
                if(torch.cuda.device_count()>0):
                    model_input, ground_truth_sdf, ground_truth_normals = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_normals)
                    
                    # Skip batch if inputs are invalid
                if (
                    torch.isnan(model_input['xyz']).any()
                    or torch.isinf(model_input['xyz']).any()
                    or model_input['xyz'].abs().max() > 10
                ):
                    print(f"Skipping bad input at step {epoch*i}, g_name={g_name}, xyz range: [{model_input['xyz'].min():.4f}, {model_input['xyz'].max():.4f}]")
                    continue

                # Compute the MLP output for the given input data and compute the loss
                #model_input['xyz'].requires_grad = True  # to calculate normal gradients
                grid_output, param_latent = model_grid(model_input)

                # print('g_name',g_name)
                # print('param_latent',param_latent)
                model_output = model_field(model_input, grid_output)
                
                # Weight updates checks
                cur_grid_latents_weight = torch.sum(model_grid.module.latents.weight) if(torch.cuda.device_count() > 1) else torch.sum(model_grid.latents.weight)
                cur_grid_submodule_weight = torch.sum(model_grid.module.submodule.grid.decoder.in_conv[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model_grid.submodule.grid.decoder.in_conv[0].weight)
                cur_grid_mlp_weight = torch.sum(model_field.module.scene_rep.mlp.layer_in[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model_field.scene_rep.mlp.layer_in[0].weight)
                cur_model_weights = [cur_grid_latents_weight, cur_grid_submodule_weight, cur_grid_mlp_weight]
                for i in range(len(model_w_types)):
                    if((cur_model_weights[i]==model_weights[i]) and (not optLatentOnly)):
                          print('The ' + model_w_types[i] + ' model weights are not updating!')
                model_weights = cur_model_weights

                # Implement a loss function between the ground truth and model output 
                loss, sdf_loss, sdf_loss_cl, latent_loss, param_loss, lip_loss, gradient_loss, normal_loss = loss_fn((model_output, param_latent), ground_truth_sdf, [model_grid, model_field], loss_param, model_input['xyz'], ground_truth_normals) 
                detailed_losses = [sdf_loss, sdf_loss_cl, latent_loss, param_loss, lip_loss, gradient_loss, normal_loss]
                
                if not torch.isfinite(loss):
                    print(f"Loss is NaN/Inf at step {total_steps}, skipping. g_name={g_name}")
                    continue
                    
                # Append loss and detailed loss information per geometry
                train_losses.append(loss.detach().cpu().numpy())
                l_tr = [model_input['idx'].detach().cpu().numpy(), epoch, i, loss.detach().cpu().numpy()]
                [d_arr[d].append(l_tr[d]) for d in range(len(l_tr))]
                [d_arr[d+4].append(detailed_losses[d].detach().cpu().numpy()) for d in range(len(detailed_losses))]
                    
                # Write summary
                if not total_steps % rep_param['steps_til_summary']:
                    logModelUpdates([model_grid, model_field], ['model_grid','model_field'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'current')
                    summary_fn(total_steps, np.mean(train_losses), [scheduler.get_last_lr()[0]] if(optLatentOnly) else [np.mean(scheduler.get_last_lr()[0:2]),scheduler.get_last_lr()[2]], model_output, [ground_truth_sdf], [model_grid, model_field], rendering_dir, det_losses = detailed_losses, det_losses_types = d_arr_names[4:], det_losses_lambdas = det_losses_lambdas) 

                # One step of optimization
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model_grid.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model_field.parameters(), max_norm=1.0)
                
                optimizer.step()

                total_steps += 1
                
                
            print("Before scheduler:", [group['lr'] for group in optimizer.param_groups])
            scheduler.step()
            print("After scheduler:", [group['lr'] for group in optimizer.param_groups])
            
            # Clamp learning rate manually
            min_lr = 1e-6
            for param_group in optimizer.param_groups:
                if param_group['lr'] < min_lr:
                    param_group['lr'] = min_lr
            
            
            lr_latent = scheduler.get_last_lr()[2] if(not optLatentOnly) else scheduler.get_last_lr()[0]
            lr_model = np.mean(scheduler.get_last_lr()[0:2]) if(not optLatentOnly) else 0 
            print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, model learning rate %0.8f, latent learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, lr_model, lr_latent))

            if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):
                # Save the model and losses at checkpoints
                logModelUpdates([model_grid, model_field], ['model_grid','model_field'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'epoch', epoch = epoch)

    # Save the final model and detailed stats
    logModelUpdates([model_grid, model_field], ['model_grid','model_field'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'final')
        

        
# Fit ML function
def fit_rot(
    model: nn.Module,
    model_dir,
    train_dataloader,
    loss_fn,
    summary_fn,
    plotting_function = None,
    fromCheckPt = None,
    val_dataloader = None,
   ):


    
    # Define data arrays
    train_losses, val_losses = [], []   
    idx_ar, epoch_ar, step_ar, loss_ar, sdf_loss_ar, sdf_clamped_loss_ar, param_loss_ar = [], [], [], [], [], [], []
    d_arr = [idx_ar, epoch_ar, step_ar, loss_ar, sdf_loss_ar, sdf_clamped_loss_ar, param_loss_ar]
    val_idx_ar, val_epoch_ar, val_step_ar, val_loss_ar, val_sdf_loss_ar, val_sdf_clamped_loss_ar, val_param_loss_ar = [], [], [], [], [], [], []
    val_d_arr = [val_idx_ar, val_epoch_ar, val_step_ar, val_loss_ar, val_sdf_loss_ar, val_sdf_clamped_loss_ar, val_param_loss_ar]
    d_arr_names = ['idx', 'epoch', 'step', 'loss', 'sdf_loss', 'sdf_clamped_loss', 'param_loss']
    
    # Setup parallel model if GPUs>1
    if (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)
        print("This training will use", torch.cuda.device_count(), "GPUs.")
    
    # Create model sub-directories
    model_dir, checkpoints_dir, training_viz_dir, rendering_dir = createModelDirs(model_dir)

    # Load model parameters
    mlp_param, loss_param, rep_param, dataset_param = loadModelParam(model_dir, param_ar = ['mlp_param', 'loss_param', 'rep_param', 'dataset_train_param'])
    det_losses_lambdas = [loss_param['sdf_loss_lambda']*(1-loss_param['sdf_loss_clamp_ratio']), loss_param['sdf_loss_lambda']*loss_param['sdf_loss_clamp_ratio'], loss_param['weight_reg_lambda']]

    # Define optimizer
    lr_model = loss_param['lr_sch_initial'] if ((fromCheckPt is None) or (int(fromCheckPt/loss_param['lr_sch_interval'])==0)) else loss_param['lr_sch_initial']*np.power(loss_param['lr_sch_factor'],int(np.floor(fromCheckPt/loss_param['lr_sch_interval'])))
    optimizer = torch.optim.Adam([{"params": model.module.mlp.parameters(),"lr": lr_model}]) if(torch.cuda.device_count() > 1) else torch.optim.Adam([{"params": model.mlp.parameters(),"lr": lr_model}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
    loss = 0

    # Loading from checkpoint
    if(fromCheckPt is not None):
        models, train_losses, d_arr = loadfromCheckpoint([model], ['rot_model'], checkpoints_dir, d_arr, d_arr_names, fromCheckPt)
        model = models[0]
    
    # Define the number of batch steps needed to complete one epoch
    num_steps_per_epoch = len(train_dataloader)
    print('The number of steps per epoch is ' + str(num_steps_per_epoch))
    total_steps = 0

    starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1
    total_epochs = rep_param['total_epochs']
    
    # Set the model to train    
    model.train()

    # Weight updates checks
    model_weights = [0,0]
    model_w_types = ['rot_enc', 'sdf']

    for epoch in range(total_epochs):

        if(epoch >= starting_epoch):

            start_time_epoch = time.time()
            
            # Create train_dataloader_iterator
            train_dataloader_iter = iter(train_dataloader)
            
            for i in range(len(train_dataloader)):
                
                start_time = time.time()

                # Get the next batch of data and move it to the GPU
                model_input, ground_truth_sdf, ground_truth_normals, g_name = next(train_dataloader_iter)
                #print('st_load', time.time()-start_time_epoch)
                if(torch.cuda.device_count()>0):
                    model_input, ground_truth_sdf = to_gpu(model_input), to_gpu(ground_truth_sdf)
                    #print('st_gpu', time.time()-start_time_epoch)

                # Compute the MLP output for the given input data and compute the loss
                model_output = model(model_input)
                
                # Weight updates checks
                cur_weight = torch.sum(model.module.mlp.layer_in[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.mlp.layer_in[0].weight)
                cur_sdf_weight = torch.sum(model.module.latent_sdf_field.mlp.layer_in[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.latent_sdf_field.mlp.layer_in[0].weight)
                cur_model_weights = [cur_weight, cur_sdf_weight]
                if(cur_model_weights[0]==model_weights[0]):
                    print('The ' + model_w_types[0] + ' model weights are potentially not updating!')
                if((cur_model_weights[1]!=model_weights[1])and(model_weights[1]!=0)):
                    print('The ' + model_w_types[1] + ' model weights seem to be updating!')
                model_weights = cur_model_weights

                # Implement a loss function between the ground truth and model output 
                loss, sdf_loss, sdf_loss_cl, param_loss = loss_fn(model_output, ground_truth_sdf, [model], loss_param) 
                detailed_losses = [sdf_loss, sdf_loss_cl, param_loss]
                
                # Append loss and detailed loss information per geometry
                train_losses.append(loss.detach().cpu().numpy())
                l_tr = [model_input['idx'].detach().cpu().numpy(), epoch, i, loss.detach().cpu().numpy()]
                [d_arr[d].append(l_tr[d]) for d in range(len(l_tr))]
                [d_arr[d+4].append(detailed_losses[d].detach().cpu().numpy()) for d in range(len(detailed_losses))]
                
                # Write summary
                if not total_steps % rep_param['steps_til_summary']:
                    logModelUpdates([model], ['rot_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'current')
                    summary_fn(total_steps, np.mean(train_losses), [scheduler.get_last_lr()[0]], model_output, [ground_truth_sdf], [model], det_losses = detailed_losses, det_losses_types = d_arr_names[4:], det_losses_lambdas = det_losses_lambdas, loss_type_names = ['sdf']) 

                # One step of optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_steps += 1
            scheduler.step()
            
            
            # Go through one validation step
            if(val_dataloader is not None):
                model.eval()
                with torch.no_grad():
                    # Get the next batch of data and move it to the GPU
                    model_input, ground_truth_sdf, ground_truth_normals, g_name = next(iter(val_dataloader))
                    if(torch.cuda.device_count()>0):
                        model_input, ground_truth_sdf = to_gpu(model_input), to_gpu(ground_truth_sdf)
                    model_output = model(model_input)
                    val_loss, val_sdf_loss, val_sdf_loss_cl, val_param_loss = loss_fn(model_output, ground_truth_sdf, [model], loss_param) 
                    val_detailed_losses = [val_sdf_loss, val_sdf_loss_cl, val_param_loss]
                    # Append loss and detailed loss information per geometry
                    val_losses.append(val_loss.detach().cpu().numpy())
                    l_val = [model_input['idx'].detach().cpu().numpy(), epoch, i, val_loss.detach().cpu().numpy()]
                    [val_d_arr[d].append(l_val[d]) for d in range(len(l_val))]
                    [val_d_arr[d+4].append(val_detailed_losses[d].detach().cpu().numpy()) for d in range(len(val_detailed_losses))]
                model.train()

                print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, validation loss %0.6f, iteration time %0.6f, model learning rate %0.8f" % (epoch, np.mean(train_losses), np.mean(val_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))

                if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):
                    # Save the model and losses at checkpoints
                    logModelUpdates([model], ['rot_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'epoch', epoch = epoch, val_losses = np.array(val_losses).flatten(), val_d_arr = val_d_arr)
            else:
                print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, model learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))

                if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):
                    # Save the model and losses at checkpoints
                    logModelUpdates([model], ['rot_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'epoch', epoch = epoch)
            
    # Save the final model and detailed stats
    if(val_dataloader is not None):
        logModelUpdates([model], ['rot_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'final', val_losses = np.array(val_losses).flatten(), val_d_arr = val_d_arr) 
    else:
        logModelUpdates([model], ['rot_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'final') 
    

    
    
    
    
        
        
        
        
# Fit ML function
def fit_perf(
    model: nn.Module,
    model_dir,
    train_dataloader,
    loss_fn,
    summary_fn,
    plotting_function = None,
    fromCheckPt = None,
    val_dataloader = None,
   ):

    # Define data arrays
    train_losses, val_losses = [], []
    idx_ar, epoch_ar, step_ar, loss_ar, perf_loss_ar, cos_sim_loss_ar, param_loss_ar = [], [], [], [], [], [], []
    d_arr = [idx_ar, epoch_ar, step_ar, loss_ar, perf_loss_ar, cos_sim_loss_ar, param_loss_ar]
    val_idx_ar, val_epoch_ar, val_step_ar, val_loss_ar, val_perf_loss_ar, val_cos_sim_loss_ar, val_param_loss_ar = [], [], [], [], [], [], []
    val_d_arr = [val_idx_ar, val_epoch_ar, val_step_ar, val_loss_ar, val_perf_loss_ar, val_cos_sim_loss_ar, val_param_loss_ar]
    d_arr_names = ['idx', 'epoch', 'step', 'loss', 'perf_loss', 'cos_sim_loss', 'param_loss']

    # Setup parallel model if GPUs>1
    if (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)
        print("This training will use", torch.cuda.device_count(), "GPUs.")
    elif(torch.cuda.device_count() == 1):
        print("This training will use 1 GPU.")
    else:
        print("This training will run on CPU.")
    
    # Create model sub-directories
    model_dir, checkpoints_dir, training_viz_dir, rendering_dir = createModelDirs(model_dir)

    # Load model parameters
    ctxtmod_param, spatenc_param, perf_param, perfdec_param, loss_param, rep_param, dataset_train_param, dataset_val_param, dataset_test_param = loadModelParam(model_dir, param_ar = ['ctxtmod_param', 'spatenc_param', 'perf_param', 'perfdec_param', 'loss_param', 'rep_param', 'dataset_train_param', 'dataset_val_param', 'dataset_test_param'])
    dataset_ids = dataset_train_param['dataset_ids']
    det_losses_lambdas = [loss_param['perf_loss_lambda'], loss_param['cos_sim_lambda'], loss_param['weight_reg_lambda']]
    if(spatenc_param is not None):
        perf_type = 'ground'
    elif(perf_param is not None):
        perf_type = 'hmap'
    else:
        perf_type = 'field'

    # Check if model containts a context encoder
    count = 0
    for name, p in model.named_parameters():
        count = (count + 1) if(name.startswith('ctxtmod_autoenc')) else count
    include_enc = True if(count>0) else False

    # Define optimizer
    lr_model = loss_param['lr_sch_initial'] if ((fromCheckPt is None) or (int(fromCheckPt/loss_param['lr_sch_interval'])==0)) else loss_param['lr_sch_initial']*np.power(loss_param['lr_sch_factor'],int(np.floor(fromCheckPt/loss_param['lr_sch_interval'])))
    if(perf_type == 'ground'):
        optimizer = torch.optim.Adam([{"params": model.module.encoder_model.parameters(),"lr": lr_model},{"params": model.module.perf_dec.parameters(),"lr": lr_model}]) if(torch.cuda.device_count() > 1) else torch.optim.Adam([{"params": model.encoder_model.parameters(),"lr": lr_model},{"params": model.perf_dec.parameters(),"lr": lr_model}])  
    elif(perf_type == 'hmap'):
        optimizer = torch.optim.Adam([{"params": model.module.encoder_model.parameters(),"lr": lr_model}]) if(torch.cuda.device_count() > 1) else torch.optim.Adam([{"params": model.encoder_model.parameters(),"lr": lr_model}])  
    elif(include_enc):
        optimizer = torch.optim.Adam([{"params": model.module.ctxtmod_autoenc.parameters(),"lr": lr_model},{"params": model.module.perf_dec.parameters(),"lr": lr_model}]) if(torch.cuda.device_count() > 1) else torch.optim.Adam([{"params": model.ctxtmod_autoenc.parameters(),"lr": lr_model},{"params": model.perf_dec.parameters(),"lr": lr_model}])
    else:
        optimizer = torch.optim.Adam([{"params": model.module.perf_dec.parameters(),"lr": lr_model}]) if(torch.cuda.device_count() > 1) else torch.optim.Adam([{"params": model.perf_dec.parameters(),"lr": lr_model}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
    loss = 0

    # Loading from checkpoint
    if(fromCheckPt is not None):
        if(val_dataloader is not None):
            models, train_losses, d_arr, val_losses, val_d_arr = loadfromCheckpoint([model], ['perf_model'], checkpoints_dir, d_arr, d_arr_names, fromCheckPt, d_arr_val = val_d_arr)
        else:
            models, train_losses, d_arr = loadfromCheckpoint([model], ['perf_model'], checkpoints_dir, d_arr, d_arr_names, fromCheckPt)
        model = models[0]
    
    # Define the number of batch steps needed to complete one epoch
    num_steps_per_epoch = len(train_dataloader)
    print('The number of steps per epoch is ' + str(num_steps_per_epoch))
    total_steps = 0

    starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1
    total_epochs = rep_param['total_epochs']
    
    # Set the model to train    
    model.train()

    # Weight updates checks
    model_weights = [0,0]
    model_w_types = ['ctxtmod', 'perf'] if(include_enc) else ['perf']

    for epoch in range(total_epochs):

        if(epoch >= starting_epoch):

            start_time_epoch = time.time()
            
            # Create train_dataloader_iterator
            train_dataloader_iter = iter(train_dataloader)
            
            for i in range(len(train_dataloader)):
                
                start_time = time.time()

                # Get the next batch of data and move it to the GPU
                model_input, ground_truth_sdf, ground_truth_perf = next(train_dataloader_iter)
                if(torch.cuda.device_count()>0):
                    model_input, ground_truth_sdf, ground_truth_perf = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_perf)

                # Compute the MLP output for the given input data and compute the loss
                model_output = model(model_input)
                
                # # Weight updates checks
                # if(include_enc):
                #     cur_ctxtmod_weight = torch.sum(model.module.ctxtmod_autoenc.layer_in_enc[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.ctxtmod_autoenc.layer_in_enc[0].weight)
                # if(perfdec_param['weight_norm']):
                #     try:
                #         cur_perf_weight = torch.sum(model.module.perf_dec.layer_in[0].module.weight_g) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_dec.layer_in[0].module.weight_g)
                #     except:
                #          cur_perf_weight = torch.sum(model.module.perf_dec.layer_in[0].module.weight) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_dec.layer_in[0].module.weight)
                # else:
                #     cur_perf_weight = torch.sum(model.module.perf_dec.layer_in[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_dec.layer_in[0].weight)
                # cur_model_weights = [cur_ctxtmod_weight, cur_perf_weight] if(include_enc) else [cur_perf_weight]
                # for i in range(len(model_w_types)):
                #     if(cur_model_weights[i]==model_weights[i]):
                #           print('The ' + model_w_types[i] + ' model weights are potentially not updating!')
                # model_weights = cur_model_weights

                # Implement a loss function between the ground truth and model output 
                loss, perf_loss, cos_sim_loss, param_loss = loss_fn(model_output, ground_truth_perf, ground_truth_sdf, [model], loss_param, model_input['xyz']) 
                detailed_losses = [perf_loss, cos_sim_loss, param_loss]
                
                # Append loss and detailed loss information per geometry
                train_losses.append(loss.detach().cpu().numpy())
                l_tr = [model_input['idx'].detach().cpu().numpy(), epoch, i, loss.detach().cpu().numpy()]
                [d_arr[d].append(l_tr[d]) for d in range(len(l_tr))]
                [d_arr[d+4].append(detailed_losses[d].detach().cpu().numpy()) for d in range(len(detailed_losses))]
                
                # Write summary
                if not total_steps % rep_param['steps_til_summary']:
                    logModelUpdates([model], ['perf_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'current')
                    summary_fn(total_steps, np.mean(train_losses), [scheduler.get_last_lr()[0]], model_output, [ground_truth_perf], [model], det_losses = detailed_losses, det_losses_types = d_arr_names[4:], det_losses_lambdas = det_losses_lambdas, loss_type_names = ['perf']) 

                # One step of optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_steps += 1
            scheduler.step()
            
            # Go through one validation step
            if(val_dataloader is not None):
                model.eval()
                with torch.no_grad():
                    # Get the next batch of data and move it to the GPU
                    # try:
                    #     if(model.perf_dec.hidden_layers_skip[0][2].training):
                    #         print('Dropout is training during validation!!: ' + str(model.perf_dec.hidden_layers_skip[0][2].training))
                    # except:
                    #     if(model.module.perf_dec.hidden_layers_skip[0][2].training):
                    #         print('Dropout is training during validation: ' + str(model.module.perf_dec.hidden_layers_skip[0][2].training))
                            
                    # Create val_dataloader_iterator
                    val_dataloader_iter = iter(val_dataloader)
                    for i in range(len(val_dataloader)):
                        model_input, ground_truth_sdf, ground_truth_perf = next(val_dataloader_iter)
                        if(torch.cuda.device_count()>0):
                            model_input, ground_truth_sdf, ground_truth_perf = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_perf)
                        model_output = model(model_input)
                        val_loss, val_perf_loss, val_cos_sim_loss, val_param_loss = loss_fn(model_output, ground_truth_perf, ground_truth_sdf, [model], loss_param, model_input['xyz']) 
                        val_detailed_losses = [val_perf_loss, val_cos_sim_loss, val_param_loss]
                        # Append loss and detailed loss information per geometry
                        val_losses.append(val_loss.detach().cpu().numpy())
                        l_val = [model_input['idx'].detach().cpu().numpy(), epoch, i, val_loss.detach().cpu().numpy()]
                        [val_d_arr[d].append(l_val[d]) for d in range(len(l_val))]
                        [val_d_arr[d+4].append(val_detailed_losses[d].detach().cpu().numpy()) for d in range(len(val_detailed_losses))]
                model.train()

                print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, validation loss %0.6f, iteration time %0.6f, model learning rate %0.8f" % (epoch, np.mean(train_losses), np.mean(val_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))

                if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):
                    # Save the model and losses at checkpoints
                    logModelUpdates([model], ['perf_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'epoch', epoch = epoch, val_losses = np.array(val_losses).flatten(), val_d_arr = val_d_arr)
            else:
                print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, model learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))

                if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):
                    # Save the model and losses at checkpoints
                    logModelUpdates([model], ['perf_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'epoch', epoch = epoch)

    # Save the final model and detailed stats
    if(val_dataloader is not None):
        logModelUpdates([model], ['perf_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'final', val_losses = np.array(val_losses).flatten(), val_d_arr = val_d_arr) 
    else:
        logModelUpdates([model], ['perf_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'final') 
        
        
# Fit ML function
def fit_perf_cdf(
    model: nn.Module,
    model_dir,
    train_dataset,
    loss_fn,
    summary_fn,
    plotting_function = None,
    fromCheckPt = None,
    val_dataset = None,
   ):

    # Define data arrays
    train_losses, val_losses = [], []
    idx_ar, epoch_ar, step_ar, loss_ar, dist_loss_ar, param_loss_ar = [], [], [], [], [], []
    d_arr = [idx_ar, epoch_ar, step_ar, loss_ar, dist_loss_ar, param_loss_ar]
    val_idx_ar, val_epoch_ar, val_step_ar, val_loss_ar, val_dist_loss_ar, val_param_loss_ar = [], [], [], [], [], []
    val_d_arr = [val_idx_ar, val_epoch_ar, val_step_ar, val_loss_ar, val_dist_loss_ar, val_param_loss_ar]
    d_arr_names = ['idx', 'epoch', 'step', 'loss', 'dist_loss', 'param_loss']
    
    #loss, dist_loss, param_loss
    
    
    # Setup parallel model if GPUs>1
    if (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)
        print("This training will use", torch.cuda.device_count(), "GPUs.")
    
    # Create model sub-directories
    model_dir, checkpoints_dir, training_viz_dir, rendering_dir = createModelDirs(model_dir)

    # Load model parameters
    ctxtmod_param, perf_com_param, perf_sep_param, loss_param, rep_param, dataset_train_param, dataset_val_param, dataset_test_param = loadModelParam(model_dir, param_ar = ['ctxtmod_param', 'perf_com_param', 'perf_sep_param', 'loss_param', 'rep_param', 'dataset_train_param', 'dataset_val_param', 'dataset_test_param'])
    dataset_ids = dataset_train_param['dataset_ids']
    det_losses_lambdas = [loss_param['dist_loss_lambda'], loss_param['weight_reg_lambda']]
    
    # Check if model containts a context encoder
    count = 0
    for name, p in model.named_parameters():
        count = (count + 1) if(name.startswith('ctxtmod_autoenc')) else count
    include_enc = True if(count>0) else False

    # Define optimizer
    lr_model = loss_param['lr_sch_initial'] if ((fromCheckPt is None) or (int(fromCheckPt/loss_param['lr_sch_interval'])==0)) else loss_param['lr_sch_initial']*np.power(loss_param['lr_sch_factor'],(int(fromCheckPt/loss_param['lr_sch_interval'])))
    if(include_enc):
        m_params = [{"params": model.module.ctxtmod_autoenc.parameters(),"lr": lr_model},{"params": model.module.perf_com.parameters(),"lr": lr_model}] if(torch.cuda.device_count() > 1) else [{"params": model.ctxtmod_autoenc.parameters(),"lr": lr_model},{"params": model.perf_com.parameters(),"lr": lr_model}]
        p_sep_models = model.module.perf_sep if(torch.cuda.device_count() > 1) else model.perf_sep
        for m in p_sep_models:
            m_params.append({"params": m.parameters(),"lr": lr_model}) 
        optimizer = torch.optim.Adam(m_params)
    else:
        m_params = [{"params": model.module.perf_com.parameters(),"lr": lr_model}] if(torch.cuda.device_count() > 1) else [{"params": model.perf_com.parameters(),"lr": lr_model}]
        p_sep_models = model.module.perf_sep if(torch.cuda.device_count() > 1) else model.perf_sep
        for m in p_sep_models:
            m_params.append({"params": m.parameters(),"lr": lr_model}) 
        optimizer = torch.optim.Adam(m_params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
    loss = 0

    # Loading from checkpoint
    if(fromCheckPt is not None):
        models, train_losses, d_arr = loadfromCheckpoint([model], ['perf_model'], checkpoints_dir, d_arr, d_arr_names, fromCheckPt)
        model = models[0]

    starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1
    total_epochs = rep_param['total_epochs']
    
    # Set the model to train    
    model.train()

    # Weight updates checks
    model_weights = [0,0,0]  
    model_w_types = ['ctxtmod', 'perf_1', 'perf_2'] if(include_enc) else ['perf_1', 'perf_2']
    if(perf_sep_param['n_networks']==2):
        model_weights.append(0)
        model_w_types.append('perf_3')
    
    for epoch in range(total_epochs):

        if(epoch >= starting_epoch):

            start_time_epoch = time.time()

            # Get the next batch of data and move it to the GPU
            model_input, ground_truth_cdf = train_dataset.__getFullDataset__()
            if(torch.cuda.device_count()>0):
                model_input, ground_truth_cdf = to_gpu(model_input), to_gpu(ground_truth_cdf)

            # Compute the MLP output for the given input data and compute the loss
            model_output = model(model_input)
                
            # Weight updates checks
            if(include_enc):
                cur_ctxtmod_weight = torch.sum(model.module.ctxtmod_autoenc.layer_in[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.ctxtmod_autoenc.layer_in[0].weight)
            if(perf_com_param['weight_norm']):
                try:
                    cur_perf_weight_1 = torch.sum(model.module.perf_com.layer_in[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_com.layer_in[0].weight)
                    cur_perf_weight_2 = torch.sum(model.module.perf_sep[0].layer_in[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_sep[0].layer_in[0].weight)
                    if(perf_sep_param['n_networks']==2):
                        cur_perf_weight_3 = torch.sum(model.module.perf_sep[1].layer_in[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_sep[1].layer_in[0].weight)
                except:
                    cur_perf_weight_1 = torch.sum(model.module.perf_com.layer_in[0].module.weight_g) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_com.layer_in[0].module.weight_g)
                    cur_perf_weight_2 = torch.sum(model.module.perf_sep[0].layer_in[0].module.weight_g) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_sep[0].layer_in[0].module.weight_g)
                    if(perf_sep_param['n_networks']==2):
                        cur_perf_weight_3 = torch.sum(model.module.perf_sep[1].layer_in[0].module.weight_g) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_sep[1].layer_in[0].module.weight_g)
            else:
                cur_perf_weight_1 = torch.sum(model.module.perf_com.layer_in[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_com.layer_in[0].weight)
                cur_perf_weight_2 = torch.sum(model.module.perf_sep[0].layer_in[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_sep[0].layer_in[0].weight)
                if(perf_sep_param['n_networks']==2):
                    cur_perf_weight_3 = torch.sum(model.module.perf_sep[1].layer_in[0].weight) if(torch.cuda.device_count() > 1) else torch.sum(model.perf_sep[1].layer_in[0].weight)
            cur_model_weights = [cur_ctxtmod_weight, cur_perf_weight_1, cur_perf_weight_2] if(include_enc) else [cur_perf_weight_1, cur_perf_weight_2]
            if(perf_sep_param['n_networks']==2):
                cur_model_weights.append(cur_perf_weight_3)
            for i in range(len(model_w_types)):
                if(cur_model_weights[i]==model_weights[i]):
                        print('The ' + model_w_types[i] + ' model weights are potentially not updating!')
            model_weights = cur_model_weights

            # Implement a loss function between the ground truth and model output 
            loss, dist_loss, param_loss = loss_fn(model_output, ground_truth_cdf, [model], loss_param) 
                
            detailed_losses = [dist_loss, param_loss]
                
            # Append loss and detailed loss information per geometry
            train_losses.append(loss.detach().cpu().numpy())
            l_tr = [model_input['idx'].detach().cpu().numpy(), epoch, i, loss.detach().cpu().numpy()]
            [d_arr[d].append(l_tr[d]) for d in range(len(l_tr))]
            [d_arr[d+4].append(detailed_losses[d].detach().cpu().numpy()) for d in range(len(detailed_losses))]
                
            # Write summary
            # print(torch.sum(model.perf_com.layer_in[0].weight))
            # print(torch.sum(model.perf_sep[0].layer_in[0].weight))
            # print(torch.sum(model.ctxtmod_autoenc.layer_in[0].weight))
            # print(torch.sum(model.ctxtmod_autoenc.layer_out[0].weight))
            logModelUpdates([model], ['perf_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'current')
            gt_diff = torch.diff(ground_truth_cdf, prepend=torch.zeros(ground_truth_cdf.shape[0],ground_truth_cdf.shape[1],1))
            summary_fn(epoch, np.mean(train_losses), [scheduler.get_last_lr()[0]], model_output, [gt_diff], [model], det_losses = detailed_losses, det_losses_types = d_arr_names[4:], det_losses_lambdas = det_losses_lambdas, loss_type_names = ['perf'], perf_first_element=False) 
            #print('cdf',ground_truth_cdf)
            # print('min',np.min(gt_diff.detach().cpu().numpy()))
            # print('max',np.max(gt_diff.detach().cpu().numpy()))
            # One step of optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Go through one validation step
            if(val_dataset is not None):
                model.eval()
                with torch.no_grad():
                    # Get the next batch of data and move it to the GPU
                    model_input, ground_truth_cdf = val_dataset.__getFullDataset__()
                    if(torch.cuda.device_count()>0):
                        model_input, ground_truth_cdf = to_gpu(model_input), to_gpu(ground_truth_cdf)
                    model_output = model(model_input)
                    #print('val_gt',ground_truth_cdf.shape)
                    #print('val_output',model_output[0].shape)
                    val_loss, val_dist_loss, val_param_loss = loss_fn(model_output, ground_truth_cdf, [model], loss_param) 
                    val_detailed_losses = [val_dist_loss, val_param_loss]
                    # Append loss and detailed loss information per geometry
                    val_losses.append(val_loss.detach().cpu().numpy())
                    l_val = [model_input['idx'].detach().cpu().numpy(), epoch, i, val_loss.detach().cpu().numpy()]
                    [val_d_arr[d].append(l_val[d]) for d in range(len(l_val))]
                    [val_d_arr[d+4].append(val_detailed_losses[d].detach().cpu().numpy()) for d in range(len(val_detailed_losses))]
                model.train()
                print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, validation loss %0.6f, iteration time %0.6f, model learning rate %0.8f" % (epoch, np.mean(train_losses), np.mean(val_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))
                if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):
                    # Save the model and losses at checkpoints
                    logModelUpdates([model], ['perf_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'epoch', epoch = epoch, val_losses = np.array(val_losses).flatten(), val_d_arr = val_d_arr)
            else:
                print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, model learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))
                if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):
                    # Save the model and losses at checkpoints
                    logModelUpdates([model], ['perf_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'epoch', epoch = epoch)

    # Save the final model and detailed stats
    if(val_dataloader is not None):
        logModelUpdates([model], ['perf_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'final', val_losses = np.array(val_losses).flatten(), val_d_arr = val_d_arr) 
    else:
        logModelUpdates([model], ['perf_model'], np.array(train_losses).flatten(), d_arr, d_arr_names, model_dir, 'final') 
        
        
        
        
        
# # Fit ML function
# def fit_perf(
#     model: nn.Module,
#     model_dir,
#     train_dataloader,
#     loss_fn,
#     trained_sdf_model,
#     loss_param,
#     dataset_param,
#     rep_param,
#     summary_fn,
#     plotting_function,
#     optLatentOnly = False,
#     fromCheckPt = None
#    ):

#     train_losses = []   
#     idx_ar, epoch_ar, step_ar, loss_ar, perf_loss_ar, latent_loss_rg_ar, param_loss_rg_ar, lip_loss_rg_ar = [], [], [], [], [], [], [], []   
    
#     # Setup parallel model if GPUs>1
#     if (torch.cuda.device_count() > 1) and (model_dir.split('/')[-1] != 'test'):
#         model = torch.nn.DataParallel(model)
#         print("This training will use", torch.cuda.device_count(), "GPUs.")

#     # Create model directory
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     # Create checkpoints directory
#     checkpoints_dir = model_dir + '/' + 'checkpoints'
#     if not os.path.exists(checkpoints_dir):
#         os.makedirs(checkpoints_dir)

#     # Create training progress visualization directory
#     training_viz_dir = model_dir + '/' + 'training_viz'
#     if not os.path.exists(training_viz_dir):
#         os.makedirs(training_viz_dir)
    
#     # Get rendering directory
#     rendering_dir = '/'.join(model_dir.split('/')[:-3]).replace('Experiments','Renders') + '/' + '_'.join(model_dir.split('/')[-3].split('_')[0:2]) + '/' + '_'.join(model_dir.split('/')[-3].split('_')[2:4])

#     # Define optimizer
#     if(not optLatentOnly):
#         optimizer = torch.optim.Adam(lr=loss_param['lr_sch_initial'], params=model.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_param['lr_sch_initial']*loss_param['lr_sch_factor']*(fromCheckPt/loss_param['lr_sch_interval']), params=model.parameters())
#     else:
#         for name, param in model.named_parameters():
#             if param.requires_grad and ('latent' not in name):
#                 param.requires_grad = False
#         if torch.cuda.device_count() > 1:
#             optimizer = torch.optim.Adam(lr=loss_param['lr_sch_initial'], params=model.module.latents.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_param['lr_sch_initial']*loss_param['lr_sch_factor']*(fromCheckPt/loss_param['lr_sch_interval']), params=model.module.latents.parameters())
#         else:
#             optimizer = torch.optim.Adam(lr=loss_param['lr_sch_initial'], params=model.latents.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_param['lr_sch_initial']*loss_param['lr_sch_factor']*(fromCheckPt/loss_param['lr_sch_interval']), params=model.latents.parameters())
    
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
#     loss = 0

#     if(fromCheckPt is not None):
#         #print(np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt))).shape)
#         model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % fromCheckPt)))
#         model.train()
#         train_check = np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt))).flatten().astype(float)
#         train_check_det = json.loads(open(checkpoints_dir + '/' + 'train_losses_detailed_epoch_%04d.json' % fromCheckPt).read())
#         idx_ar = train_check_det['idx']
#         epoch_ar = train_check_det['epoch']
#         step_ar = train_check_det['step']
#         loss_ar = train_check_det['loss']
#         perf_loss_ar = train_check_det['perf_loss']
#         latent_loss_rg_ar  = train_check_det['latent_loss_rg']
#         param_loss_rg_ar = train_check_det['param_loss_rg']
#         lip_loss_rg_ar = train_check_det['lip_loss_rg']
        

#     # Get XYZ for reconstruction
#     XYZ_coord = get3DGrid(rep_param['mc_resolution'])
    
#     # Get bounding box
#     buffered_bbox = np.array(rep_param['buffered_bbox'])
#     #print(buffered_bbox)
#     #print(buffered_bbox[:,1])
#     #print(buffered_bbox[:,0])
    
#     trained_sdf_model.eval()
#     try:
#         sdf_latents = trained_sdf_model.latents
#     except:
#         sdf_latents = trained_sdf_model.module.latents
#     sdf_latents = sdf_latents.weight
    
#    # print('sdf_latents',sdf_latents.shape)

#     num_steps_per_epoch = len(train_dataloader)*dataset_param['sets_perbatch']
#     print('The number of steps per epoch is ' + str(num_steps_per_epoch))
#     total_steps = 0

#     starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1


#     for epoch in range(rep_param['total_epochs']):

#         if(epoch >= starting_epoch):

#             start_time_epoch = time.time()
                
#             for i in range(num_steps_per_epoch):
#                 start_time = time.time()

#                 # Get the next batch of data and move it to the GPU
#                 model_input, ground_truth_sdf, ground_truth_normals, ground_truth_perf = next(iter(train_dataloader))
#                 model_input, ground_truth_sdf, ground_truth_normals, ground_truth_perf = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_normals), to_gpu(ground_truth_perf)

#                 # Compute the MLP output for the given input data and compute the loss
#                 sdf_latents = to_gpu(sdf_latents)
#                 #sdf_latents.requires_grad = False
                
#                 model_output = model(model_input, sdf_latents)
#                 print(model_output[0].device)
                
#                 print("Outside: input size", model_input['xyz'].size(),"output_size", model_output[0].size())

#                 # Implement a loss function between the ground truth and model output 
#                 loss, perf_loss, latent_loss_rg, param_loss_rg, lip_loss_rg = loss_fn(model_output, ground_truth_perf, model, loss_param) 
#                 # print('loss', loss)
#                 # print('perf_loss', perf_loss)
#     #           print('latent_loss_rg', latent_loss_rg)
#     #           print('param_loss_rg', param_loss_rg)
#     #           print('lip_loss_rg', lip_loss_rg)
#                 if((fromCheckPt is not None) and (epoch==starting_epoch)):
#                     #print(np.array(train_check).shape)
#                     #print(np.expand_dims(loss.detach().cpu().numpy(),0).shape)
#                     #train_losses.append(np.concatenate([train_check,np.expand_dims(loss.detach().cpu().numpy(),0)])) 
#                     train_losses.append(loss.detach().cpu().numpy().flatten())
#                 train_losses.append(loss.detach().cpu().numpy())
#                 # Save detailed loss information per geometry
#                 idx_ar.append(model_input['idx'].detach().cpu().numpy())
#                 epoch_ar.append(epoch)
#                 step_ar.append(i)
#                 loss_ar.append(loss.detach().cpu().numpy())
#                 perf_loss_ar.append(perf_loss.detach().cpu().numpy())
#                 latent_loss_rg_ar.append(latent_loss_rg.detach().cpu().numpy())
#                 param_loss_rg_ar.append(param_loss_rg.detach().cpu().numpy())
#                 lip_loss_rg_ar.append(lip_loss_rg.detach().cpu().numpy())
                
#                 # print(np.mean(train_losses))
#                 # print(np.array(train_losses).shape)
#                 #print(train_losses)loss_param['sdf_loss_lambda']

#                 if not total_steps % rep_param['steps_til_summary']:
#                     torch.save(model.state_dict(),os.path.join(model_dir, 'model_current.pth'))
#                     np.savetxt(os.path.join(model_dir, 'epoch_current.txt'),np.array([epoch]))
#                     np.savetxt(os.path.join(model_dir, 'train_losses_current.txt'),np.array(train_losses).flatten())
#                     summary_fn(total_steps, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model, rendering_dir, det_losses = [perf_loss, latent_loss_rg, param_loss_rg, lip_loss_rg], det_losses_types = ['perf_loss', 'latent_loss', 'param_loss', 'lip_loss'], det_losses_lambdas = [loss_param['perf_loss_lambda'], loss_param['latent_reg_lambda'], loss_param['weight_reg_lambda'], loss_param['lipschitz_reg_lambda']]) 
                    
#                     train_losses_detailed =  dict({'idx': idx_ar, 'epoch': epoch_ar, 'step': step_ar, 'loss': loss_ar, 'perf_loss': perf_loss_ar, 'latent_loss_rg': latent_loss_rg_ar, 'param_loss_rg': param_loss_rg_ar, 'lip_loss_rg': lip_loss_rg_ar})
#                     with open(model_dir + '/' + 'train_losses_detailed_current.json', 'w') as f:
#                         json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)

#                 # One step of optimization
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 total_steps += 1
#                 scheduler.step()

#             print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))

#             if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):

#                 # Save the model and losses at checkpoints
#                 torch.save(model.state_dict(),os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
#                 np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),np.array(train_losses))
                
#                 train_losses_detailed =  dict({'idx': idx_ar, 'epoch': epoch_ar, 'step': step_ar, 'loss': loss_ar, 'perf_loss': perf_loss_ar, 'latent_loss_rg': latent_loss_rg_ar, 'param_loss_rg': param_loss_rg_ar, 'lip_loss_rg': lip_loss_rg_ar})
#                 with open(checkpoints_dir + '/' + 'train_losses_detailed_epoch_%04d.json' % epoch, 'w') as f:
#                     json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)

#                 # Plot summary stats and visual reconstructions
#                 model.eval()
#                 with torch.no_grad():
#                     rand_index = np.random.randint(0,len(dataset_param['bld_dataset_ind'])/2)
#                      #print(len(dataset_param['bld_dataset_ind']))
#                      #print(rand_index)
#                      #print(dataset_param['bld_dataset_ind'])
#                      #rand_index = 0
#                     act_index = dataset_param['bld_dataset_ind'][rand_index]
#                     typeLoss = 'test ' if optLatentOnly else 'train '
#                     summary_fn(total_steps, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model, rendering_dir, resolution=rep_param['mc_resolution'], XYZ_coord=XYZ_coord,  index=rand_index, act_index=act_index, viz_dir=training_viz_dir, plotting_function=plotting_function, buffered_bbox=buffered_bbox, typeLoss='') 
#                     model = model.to(device)
                    
#                 model.train()

#     torch.save(model.state_dict(),os.path.join(model_dir, 'model_final.pth'))
#     np.savetxt(os.path.join(model_dir, 'train_losses_final.txt'),np.array(train_losses).flatten())
    
#     train_losses_detailed =  dict({'idx': idx_ar, 'epoch': epoch_ar, 'step': step_ar, 'loss': loss_ar, 'perf_loss': perf_loss_ar, 'latent_loss_rg': latent_loss_rg_ar, 'param_loss_rg': param_loss_rg_ar, 'lip_loss_rg': lip_loss_rg_ar})
#     with open(model_dir + '/' + 'train_losses_detailed_final.json', 'w') as f:
#         json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)

        
        
        
        
        

# # Fit ML function
# def fit_perf(
#     model: nn.Module,
#     model_dir,
#     train_dataloader,
#     loss_fn,
#     trained_sdf_model,
#     loss_param,
#     dataset_param,
#     rep_param,
#     summary_fn,
#     plotting_function,
#     optLatentOnly = False,
#     fromCheckPt = None
#    ):

#     train_losses = []

#     # Create model directory
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     # Create checkpoints directory
#     checkpoints_dir = model_dir + '/' + 'checkpoints'
#     if not os.path.exists(checkpoints_dir):
#         os.makedirs(checkpoints_dir)

#     # Create training progress visualization directory
#     training_viz_dir = model_dir + '/' + 'training_viz'
#     if not os.path.exists(training_viz_dir):
#         os.makedirs(training_viz_dir)
    
#     # Get rendering directory
#     rendering_dir = '/'.join(model_dir.split('/')[:-3]).replace('Experiments','Renders') + '/' + '_'.join(model_dir.split('/')[-3].split('_')[0:2]) + '/' + '_'.join(model_dir.split('/')[-3].split('_')[2:4])

#     # Define optimizer
#     if(not optLatentOnly):
#         optimizer = torch.optim.Adam(lr=loss_param['lr_sch_initial'], params=model.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_param['lr_sch_initial']*loss_param['lr_sch_factor']*(fromCheckPt/loss_param['lr_sch_interval']), params=model.parameters())
#     else:
#         for name, param in model.named_parameters():
#             if param.requires_grad and ('latent' not in name):
#                 param.requires_grad = False
#         optimizer = torch.optim.Adam(lr=loss_param['lr_sch_initial'], params=model.latents.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_param['lr_sch_initial']*loss_param['lr_sch_factor']*(fromCheckPt/loss_param['lr_sch_interval']), params=model.latents.parameters())
    
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
#     loss = 0.

#     if(fromCheckPt is not None):
#         #print(np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt))).shape)
#         model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % fromCheckPt)))
#         model.train()
#         train_check = np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt)),dtype=object).flatten()

#     # Get XYZ for reconstruction
#     XYZ_coord = get3DGrid(rep_param['mc_resolution'])
    
#     # Get bounding box
#     buffered_bbox = np.array(rep_param['buffered_bbox'])
#     #print(buffered_bbox)
#     #print(buffered_bbox[:,1])
#     #print(buffered_bbox[:,0])

#     num_steps_per_epoch = len(train_dataloader)*dataset_param['sets_perbatch']
#     #print('The number of steps per epoch is ' + str(num_steps_per_epoch))
#     total_steps = 0
    
#     trained_sdf_model.eval()
#     sdf_latents = trained_sdf_model.latents
#    # print('sdf_latents',sdf_latents.shape)

#     starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1

#     for epoch in range(rep_param['total_epochs']):

#         if(epoch >= starting_epoch):
#             #print('epoch', epoch)

#             start_time_epoch = time.time()
                
#             for i in range(num_steps_per_epoch):
#                 start_time = time.time()
#                 #print('i',i)

#                 # Get the next batch of data and move it to the GPU
#                 model_input, ground_truth_sdf, ground_truth_perf = next(iter(train_dataloader))
#                 model_input, ground_truth_sdf, ground_truth_perf = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_perf)

#                 # Compute the MLP output for the given input data and compute the loss

#                 model_output = model(model_input, sdf_latents)
#                 #print('model_output', model_output[0].shape)
#                 #print('model_output', model_output)
                
#                 # Implement a loss function between the ground truth and model output 
#                 loss, sdf_loss, latent_loss_rg, param_loss_rg, lip_loss_rg = loss_fn(model_output, ground_truth_perf, model, loss_param) 
#                 #print('sdf_loss', sdf_loss)
#                 #print('latent_loss_rg', latent_loss_rg)
#                 #print('param_loss_rg', param_loss_rg)
#                 #print('lip_loss_rg', lip_loss_rg)
#                 if((fromCheckPt is not None) and (epoch==starting_epoch)):
#                       train_losses.append(np.concatenate([train_check,loss.detach().cpu().numpy()])) 
#                 train_losses.append(loss.detach().cpu().numpy())

#                 if not total_steps % rep_param['steps_til_summary']:
#                     torch.save(model.state_dict(),os.path.join(model_dir, 'model_current.pth'))
#                     np.savetxt(os.path.join(model_dir, 'epoch_current.txt'),np.array([epoch]))
#                     np.savetxt(os.path.join(model_dir, 'train_losses_current.txt'),np.array(train_losses,dtype=object))
#                     summary_fn(i, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model) 
                

#             # One step of optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_steps += 1
#             scheduler.step()

#             print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, learning rate %0.6f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))

#             if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):

#                 # Save the model and losses at checkpoints
#                 torch.save(model.state_dict(),os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
#                 np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),np.array(train_losses))

#                 # Plot summary stats and visual reconstructions
#                 model.eval()
#                 with torch.no_grad():
#                     rand_index = np.random.randint(0,len(dataset_param['bld_dataset_ind'])/2)
#                      #print(len(dataset_param['bld_dataset_ind']))
#                      #print(rand_index)
#                      #print(dataset_param['bld_dataset_ind'])
#                      #rand_index = 0
#                     act_index = dataset_param['bld_dataset_ind'][rand_index]
#                     typeLoss = 'test ' if optLatentOnly else 'train '
#                     summary_fn(i, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model, rendering_dir, resolution=rep_param['mc_resolution'], XYZ_coord=XYZ_coord,  index=rand_index, act_index=act_index, viz_dir=training_viz_dir, plotting_function=plotting_function, buffered_bbox=buffered_bbox, typeLoss='') 
#                     model = model.to(device)
#                 model.train()

#     torch.save(model.state_dict(),os.path.join(model_dir, 'model_final.pth'))
#     np.savetxt(os.path.join(model_dir, 'train_losses_final.txt'),np.array(train_losses))
    
    

# Fit ML function
def fit_joint(
    model: nn.Module,
    model_dir,
    train_dataloader,
    loss_fn,
    loss_param,
    dataset_param,
    rep_param,
    summary_fn,
    plotting_function,
    optLatentOnly = False,
    fromCheckPt = None
   ):

    train_losses = []

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
    rendering_dir = '/'.join(model_dir.split('/')[:-3]).replace('Experiments','Renders') + '/' + '_'.join(model_dir.split('/')[-3].split('_')[0:2]) + '/' + '_'.join(model_dir.split('/')[-3].split('_')[2:4])

    # Define optimizer
    if(not optLatentOnly):
        optimizer = torch.optim.Adam(lr=loss_param['lr_sch_initial'], params=model.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_param['lr_sch_initial']*np.power(loss_param['lr_sch_factor'],(fromCheckPt/loss_param['lr_sch_interval']), params=model.parameters()))
    else:
        for name, param in model.named_parameters():
            if param.requires_grad and ('latent' not in name):
                param.requires_grad = False
        optimizer = torch.optim.Adam(lr=loss_param['lr_sch_initial'], params=model.latents.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_param['lr_sch_initial']*np.power(loss_param['lr_sch_factor'],(fromCheckPt/loss_param['lr_sch_interval']), params=model.latents.parameters()))
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
    loss = 0

    if(fromCheckPt is not None):
        #print(np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt))).shape)
        model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % fromCheckPt)))
        model.train()
        train_check = np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt)),dtype=object).flatten()

    # Get XYZ for reconstruction
    XYZ_coord = get3DGrid(rep_param['mc_resolution'])
    
    # Get bounding box
    buffered_bbox = np.array(rep_param['buffered_bbox'])
    #print(buffered_bbox)
    #print(buffered_bbox[:,1])
    #print(buffered_bbox[:,0])

    num_steps_per_epoch = len(train_dataloader)*dataset_param['sets_perbatch']
    print('The number of steps per epoch is ' + str(num_steps_per_epoch))
    total_steps = 0

    starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1

    for epoch in range(rep_param['total_epochs']):

        if(epoch >= starting_epoch):

            start_time_epoch = time.time()
                
            for i in range(num_steps_per_epoch):
                start_time = time.time()

                # Get the next batch of data and move it to the GPU
                model_input, ground_truth_sdf, ground_truth_perf = next(iter(train_dataloader))
                model_input, ground_truth_sdf, ground_truth_perf = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_perf)

                # Compute the MLP output for the given input data and compute the loss

                model_output = model(model_input)
                #print('model_output', model_output[0].shape)
                #print('model_output', model_output)
                
                # Implement a loss function between the ground truth and model output 
                loss, sdf_loss_comb, perf_loss, sdf_loss, latent_loss_rg, param_loss_rg, lip_loss_rg = loss_fn(model_output, ground_truth_sdf, ground_truth_perf, model, loss_param) 
                print(f"Step {i}: sdf_loss = {float(sdf_loss):.5f}, sdf_loss_comb = {float(sdf_loss_comb):.5f}, perf_loss = {float(perf_loss/10):.5f}, total_loss = {float(loss):.5f}")

                #print('sdf_loss', sdf_loss)
                #print('sdf_loss_comb', sdf_loss_comb)
                #print('perf_loss', perf_loss)
                #print('latent_loss_rg', latent_loss_rg)
                #print('param_loss_rg', param_loss_rg)
                #print('lip_loss_rg', lip_loss_rg)
                if((fromCheckPt is not None) and (epoch==starting_epoch)):
                      train_losses.append(np.concatenate([train_check,loss.detach().cpu().numpy()])) 
                train_losses.append(loss.detach().cpu().numpy())
                
                # One step of optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_steps += 1
                scheduler.step()

            print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))

            if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):

                # Save the model and losses at checkpoints
                torch.save(model.state_dict(),os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),np.array(train_losses))

                # Plot summary stats and visual reconstructions
                model.eval()
                with torch.no_grad():
                    rand_index = np.random.randint(0,len(dataset_param['bld_dataset_ind'])/2)
                     #print(len(dataset_param['bld_dataset_ind']))
                     #print(rand_index)
                     #print(dataset_param['bld_dataset_ind'])
                     #rand_index = 0
                    act_index = dataset_param['bld_dataset_ind'][rand_index]
                    typeLoss = 'test ' if optLatentOnly else 'train '
                    summary_fn(total_steps, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model, rendering_dir, resolution=rep_param['mc_resolution'], XYZ_coord=XYZ_coord,  index=rand_index, act_index=act_index, viz_dir=training_viz_dir, plotting_function=plotting_function, buffered_bbox=buffered_bbox, typeLoss='') 
                    model = model.to(device)
                model.train()

    torch.save(model.state_dict(),os.path.join(model_dir, 'model_final.pth'))
    np.savetxt(os.path.join(model_dir, 'train_losses_final.txt'),np.array(train_losses))
    
    

# Fit ML function
def fit_parallel(
    model: nn.Module,
    model_dir,
    train_dataloader,
    loss_fn,
    loss_param,
    dataset_param,
    rep_param,
    summary_fn,
    plotting_function,
    optLatentOnly = False,
    fromCheckPt = None
   ):

    train_losses = []

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
    rendering_dir = '/'.join(model_dir.split('/')[:-3]).replace('Experiments','Renders') + '/' + '_'.join(model_dir.split('/')[-3].split('_')[0:2]) + '/' + '_'.join(model_dir.split('/')[-3].split('_')[2:4])

    # Define optimizer
    if(not optLatentOnly):
        optimizer_sdf = torch.optim.Adam(lr=loss_param['lr_sch_initial_sdf'], params=model.mlp_sdf.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_param['lr_sch_initial_sdf']*np.power(loss_param['lr_sch_factor'],(fromCheckPt/loss_param['lr_sch_interval'])), params=model.mlp_sdf.parameters())
        optimizer_perf = torch.optim.Adam(lr=loss_param['lr_sch_initial_perf'], params=model.mlp_perf.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_param['lr_sch_initial_perf']*np.power(loss_param['lr_sch_factor'],(fromCheckPt/loss_param['lr_sch_interval'])), params=model.mlp_perf.parameters())
        optimizer_latent = torch.optim.Adam(lr=loss_param['lr_sch_initial_latent'], params=model.latents.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_param['lr_sch_initial_latent']*np.power(loss_param['lr_sch_factor'],(fromCheckPt/loss_param['lr_sch_interval'])), params=model.latents.parameters())
        scheduler_sdf = torch.optim.lr_scheduler.StepLR(optimizer_sdf, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
        scheduler_perf = torch.optim.lr_scheduler.StepLR(optimizer_perf, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
        scheduler_latent = torch.optim.lr_scheduler.StepLR(optimizer_latent, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
    else:
        for name, param in model.named_parameters():
            if param.requires_grad and ('latent' not in name):
                param.requires_grad = False
        optimizer = torch.optim.Adam(lr=loss_param['lr_sch_initial_latent'], params=model.latents.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_param['lr_sch_initial_latent']*np.power(loss_param['lr_sch_factor'],(fromCheckPt/loss_param['lr_sch_interval'])), params=model.latents.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
    
    loss = 0

    if(fromCheckPt is not None):
        #print(np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt))).shape)
        model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % fromCheckPt)))
        model.train()
        train_check = np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt)),dtype=object).flatten()

    # Get XYZ for reconstruction
    XYZ_coord = get3DGrid(rep_param['mc_resolution'])
    
    # Get bounding box
    buffered_bbox = np.array(rep_param['buffered_bbox'])
    #print(buffered_bbox)
    #print(buffered_bbox[:,1])
    #print(buffered_bbox[:,0])

    num_steps_per_epoch = len(train_dataloader)*dataset_param['sets_perbatch']
    print('The number of steps per epoch is ' + str(num_steps_per_epoch))
    total_steps = 0

    starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1

    for epoch in range(rep_param['total_epochs']):

        if(epoch >= starting_epoch):

            start_time_epoch = time.time()
                
            for i in range(num_steps_per_epoch):
                start_time = time.time()

                # Get the next batch of data and move it to the GPU
                model_input, ground_truth_sdf, ground_truth_perf = next(iter(train_dataloader))
                model_input, ground_truth_sdf, ground_truth_perf = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_perf)

                # Compute the MLP output for the given input data and compute the loss

                model_output = model(model_input)
                #print('model_output', model_output[0].shape)
                #print('model_output', model_output)
                
                # Implement a loss function between the ground truth and model output 
                loss, sdf_loss_comb, perf_loss, sdf_loss, latent_loss_rg, param_loss_rg, lip_loss_rg = loss_fn(model_output, ground_truth_sdf, ground_truth_perf, model, loss_param) 
                if(int(i%10)==0):
                    print(f"Step {i}: sdf_loss = {float(sdf_loss):.5f}, sdf_loss_comb = {float(sdf_loss_comb):.5f}, perf_loss = {float(perf_loss/10):.5f}, total_loss = {float(loss):.5f}")

                #print('sdf_loss', sdf_loss)
                #print('sdf_loss_comb', sdf_loss_comb)
                #print('perf_loss', perf_loss)
                #print('latent_loss_rg', latent_loss_rg)
                #print('param_loss_rg', param_loss_rg)
                #print('lip_loss_rg', lip_loss_rg)
                if((fromCheckPt is not None) and (epoch==starting_epoch)):
                      train_losses.append(np.concatenate([train_check,loss.detach().cpu().numpy()])) 
                train_losses.append(loss.detach().cpu().numpy())
                
                # One step of optimization
                if(not optLatentOnly):
                    optimizer_sdf.zero_grad()
                    optimizer_perf.zero_grad()
                    optimizer_latent.zero_grad()
                    
                    sdf_loss_comb.backward(retain_graph=True)
                    perf_loss.backward()
                    
                    optimizer_sdf.step()
                    optimizer_perf.step()
                    optimizer_latent.step()
                    
                    total_steps += 1
                    scheduler_sdf.step()
                    scheduler_perf.step()
                    scheduler_latent.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_steps += 1
                    scheduler.step()

            if(not optLatentOnly):
                print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, scheduler_latent.get_last_lr()[0]))
            else:
                print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))

            if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):

                # Save the model and losses at checkpoints
                torch.save(model.state_dict(),os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),np.array(train_losses))

                # Plot summary stats and visual reconstructions
                model.eval()
                with torch.no_grad():
                    rand_index = np.random.randint(0,len(dataset_param['bld_dataset_ind'])/2)
                     #print(len(dataset_param['bld_dataset_ind']))
                     #print(rand_index)
                     #print(dataset_param['bld_dataset_ind'])
                     #rand_index = 0
                    act_index = dataset_param['bld_dataset_ind'][rand_index]
                    typeLoss = 'test ' if optLatentOnly else 'train '
                    if(not optLatentOnly):
                        summary_fn(total_steps, np.mean(train_losses), scheduler_latent.get_last_lr()[0], model_output, model, rendering_dir, resolution=rep_param['mc_resolution'], XYZ_coord=XYZ_coord,  index=rand_index, act_index=act_index, viz_dir=training_viz_dir, plotting_function=plotting_function, buffered_bbox=buffered_bbox, typeLoss='') 
                    else:
                        summary_fn(total_steps, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model, rendering_dir, resolution=rep_param['mc_resolution'], XYZ_coord=XYZ_coord,  index=rand_index, act_index=act_index, viz_dir=training_viz_dir, plotting_function=plotting_function, buffered_bbox=buffered_bbox, typeLoss='') 
                    model = model.to(device)
                model.train()

    torch.save(model.state_dict(),os.path.join(model_dir, 'model_final.pth'))
    np.savetxt(os.path.join(model_dir, 'train_losses_final.txt'),np.array(train_losses))
    

    
    
# def fit_hybrid(
#     model_urban_voxel: nn.Module,
#     model_bldg_voxel: nn.Module,
#     model_sdf_field: nn.Module,
#     model_dir,
#     train_dataloader_urb,
#     train_dataloader_bldg,
#     loss_fn,
#     loss_param,
#     dataset_param,
#     rep_param,
#     summary_fn,
#     plotting_function,
#     optLatentOnly = False,
#     fromCheckPt = None,
#     train_urban = False,
#     train_building = False,
#    ):

#     # Create model directory
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     # Create checkpoints directory
#     checkpoints_dir = model_dir + '/' + 'checkpoints'
#     if not os.path.exists(checkpoints_dir):
#         os.makedirs(checkpoints_dir)

#     # Create training progress visualization directory
#     training_viz_dir = model_dir + '/' + 'training_viz'
#     if not os.path.exists(training_viz_dir):
#         os.makedirs(training_viz_dir)
    
#     # Get rendering directory
#     rendering_dir = '/'.join(model_dir.split('/')[:-3]).replace('Experiments','Renders') + '/' + '_'.join(model_dir.split('/')[-3].split('_')[0:2]) + '/' + '_'.join(model_dir.split('/')[-3].split('_')[2:4])

#     # Define optimizer
#     if(optLatentOnly):    
#         for name, param in model_sdf_field.named_parameters():
#             if param.requires_grad and ('latent' not in name):
#                 param.requires_grad = False
#         if(train_urban):
#             for name, param in model_urban_voxel.named_parameters():
#                 if param.requires_grad and ('latent' not in name):
#                     param.requires_grad = False
#             optimizer_urb = torch.optim.Adam([{'params':model_urban_voxel.latents.parameters()}],lr=loss_param['lr_sch_initial']*10) if (fromCheckPt is None) else torch.optim.Adam([{'params':model_urban_voxel.latents.parameters()}],lr=loss_param['lr_sch_initial']*10*loss_param['lr_sch_factor']*(fromCheckPt/loss_param['lr_sch_interval']))
#             scheduler_urban = torch.optim.lr_scheduler.StepLR(optimizer_urb, gamma=lr_sch['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
#             num_steps_per_epoch_urb = len(train_dataloader_urb)
#         if(train_building):
#             for name, param in model_bldg_voxel.named_parameters():
#                 if param.requires_grad and ('latent' not in name):
#                     param.requires_grad = False   
#             optimizer_bldg = torch.optim.Adam([{'params':model_bldg_voxel.latents.parameters()}],lr=loss_param['lr_sch_initial']) if (fromCheckPt is None) else torch.optim.Adam([{'params':model_bldg_voxel.latents.parameters()}],lr=loss_param['lr_sch_initial']*loss_param['lr_sch_factor']*(fromCheckPt/loss_param['lr_sch_interval']))
#             scheduler_bldg = torch.optim.lr_scheduler.StepLR(optimizer_bldg, gamma=loss_param['lr_sch_factor'], step_size=lr_sch['lr_sch_interval'])
#             num_steps_per_epoch_bldg = len(train_dataloader_bldg)         
#     else:
#         if(train_urban):
#             optimizer_urb = torch.optim.Adam([{'params':model_urban_voxel.parameters()},{'params':model_sdf_field.parameters()}],lr=loss_param['lr_sch_initial']*10) if (fromCheckPt is None) else torch.optim.Adam([{'params':model_urban_voxel.parameters()},{'params':model_sdf_field.parameters()}],lr=loss_param['lr_sch_initial']*10*loss_param['lr_sch_factor']*(fromCheckPt/loss_param['lr_sch_interval']))
#             scheduler_urb = torch.optim.lr_scheduler.StepLR(optimizer_urb, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
#             num_steps_per_epoch_urb = len(train_dataloader_urb)
#         if(train_building):
#             optimizer_bldg = torch.optim.Adam([{'params':model_bldg_voxel.parameters()},{'params':model_sdf_field.parameters()}],lr=loss_param['lr_sch_initial']) if (fromCheckPt is None) else torch.optim.Adam([{'params':model_bldg_voxel.parameters()},{'params':model_sdf_field.parameters()}],lr=loss_param['lr_sch_initial']*10*loss_param['lr_sch_factor']*(fromCheckPt/loss_param['lr_sch_interval']))
#             scheduler_bldg = torch.optim.lr_scheduler.StepLR(optimizer_bldg, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
#             num_steps_per_epoch_bldg = len(train_dataloader_bldg)

#     # Start from checkpoint if True
#     if(fromCheckPt is not None):
#         #print(np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt))).shape)
#         model_sdf_field.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_sdf_field_epoch_%04d.pth' % fromCheckPt)))
#         model_sdf_field.train()
#         if(train_urban):
#             model_urban_voxel.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_urban_voxel_epoch_%04d.pth' % fromCheckPt)))
#             model_urban_voxel.train()
#         if(train_building):
#             model_bldg_voxel.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_bldg_voxel_epoch_%04d.pth' % fromCheckPt)))
#             model_bldg_voxel.train()
#         train_check = np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt)),dtype=object).flatten()

#     # Get XYZ for reconstruction
#     XYZ_coord = get3DGrid(rep_param['mc_resolution'])
    
#     # Get bounding box
#     buffered_bbox = np.array(rep_param['buffered_bbox'])
#     #print(buffered_bbox)
#     #print(buffered_bbox[:,1])
#     #print(buffered_bbox[:,0])

#     if(train_urban):
#         num_steps_per_epoch_urb = len(train_dataloader_urb)*dataset_param['sets_perbatch_urb']
#         print('The number of steps per epoch for urban blocks is ' + str(num_steps_per_epoch_urb))
#     if(train_building):
#         num_steps_per_epoch_bldg = len(train_dataloader_bldg)*dataset_param['sets_perbatch_bldg']
#         print('The number of steps per epoch for buildings is ' + str(num_steps_per_epoch_bldg))
    
     
#     loss_urban = 0
#     loss_bldg = 0

#     total_steps = 0
    
#     train_losses = []


#     starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1

#     for epoch in range(rep_param['total_epochs']):

#         if(epoch >= starting_epoch):

#             start_time_epoch = time.time()
            
#             if(train_building):
#             #Go over all building models
                
#                 for i in range(num_steps_per_epoch_bldg):
#                     start_time = time.time()

#                     # Get the next batch of data and move it to the GPU
#                     model_input, ground_truth = next(iter(train_dataloader_bldg))
#                     model_input, ground_truth = to_gpu(model_input), to_gpu(ground_truth)

#                     # Compute the MLP output for the given input data and compute the loss
#                     grid_output, param_latent = model_bldg_voxel(model_input)
#                     model_output = model_sdf_field(model_input, grid_output)
                    
#                     # Implement a loss function between the ground truth and model output 
#                     loss_bldg, sdf_loss, latent_loss_rg, param_loss_rg, lip_loss_rg = loss_fn(model_output, ground_truth, param_latent,[model_bldg_voxel, model_sdf_field], loss_param) 
#         #           print('sdf_loss', sdf_loss)
#         #           print('latent_loss_rg', latent_loss_rg)
#         #           print('param_loss_rg', param_loss_rg)
#         #           print('lip_loss_rg', lip_loss_rg)
#                     if((fromCheckPt is not None) and (epoch==starting_epoch)):
#                           train_losses.append(np.concatenate([train_check,loss_bldg.detach().cpu().numpy()])) 
#                     train_losses.append(loss_bldg.detach().cpu().numpy())

#                     if not total_steps % rep_param['steps_til_summary']:
#                         torch.save(model_sdf_field.state_dict(),os.path.join(model_dir, 'model_sdf_field_current.pth'))
#                         torch.save(model_bldg_voxel.state_dict(),os.path.join(model_dir, 'model_bldg_voxel_current.pth'))
#                         np.savetxt(os.path.join(model_dir, 'epoch_current.txt'),np.array([epoch]))
#                         np.savetxt(os.path.join(model_dir, 'train_losses_current.txt'),np.array(train_losses))
#                         summary_fn(total_steps, loss_bldg, scheduler.get_last_lr()[0], model_output, model_bldg_voxel, rendering_dir) 


#                     # One step of optimization
#                     optimizer_bldg.zero_grad()
#                     loss_bldg.backward()
#                     optimizer_bldg.step()

#                     total_steps += 1
#                     scheduler_bldg.step()

#             if(train_urban):
#             #Go over all urban models
                
#                 for i in range(num_steps_per_epoch_urb):
#                     start_time = time.time()

#                     # Get the next batch of data and move it to the GPU
#                     model_input, ground_truth = next(iter(train_dataloader_urb))
#                     model_input, ground_truth = to_gpu(model_input), to_gpu(ground_truth)

#                     # Compute the MLP output for the given input data and compute the loss
#                     grid_output, param_latent = model_urb_voxel(model_input)
#                     model_output = model_sdf_field(model_input, grid_output)

#                     # Implement a loss function between the ground truth and model output 
#                     loss_urb, sdf_loss, latent_loss_rg, param_loss_rg, lip_loss_rg = loss_fn(model_output, ground_truth, param_latent,[model_urb_voxel, model_sdf_field], loss_param) 
#         #           print('sdf_loss', sdf_loss)
#         #           print('latent_loss_rg', latent_loss_rg)
#         #           print('param_loss_rg', param_loss_rg)
#         #           print('lip_loss_rg', lip_loss_rg)
#                     if((fromCheckPt is not None) and (epoch==starting_epoch)):
#                           train_losses.append(np.concatenate([train_check,loss_urb.detach().cpu().numpy()])) 
#                     train_losses.append(loss_urb.detach().cpu().numpy())

#                     if not total_steps % rep_param['steps_til_summary']:
#                         torch.save(model_sdf_field.state_dict(),os.path.join(model_dir, 'model_sdf_field_current.pth'))
#                         torch.save(model_urb_voxel.state_dict(),os.path.join(model_dir, 'model_urb_voxel_current.pth'))
#                         np.savetxt(os.path.join(model_dir, 'epoch_current.txt'),np.array([epoch]))
#                         np.savetxt(os.path.join(model_dir, 'train_losses_current.txt'),np.array(train_losses))
#                         summary_fn(total_steps, loss_urb, scheduler.get_last_lr()[0], model_output, model_urb_voxel, rendering_dir) 


#                     # One step of optimization
#                     optimizer_urb.zero_grad()
#                     loss_urb.backward()
#                     optimizer_urb.step()

#                     total_steps += 1
#                     scheduler_urb.step()


#             if(train_urban and train_building):
#                 cur_lr = (optimizer_bldg.param_groups[0]['lr'] + optimizer_urb.param_groups[0]['lr'])/2
#             elif(train_urban and not train_building):
#                 cur_lr = optimizer_urb.param_groups[0]['lr']
#             else:
#                 cur_lr = optimizer_bldg.param_groups[0]['lr']

#             print("-------------------------- Epoch %d -------------------------- Total loss %0.6f, lr %0.6f, iteration time %0.6f" % (epoch, loss_urban+loss_bldg, cur_lr, time.time() - start_time))




#             if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):

#                 # Save the model and losses at checkpoints
#                 torch.save(model_sdf_field.state_dict(),os.path.join(model_dir, 'model_sdf_field_epoch_%04d.pth'))
#                 if(train_urban):
#                     torch.save(model_urb_voxel.state_dict(),os.path.join(model_dir, 'model_urb_voxel_epoch_%04d.pth'))
#                 if(train_building):
#                     torch.save(model_bldg_voxel.state_dict(),os.path.join(model_dir, 'model_bldg_voxel_epoch_%04d.pth'))
#                 np.savetxt(os.path.join(model_dir, 'train_losses_epoch_%04d.txt'),np.array(train_losses))
                
#     torch.save(model_sdf_field.state_dict(),os.path.join(model_dir, 'model_sdf_field_final.pth'))
#     if(train_urban):
#         torch.save(model_urb_voxel.state_dict(),os.path.join(model_dir, 'model_urb_voxel_final.pth'))
#     if(train_building):
#         torch.save(model_bldg_voxel.state_dict(),os.path.join(model_dir, 'model_bldg_voxel_final.pth'))
#     np.savetxt(os.path.join(model_dir, 'train_losses_final.txt'),np.array(train_losses))



# # Fit ML function
# def fit(
#     model: nn.Module,
#     model_dir,
#     train_dataloader,
#     loss_fn,
#     summary_fn,
#     plotting_function,
#     optLatentOnly = False,
#     fromCheckPt = None
#    ):

#     # Define data arrays
#     train_losses = []   
#     idx_ar, epoch_ar, step_ar, loss_ar, sdf_loss_ar, sdf_clamped_loss_ar, latent_loss_ar, param_loss_ar, lip_loss_ar, gradient_loss_ar, normal_loss_ar = [], [], [], [], [], [], [], [], [], [], []
#     d_arr = [idx_ar, epoch_ar, step_ar, loss_ar, sdf_loss_ar, sdf_clamped_loss_ar, latent_loss_ar, param_loss_ar, lip_loss_ar, gradient_loss_ar, normal_loss_ar]
#     d_arr_names = ['idx', 'epoch', 'step', 'loss', 'sdf_loss', 'sdf_clamped_loss', 'latent_loss', 'param_loss', 'lip_loss', 'gradient_loss', 'normal_loss']
    
#     # Setup parallel model if GPUs>1
#     if (torch.cuda.device_count() > 1):
#         model = torch.nn.DataParallel(model)
#         print("This training will use", torch.cuda.device_count(), "GPUs.")
    
#     # Create model sub-directories
#     model_dir, checkpoints_dir, training_viz_dir, rendering_dir = createModelDirs(model_dir)
#     rendering_dir = '/'.join(model_dir.split('/')[:-3]).replace('Experiments','Renders') + '/' + '_'.join(model_dir.split('/')[-3].split('_')[0:2]) + '/' + '_'.join(model_dir.split('/')[-3].split('_')[2:4])
#     #print('rendering_dir',rendering_dir)
    
#     # Create model sub-directories
#     mlp_param, loss_param, rep_param, dataset_param, dataset_test_param = loadModelParam(model_dir, param_ar = ['mlp_param', 'loss_param', 'rep_param', 'dataset_param', 'dataset_test_param'])
#     if(model_dir.split('/')[-1] == 'test'): # if model is test, do not use the latent and model regularizations
#         loss_param['weight_reg_lambda'] = 0
#         loss_param['lipschitz_reg_lambda'] = 0
#         loss_param['latent_reg_lambda'] = 0
    
#     # Disable dropout if in test phase
#     if(model_dir.split('/')[-1] == 'test'):
#         dropout_modules = [module for module in model.modules() if isinstance(module,torch.nn.Dropout)]
#         [module.eval() for module in dropout_modules] # disable dropout
    
#     # Load model parameters
#     det_losses_lambdas = [loss_param['sdf_loss_lambda']*(1-loss_param['sdf_loss_clamp_ratio']), loss_param['sdf_loss_lambda']*loss_param['sdf_loss_clamp_ratio'], loss_param['latent_reg_lambda'], loss_param['weight_reg_lambda'], loss_param['lipschitz_reg_lambda'], loss_param['gradient_loss_lambda'], loss_param['normal_loss_lambda']]


#     if (fromCheckPt is None):
#         lr_model = loss_param['lr_sch_initial_model']
#         lr_latent = loss_param['lr_sch_initial_latent']
#     else:
#         lr_model = loss_param['lr_sch_initial_model']*loss_param['lr_sch_factor']*(fromCheckPt/loss_param['lr_sch_interval'])
#         lr_latent = loss_param['lr_sch_initial_latent']*loss_param['lr_sch_factor']*(fromCheckPt/loss_param['lr_sch_interval'])
    
#     # Define optimizer
#     if(not optLatentOnly):
#         if(torch.cuda.device_count() > 1):
#             optimizer = torch.optim.Adam(
#                     [
#                         {
#                             "params": model.module.mlp.parameters(),
#                             "lr": lr_model,
#                         },
#                         {
#                             "params": model.module.latents.parameters(),
#                             "lr": lr_latent,
#                         },
#                     ]
#                 )
#         else:
#                         optimizer = torch.optim.Adam(
#                     [
#                         {
#                             "params": model.mlp.parameters(),
#                             "lr": loss_param['lr_sch_initial_model'],
#                         },
#                         {
#                             "params": model.latents.parameters(),
#                             "lr": loss_param['lr_sch_initial_latent'],
#                         },
#                     ]
#                 )
#     else:
#         for name, param in model.named_parameters():
#             if param.requires_grad and ('latent' not in name):
#                 param.requires_grad = False
#         if(torch.cuda.device_count() > 1):
#             optimizer = torch.optim.Adam(
#                     [
#                         {
#                             "params": model.module.latents.parameters(),
#                             "lr": lr_latent,
#                         },
#                     ]
#                 )
#         else:
#                         optimizer = torch.optim.Adam(
#                     [
#                         {
#                             "params": model.latents.parameters(),
#                             "lr": loss_param['lr_sch_initial_latent'],
#                         },
#                     ]
#                 )
    
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_param['lr_sch_factor'], step_size=loss_param['lr_sch_interval'])
#     loss = 0

#     # Loading from checkpoint
#     if(fromCheckPt is not None):
#         model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % fromCheckPt)))
#         model.train()
#         train_losses = list(np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt))).flatten().astype(float))
#         train_check_det = json.loads(open(checkpoints_dir + '/' + 'train_losses_detailed_epoch_%04d.json' % fromCheckPt).read())
#         for t in range(len(d_arr)):
#             d_arr[t] = train_check_det[d_arr_names[t]]
        
#     # Get XYZ for reconstruction
#     XYZ_coord = get3DGrid(rep_param['mc_resolution']) if(plotting_function is not None) else None
    
#     # Get bounding box
#     buffered_bbox = np.array(rep_param['buffered_bbox'])
    
#     # Define the number of batch steps needed to complete one epoch
#     num_steps_per_epoch = len(train_dataloader)*dataset_param['sets_perbatch']
#     print('The number of steps per epoch is ' + str(num_steps_per_epoch))
#     total_steps = 0

#     starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1

#     for epoch in range(rep_param['total_epochs']):

#         if(epoch >= starting_epoch):

#             start_time_epoch = time.time()
                
#             for i in range(num_steps_per_epoch):
#                 start_time = time.time()

#                 # Get the next batch of data and move it to the GPU
#                 model_input, ground_truth_sdf, ground_truth_normals = next(iter(train_dataloader))
#                 if(torch.cuda.device_count()>0):
#                     model_input, ground_truth_sdf, ground_truth_normals = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_normals)

#                 # Compute the MLP output for the given input data and compute the loss
#                 model_input['xyz'].requires_grad = True  # to calculate normal gradients
#                 model_output = model(model_input)

#                 # Implement a loss function between the ground truth and model output 
#                 loss, sdf_loss, sdf_loss_cl, latent_loss, param_loss, lip_loss, gradient_loss, normal_loss = loss_fn(model_output, ground_truth_sdf, [model], loss_param, model_input['xyz'], ground_truth_normals) 
#                 detailed_losses = [sdf_loss, sdf_loss_cl, latent_loss, param_loss, lip_loss, gradient_loss, normal_loss]
                
#                 # Append loss and detailed loss information per geometry
#                 train_losses.append(loss.detach().cpu().numpy())
#                 d_arr[0].append(model_input['idx'].detach().cpu().numpy())
#                 d_arr[1].append(epoch)
#                 d_arr[2].append(i)
#                 d_arr[3].append(loss.detach().cpu().numpy())
#                 for d in range(len(detailed_losses)):
#                     d_arr[d+4].append(detailed_losses[d].detach().cpu().numpy())
                
#                 # Write summary
#                 if not total_steps % rep_param['steps_til_summary']:
#                     torch.save(model.state_dict(),os.path.join(model_dir, 'model_current.pth'))
#                     np.savetxt(os.path.join(model_dir, 'epoch_current.txt'),np.array([epoch]))
#                     np.savetxt(os.path.join(model_dir, 'train_losses_current.txt'),np.array(train_losses).flatten())
#                     summary_fn(total_steps, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model, rendering_dir, det_losses = detailed_losses, det_losses_types = d_arr_names[4:], det_losses_lambdas = det_losses_lambdas) 
#                     train_losses_detailed = {}
#                     for r in range(len(d_arr)):
#                         train_losses_detailed[d_arr_names[r]] = d_arr[r]
#                     with open(model_dir + '/' + 'train_losses_detailed_current.json', 'w') as f:
#                         json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)

#                 # One step of optimization
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 total_steps += 1
#                 scheduler.step()

#             print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))

#             if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):

#                 # Save the model and losses at checkpoints
#                 torch.save(model.state_dict(),os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
#                 np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),np.array(train_losses))
#                 train_losses_detailed = {}
#                 for r in range(len(d_arr)):
#                     train_losses_detailed[d_arr_names[r]] = d_arr[r]
#                 with open(checkpoints_dir + '/' + 'train_losses_detailed_epoch_%04d.json' % epoch, 'w') as f:
#                     json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)

#                 # Plot summary stats and visual reconstructions
#                 # model.eval()
#                 # with torch.no_grad():
#                 #     rand_index = np.random.randint(0,len(dataset_param['bld_dataset_ind'])/2)
#                 #     act_index = dataset_param['bld_dataset_ind'][rand_index]
#                 #     typeLoss = 'test ' if optLatentOnly else 'train '
#                 #     summary_fn(total_steps, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model, rendering_dir, resolution=rep_param['mc_resolution'], XYZ_coord=XYZ_coord,  index=rand_index, act_index=act_index, viz_dir=training_viz_dir, plotting_function=plotting_function, buffered_bbox=buffered_bbox, typeLoss='') 
#                 #     model = model.to(device)
#                 # model.train()

#     # Save the final model and detailed stats
#     torch.save(model.state_dict(),os.path.join(model_dir, 'model_final.pth'))
#     np.savetxt(os.path.join(model_dir, 'train_losses_final.txt'),np.array(train_losses).flatten())
#     train_losses_detailed = {}
#     for r in range(len(d_arr)):
#         train_losses_detailed[d_arr_names[r]] = d_arr[r]
#     with open(model_dir + '/' + 'train_losses_detailed_final.json', 'w') as f:
#         json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)

# # Fit ML function
# def fit(
#     model: nn.Module,
#     model_dir,
#     train_dataloader,
#     loss_fn,
#     loss_fn_param,
#     dataset_param,
#     rep_param,
#     summary_fn,
#     plotting_function,
#     optLatentOnly = False,
#     fromCheckPt = None
#    ):

#     train_losses = []   
#     idx_ar, epoch_ar, step_ar, loss_ar, sdf_loss_ar, sdf_clamped_loss_ar, latent_loss_rg_ar, param_loss_rg_ar, lip_loss_rg_ar, gradient_loss_ar, normal_loss_ar = [], [], [], [], [], [], [], [], [], [], []
    
    
#     # Setup parallel model if GPUs>1
#     if (torch.cuda.device_count() > 1) and (model_dir.split('/')[-1] != 'test'):
#         model = torch.nn.DataParallel(model)
#         print("This training will use", torch.cuda.device_count(), "GPUs.")

#     # Create model directory
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     # Create checkpoints directory
#     checkpoints_dir = model_dir + '/' + 'checkpoints'
#     if not os.path.exists(checkpoints_dir):
#         os.makedirs(checkpoints_dir)

#     # Create training progress visualization directory
#     training_viz_dir = model_dir + '/' + 'training_viz'
#     if not os.path.exists(training_viz_dir):
#         os.makedirs(training_viz_dir)
    
#     # Get rendering directory
#     rendering_dir = '/'.join(model_dir.split('/')[:-3]).replace('Experiments','Renders') + '/' + '_'.join(model_dir.split('/')[-3].split('_')[0:2]) + '/' + '_'.join(model_dir.split('/')[-3].split('_')[2:4])

#     # Define optimizer
#     if(not optLatentOnly):
#         optimizer = torch.optim.Adam(lr=loss_fn_param['lr_sch_initial'], params=model.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_fn_param['lr_sch_initial']*loss_fn_param['lr_sch_factor']*(fromCheckPt/loss_fn_param['lr_sch_interval']), params=model.parameters())
#     else:
#         for name, param in model.named_parameters():
#             if param.requires_grad and ('latent' not in name):
#                 param.requires_grad = False
#         if torch.cuda.device_count() > 1:
#             optimizer = torch.optim.Adam(lr=loss_fn_param['lr_sch_initial'], params=model.module.latents.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_fn_param['lr_sch_initial']*loss_fn_param['lr_sch_factor']*(fromCheckPt/loss_fn_param['lr_sch_interval']), params=model.module.latents.parameters())
#         else:
#             optimizer = torch.optim.Adam(lr=loss_fn_param['lr_sch_initial'], params=model.latents.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_fn_param['lr_sch_initial']*loss_fn_param['lr_sch_factor']*(fromCheckPt/loss_fn_param['lr_sch_interval']), params=model.latents.parameters())
    
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_fn_param['lr_sch_factor'], step_size=loss_fn_param['lr_sch_interval'])
#     loss = 0

#     if(fromCheckPt is not None):
#         #print(np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt))).shape)
#         model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % fromCheckPt)))
#         model.train()
#         train_check = np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt))).flatten().astype(float)
#         train_losses = list(train_check)
#         train_check_det = json.loads(open(checkpoints_dir + '/' + 'train_losses_detailed_epoch_%04d.json' % fromCheckPt).read())
#         idx_ar = train_check_det['idx']
#         epoch_ar = train_check_det['epoch']
#         step_ar = train_check_det['step']
#         loss_ar = train_check_det['loss']
#         sdf_loss_ar = train_check_det['sdf_loss']
#         sdf_clamped_loss_ar = train_check_det['sdf_clamped_loss']
#         latent_loss_rg_ar  = train_check_det['latent_loss_rg']
#         param_loss_rg_ar = train_check_det['param_loss_rg']
#         lip_loss_rg_ar = train_check_det['lip_loss_rg']
#         gradient_loss_ar = train_check_det['gradient_loss']
#         normal_loss_ar = train_check_det['normal_loss']
        

#     # Get XYZ for reconstruction
#     XYZ_coord = get3DGrid(rep_param['mc_resolution'])
    
#     # Get bounding box
#     buffered_bbox = np.array(rep_param['buffered_bbox'])
    
#     # Define the number of batch steps needed to complete one epoch
#     num_steps_per_epoch = len(train_dataloader)*dataset_param['sets_perbatch']
#     print('The number of steps per epoch is ' + str(num_steps_per_epoch))
#     total_steps = 0

#     starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1

#     for epoch in range(rep_param['total_epochs']):

#         if(epoch >= starting_epoch):

#             start_time_epoch = time.time()
                
#             for i in range(num_steps_per_epoch):
#                 start_time = time.time()

#                 # Get the next batch of data and move it to the GPU
#                 model_input, ground_truth_sdf, ground_truth_normals = next(iter(train_dataloader))
#                 if(torch.cuda.device_count()>0):
#                     model_input, ground_truth_sdf, ground_truth_normals = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_normals)

#                 # Compute the MLP output for the given input data and compute the loss
#                 model_input['xyz'].requires_grad = True
#                 model_output = model(model_input)
                
#                 #print("Outside: input size", model_input['xyz'].size(),"output_size", model_output[0].size())

#                 # Implement a loss function between the ground truth and model output 
#                 loss, sdf_loss, sdf_clamped_loss, latent_loss_rg, param_loss_rg, lip_loss_rg, gradient_loss, normal_loss = loss_fn(model_output, ground_truth_sdf, model, loss_fn_param, model_input['xyz'], ground_truth_normals) 
#                 train_losses.append(loss.detach().cpu().numpy())
#                 # Save detailed loss information per geometry
#                 idx_ar.append(model_input['idx'].detach().cpu().numpy())
#                 epoch_ar.append(epoch)
#                 step_ar.append(i)
#                 loss_ar.append(loss.detach().cpu().numpy())
#                 sdf_loss_ar.append(sdf_loss.detach().cpu().numpy())
#                 sdf_clamped_loss_ar.append(sdf_clamped_loss.detach().cpu().numpy())
#                 latent_loss_rg_ar.append(latent_loss_rg.detach().cpu().numpy())
#                 param_loss_rg_ar.append(param_loss_rg.detach().cpu().numpy())
#                 lip_loss_rg_ar.append(lip_loss_rg.detach().cpu().numpy())
#                 gradient_loss_ar.append(gradient_loss.detach().cpu().numpy())
#                 normal_loss_ar.append(normal_loss.detach().cpu().numpy())
                
#                 # print(np.mean(train_losses))
#                 # print(np.array(train_losses).shape)
#                 #print(train_losses)loss_fn_param['sdf_loss_lambda']

#                 if not total_steps % rep_param['steps_til_summary']:
#                     torch.save(model.state_dict(),os.path.join(model_dir, 'model_current.pth'))
#                     np.savetxt(os.path.join(model_dir, 'epoch_current.txt'),np.array([epoch]))
#                     np.savetxt(os.path.join(model_dir, 'train_losses_current.txt'),np.array(train_losses).flatten())
#                     summary_fn(total_steps, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model, rendering_dir, det_losses = [sdf_loss, sdf_clamped_loss, latent_loss_rg, param_loss_rg, lip_loss_rg, gradient_loss, normal_loss], det_losses_types = ['sdf_loss', 'sdf_clamped_loss', 'latent_loss', 'param_loss', 'lip_loss', 'gradient_loss', 'normal_loss'], det_losses_lambdas = [loss_fn_param['sdf_loss_lambda']*(1-loss_fn_param['sdf_loss_clamp_ratio']), loss_fn_param['sdf_loss_lambda']*loss_fn_param['sdf_loss_clamp_ratio'], loss_fn_param['latent_reg_lambda'], loss_fn_param['weight_reg_lambda'], loss_fn_param['lipschitz_reg_lambda'], loss_fn_param['gradient_loss_lambda'], loss_fn_param['normal_loss_lambda']]) 
                    
#                     train_losses_detailed =  dict({'idx': idx_ar, 'epoch': epoch_ar, 'step': step_ar, 'loss': loss_ar, 'sdf_loss': sdf_loss_ar, 'sdf_clamped_loss': sdf_clamped_loss_ar, 'latent_loss_rg': latent_loss_rg_ar, 'param_loss_rg': param_loss_rg_ar, 'lip_loss_rg': lip_loss_rg_ar, 'gradient_loss': gradient_loss_ar , 'normal_loss': normal_loss_ar})
#                     with open(model_dir + '/' + 'train_losses_detailed_current.json', 'w') as f:
#                         json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)

#                 # One step of optimization
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 total_steps += 1
#                 scheduler.step()

#             print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))

#             if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):

#                 # Save the model and losses at checkpoints
#                 torch.save(model.state_dict(),os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
#                 np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),np.array(train_losses))
                
#                 train_losses_detailed =  dict({'idx': idx_ar, 'epoch': epoch_ar, 'step': step_ar, 'loss': loss_ar, 'sdf_loss': sdf_loss_ar, 'sdf_clamped_loss': sdf_clamped_loss_ar, 'latent_loss_rg': latent_loss_rg_ar, 'param_loss_rg': param_loss_rg_ar, 'lip_loss_rg': lip_loss_rg_ar, 'gradient_loss': gradient_loss_ar , 'normal_loss': normal_loss_ar})
#                 with open(checkpoints_dir + '/' + 'train_losses_detailed_epoch_%04d.json' % epoch, 'w') as f:
#                     json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)

#                 # Plot summary stats and visual reconstructions
#                 model.eval()
#                 with torch.no_grad():
#                     rand_index = np.random.randint(0,len(dataset_param['bld_dataset_ind'])/2)
#                      #print(len(dataset_param['bld_dataset_ind']))
#                      #print(rand_index)
#                      #print(dataset_param['bld_dataset_ind'])
#                      #rand_index = 0
#                     act_index = dataset_param['bld_dataset_ind'][rand_index]
#                     typeLoss = 'test ' if optLatentOnly else 'train '
#                     summary_fn(total_steps, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model, rendering_dir, resolution=rep_param['mc_resolution'], XYZ_coord=XYZ_coord,  index=rand_index, act_index=act_index, viz_dir=training_viz_dir, plotting_function=plotting_function, buffered_bbox=buffered_bbox, typeLoss='') 
#                     model = model.to(device)
                    
#                 model.train()

#     torch.save(model.state_dict(),os.path.join(model_dir, 'model_final.pth'))
#     np.savetxt(os.path.join(model_dir, 'train_losses_final.txt'),np.array(train_losses).flatten())
    
#     train_losses_detailed =  dict({'idx': idx_ar, 'epoch': epoch_ar, 'step': step_ar, 'loss': loss_ar, 'sdf_loss': sdf_loss_ar, 'sdf_clamped_loss': sdf_clamped_loss_ar, 'latent_loss_rg': latent_loss_rg_ar, 'param_loss_rg': param_loss_rg_ar, 'lip_loss_rg': lip_loss_rg_ar, 'gradient_loss': gradient_loss_ar , 'normal_loss': normal_loss_ar})
#     with open(model_dir + '/' + 'train_losses_detailed_final.json', 'w') as f:
#         json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)
        
# # Fit ML function
# def fit_hybrid(
#     model_grid: nn.Module,
#     model_field: nn.Module,
#     model_dir,
#     train_dataloader,
#     loss_fn,
#     summary_fn,
#     plotting_function,
#     optLatentOnly = False,
#     fromCheckPt = None
#    ):

#     train_losses = []   
#     idx_ar, epoch_ar, step_ar, loss_ar, sdf_loss_ar, sdf_clamped_loss_ar, latent_loss_rg_ar, param_loss_rg_ar, lip_loss_rg_ar, gradient_loss_ar, normal_loss_ar = [], [], [], [], [], [], [], [], [], [], []
    
#     # Setup parallel model if GPUs>1
#     if (torch.cuda.device_count() > 1) and (model_dir.split('/')[-1] != 'test'):
#         model = torch.nn.DataParallel(model)
#         print("This training will use", torch.cuda.device_count(), "GPUs.")

#     # Create model directory
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     # Create checkpoints directory
#     checkpoints_dir = model_dir + '/' + 'checkpoints'
#     if not os.path.exists(checkpoints_dir):
#         os.makedirs(checkpoints_dir)

#     # Create training progress visualization directory
#     training_viz_dir = model_dir + '/' + 'training_viz'
#     if not os.path.exists(training_viz_dir):
#         os.makedirs(training_viz_dir)
    
#     # Get rendering directory
#     rendering_dir = '/'.join(model_dir.split('/')[:-3]).replace('Experiments','Renders') + '/' + '_'.join(model_dir.split('/')[-3].split('_')[0:2]) + '/' + '_'.join(model_dir.split('/')[-3].split('_')[2:4])
    
#     # Get model parameters
#     grid_param, field_param, loss_fn_param, rep_param, dataset_param, datasetTest_param = loadHybridModelParam(model_dir)

#     # Define optimizer
#     if(not optLatentOnly):
#         optimizer = torch.optim.Adam(lr=loss_fn_param['lr_sch_initial'], params=[{'params':model_grid.parameters()},{'params':model_field.parameters()}]) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_fn_param['lr_sch_initial']*loss_fn_param['lr_sch_factor']*(fromCheckPt/loss_fn_param['lr_sch_interval']), params=[{'params':model_grid.parameters()},{'params':model_field.parameters()}])
#     else:
#         for name, param in model_grid.named_parameters():
#             if param.requires_grad and ('latent' not in name):
#                 param.requires_grad = False
#         if torch.cuda.device_count() > 1:
#             optimizer = torch.optim.Adam(lr=loss_fn_param['lr_sch_initial'], params=model_grid.module.latents.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_fn_param['lr_sch_initial']*loss_fn_param['lr_sch_factor']*(fromCheckPt/loss_fn_param['lr_sch_interval']), params=model_grid.module.latents.parameters())
#         else:
#             optimizer = torch.optim.Adam(lr=loss_fn_param['lr_sch_initial'], params=model_grid.latents.parameters()) if (fromCheckPt is None) else torch.optim.Adam(lr=loss_fn_param['lr_sch_initial']*loss_fn_param['lr_sch_factor']*(fromCheckPt/loss_fn_param['lr_sch_interval']), params=model_grid.latents.parameters())
    
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=loss_fn_param['lr_sch_factor'], step_size=loss_fn_param['lr_sch_interval'])
#     loss = 0

#     if(fromCheckPt is not None):
#         model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % fromCheckPt)))
#         model.train()
#         train_check = np.array(np.loadtxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % fromCheckPt))).flatten().astype(float)
#         train_losses = list(train_check)
#         train_check_det = json.loads(open(checkpoints_dir + '/' + 'train_losses_detailed_epoch_%04d.json' % fromCheckPt).read())
#         idx_ar = train_check_det['idx']
#         epoch_ar = train_check_det['epoch']
#         step_ar = train_check_det['step']
#         loss_ar = train_check_det['loss']
#         sdf_loss_ar = train_check_det['sdf_loss']
#         sdf_clamped_loss_ar = train_check_det['sdf_clamped_loss']
#         latent_loss_rg_ar  = train_check_det['latent_loss_rg']
#         param_loss_rg_ar = train_check_det['param_loss_rg']
#         lip_loss_rg_ar = train_check_det['lip_loss_rg']
#         gradient_loss_ar = train_check_det['gradient_loss']
#         normal_loss_ar = train_check_det['normal_loss']
        

#     # Get XYZ for reconstruction
#     XYZ_coord = get3DGrid(rep_param['mc_resolution'])
    
#     # Get bounding box
#     buffered_bbox = np.array(rep_param['buffered_bbox'])
    
#     # Define the number of batch steps needed to complete one epoch
#     num_steps_per_epoch = len(train_dataloader)*dataset_param['sets_perbatch']
#     print('The number of steps per epoch is ' + str(num_steps_per_epoch))
#     total_steps = 0

#     starting_epoch = 0 if (fromCheckPt is None) else fromCheckPt+1

#     for epoch in range(rep_param['total_epochs']):

#         if(epoch >= starting_epoch):

#             start_time_epoch = time.time()
                
#             for i in range(num_steps_per_epoch):
#                 start_time = time.time()

#                 # Get the next batch of data and move it to the GPU
#                 model_input, ground_truth_sdf, ground_truth_normals = next(iter(train_dataloader))
#                 if(torch.cuda.device_count()>0):
#                     model_input, ground_truth_sdf, ground_truth_normals = to_gpu(model_input), to_gpu(ground_truth_sdf), to_gpu(ground_truth_normals)

#                 # Compute the MLP output for the given input data and compute the loss
#                 model_input['xyz'].requires_grad = True
#                 model_output = model(model_input)
                
#                 #print("Outside: input size", model_input['xyz'].size(),"output_size", model_output[0].size())

#                 # Implement a loss function between the ground truth and model output 
#                 loss, sdf_loss, sdf_clamped_loss, latent_loss_rg, param_loss_rg, lip_loss_rg, gradient_loss, normal_loss = loss_fn(model_output, ground_truth_sdf, model, loss_fn_param, model_input['xyz'], ground_truth_normals) 
#                 train_losses.append(loss.detach().cpu().numpy())
#                 # Save detailed loss information per geometry
#                 idx_ar.append(model_input['idx'].detach().cpu().numpy())
#                 epoch_ar.append(epoch)
#                 step_ar.append(i)
#                 loss_ar.append(loss.detach().cpu().numpy())
#                 sdf_loss_ar.append(sdf_loss.detach().cpu().numpy())
#                 sdf_clamped_loss_ar.append(sdf_clamped_loss.detach().cpu().numpy())
#                 latent_loss_rg_ar.append(latent_loss_rg.detach().cpu().numpy())
#                 param_loss_rg_ar.append(param_loss_rg.detach().cpu().numpy())
#                 lip_loss_rg_ar.append(lip_loss_rg.detach().cpu().numpy())
#                 gradient_loss_ar.append(gradient_loss.detach().cpu().numpy())
#                 normal_loss_ar.append(normal_loss.detach().cpu().numpy())
                
#                 # print(np.mean(train_losses))
#                 # print(np.array(train_losses).shape)
#                 #print(train_losses)loss_fn_param['sdf_loss_lambda']

#                 if not total_steps % rep_param['steps_til_summary']:
#                     torch.save(model.state_dict(),os.path.join(model_dir, 'model_current.pth'))
#                     np.savetxt(os.path.join(model_dir, 'epoch_current.txt'),np.array([epoch]))
#                     np.savetxt(os.path.join(model_dir, 'train_losses_current.txt'),np.array(train_losses).flatten())
#                     summary_fn(total_steps, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model, rendering_dir, det_losses = [sdf_loss, sdf_clamped_loss, latent_loss_rg, param_loss_rg, lip_loss_rg, gradient_loss, normal_loss], det_losses_types = ['sdf_loss', 'sdf_clamped_loss', 'latent_loss', 'param_loss', 'lip_loss', 'gradient_loss', 'normal_loss'], det_losses_lambdas = [loss_fn_param['sdf_loss_lambda']*(1-loss_fn_param['sdf_loss_clamp_ratio']), loss_fn_param['sdf_loss_lambda']*loss_fn_param['sdf_loss_clamp_ratio'], loss_fn_param['latent_reg_lambda'], loss_fn_param['weight_reg_lambda'], loss_fn_param['lipschitz_reg_lambda'], loss_fn_param['gradient_loss_lambda'], loss_fn_param['normal_loss_lambda']]) 
                    
#                     train_losses_detailed =  dict({'idx': idx_ar, 'epoch': epoch_ar, 'step': step_ar, 'loss': loss_ar, 'sdf_loss': sdf_loss_ar, 'sdf_clamped_loss': sdf_clamped_loss_ar, 'latent_loss_rg': latent_loss_rg_ar, 'param_loss_rg': param_loss_rg_ar, 'lip_loss_rg': lip_loss_rg_ar, 'gradient_loss': gradient_loss_ar , 'normal_loss': normal_loss_ar})
#                     with open(model_dir + '/' + 'train_losses_detailed_current.json', 'w') as f:
#                         json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)

#                 # One step of optimization
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 total_steps += 1
#                 scheduler.step()

#             print("-------------------------- Epoch %d -------------------------- Total training loss %0.6f, iteration time %0.6f, learning rate %0.8f" % (epoch, np.mean(train_losses), time.time() - start_time_epoch, scheduler.get_last_lr()[0]))

#             if (not epoch % rep_param['epochs_til_checkpoint'] and epoch) or (epoch < 10):

#                 # Save the model and losses at checkpoints
#                 torch.save(model.state_dict(),os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
#                 np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),np.array(train_losses))
                
#                 train_losses_detailed =  dict({'idx': idx_ar, 'epoch': epoch_ar, 'step': step_ar, 'loss': loss_ar, 'sdf_loss': sdf_loss_ar, 'sdf_clamped_loss': sdf_clamped_loss_ar, 'latent_loss_rg': latent_loss_rg_ar, 'param_loss_rg': param_loss_rg_ar, 'lip_loss_rg': lip_loss_rg_ar, 'gradient_loss': gradient_loss_ar , 'normal_loss': normal_loss_ar})
#                 with open(checkpoints_dir + '/' + 'train_losses_detailed_epoch_%04d.json' % epoch, 'w') as f:
#                     json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)

#                 # Plot summary stats and visual reconstructions
#                 model.eval()
#                 with torch.no_grad():
#                     rand_index = np.random.randint(0,len(dataset_param['bld_dataset_ind'])/2)
#                      #print(len(dataset_param['bld_dataset_ind']))
#                      #print(rand_index)
#                      #print(dataset_param['bld_dataset_ind'])
#                      #rand_index = 0
#                     act_index = dataset_param['bld_dataset_ind'][rand_index]
#                     typeLoss = 'test ' if optLatentOnly else 'train '
#                     summary_fn(total_steps, np.mean(train_losses), scheduler.get_last_lr()[0], model_output, model, rendering_dir, resolution=rep_param['mc_resolution'], XYZ_coord=XYZ_coord,  index=rand_index, act_index=act_index, viz_dir=training_viz_dir, plotting_function=plotting_function, buffered_bbox=buffered_bbox, typeLoss='') 
#                     model = model.to(device)
                    
#                 model.train()

#     torch.save(model.state_dict(),os.path.join(model_dir, 'model_final.pth'))
#     np.savetxt(os.path.join(model_dir, 'train_losses_final.txt'),np.array(train_losses).flatten())
    
#     train_losses_detailed =  dict({'idx': idx_ar, 'epoch': epoch_ar, 'step': step_ar, 'loss': loss_ar, 'sdf_loss': sdf_loss_ar, 'sdf_clamped_loss': sdf_clamped_loss_ar, 'latent_loss_rg': latent_loss_rg_ar, 'param_loss_rg': param_loss_rg_ar, 'lip_loss_rg': lip_loss_rg_ar, 'gradient_loss': gradient_loss_ar , 'normal_loss': normal_loss_ar})
#     with open(model_dir + '/' + 'train_losses_detailed_final.json', 'w') as f:
#         json.dump(train_losses_detailed, f, indent=2, cls=NumpyArrayEncoder)
        
    