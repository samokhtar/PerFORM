import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import IterableDataset
#from torch.nn.utils.stateless import functional_call
from torch.func import functional_call
from building_sdf.learning.network_modules import *


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


# class LatentNeuralField(nn.Module):
#     def __init__(self, 
#                  num_latents, 
#                  mlp_param, 
#                  in_dim = 3, 
#                  out_dim = 1, 
#                 ):
#         super().__init__()
#         #########

#         # Generate an nn.Embedding with num_latents each of dimension as latent_dim.
#         self.latents = nn.Embedding(num_embeddings = num_latents, embedding_dim = mlp_param['feature_dim'])

#         # Initialize the embedding using a normal distribution.
#         self.latents.weight.data.normal_(0, mlp_param['latent_init'])   # training very sensitive to initialization so try changing the std
#         #self.latents.weight.data = torch.absolute(self.latents.weight.data)
#         #print(self.latents.weight)
#         # Use nn.Sequential to generate an mlp with select parameters:
#         self.mlp = MLP(mlp_param=mlp_param, input_dim=in_dim, output_dim=out_dim).to(device)

#         self.latent_dim = mlp_param['feature_dim']
    
#     def forward(self, inputs):
#         #########

#         coordinate = inputs['xyz']
#         #print('coordinate',coordinate.shape)
#         idcs = inputs['idx']

#         # Using the input indices 'idcs', get the respective latents
#         latents = self.latents(idcs)
        
#         # Reshape the latents to (batch_size, latent_dim)
#         latents = torch.reshape(latents, (idcs.shape[0], self.latent_dim))
#         #print('latents',latents.shape)

#         # Tile the latents to all points
#         latents_r = torch.unsqueeze(latents, 1)
#        # print('latents',latents_r.shape)
#         latents_r = latents_r.repeat(1, coordinate.shape[1], 1)

#         # Concatenate points and latents for model prediction
#        # print('latents',latents.shape)
#        # print('coordinate',coordinate.shape)
#         con_inputs = torch.cat([latents_r, coordinate], dim=2)
#         # print('latents',latents_r.shape)
#         # print('latents',latents_r)
#         # print('coordinate',coordinate.shape)
#         # print('coordinate',coordinate)
#         # print('inputs',con_inputs.shape)
#         # print('inputs',con_inputs)
#         #print('con_inputs',con_inputs.shape)

#         # Compute the output of decoder of the latents
#         outputs = self.mlp(coordinate)
#         #print('outputs',outputs.shape)
        
#         #print("\tIn Model: input size", con_inputs.size(),"output size", outputs.size())

    

class LatentNeuralField(nn.Module):
    def __init__(self, 
                 num_latents, 
                 mlp_param, 
                 in_dim = 3, 
                 out_dim = 1, 
                ):
        super().__init__()
        #########

        # Generate an nn.Embedding with num_latents each of dimension as latent_dim.
        self.latents = nn.Embedding(num_embeddings = num_latents, embedding_dim = mlp_param['feature_dim'])

        # Initialize the embedding using a normal distribution.
        self.latents.weight.data.normal_(0, mlp_param['latent_init'])   # training very sensitive to initialization so try changing the std
        #self.latents.weight.data = torch.absolute(self.latents.weight.data)
        #print(self.latents.weight)
        
        # Use GaussFFT for input encoding - if True
        self.apply_encoding = bool(mlp_param['gauss_fft'])
        if(self.apply_encoding):
            #print('Input encoding ON')
            self.input_encoding = GaussianFourierFeatureTransform(in_dim, mapping_size=mlp_param['gauss_fft_param'][0], scale=mlp_param['gauss_fft_param'][1], seed=mlp_param['seed'])
            in_dim = mlp_param['gauss_fft_param'][0]*2
        
        # Use nn.Sequential to generate an mlp with select parameters:
        self.mlp = MLP(mlp_param=mlp_param, input_dim=in_dim+mlp_param['feature_dim'], output_dim=out_dim).to(device)

        self.latent_dim = mlp_param['feature_dim']
    
    def forward(self, inputs):
        #########

        coordinate = inputs['xyz']
        #print('coordinate',coordinate)
        
        # Apply coordinate fft input encoding
        if(self.apply_encoding):
            coordinate = self.input_encoding(coordinate)
        #print('coordinate',coordinate)
        
        #print('coordinate',coordinate.shape)
        idcs = inputs['idx']

        # Using the input indices 'idcs', get the respective latents
        latents = self.latents(idcs)
        
        # Reshape the latents to (batch_size, latent_dim)
        latents = torch.reshape(latents, (idcs.shape[0], self.latent_dim))

        # Tile the latents to all points
        latents_r = torch.unsqueeze(latents, 1)
        latents_r = latents_r.repeat(1, coordinate.shape[1], 1)

        # Concatenate points and latents for model prediction
        con_inputs = torch.cat([latents_r, coordinate], dim=-1)

        # Compute the output of decoder of the latents
        outputs = self.mlp(con_inputs)
        # print('outputs_nan',torch.sum(torch.isnan(outputs)))
        # print('inputs_nan',torch.sum(torch.isnan(con_inputs)))
        # print('outputs',outputs.shape)
        #print('outputs',outputs)
        
        #print("\tIn Model: input size", con_inputs.size(),"output size", outputs.size())

        return outputs, latents
    
    
# class LatentRotationEncoder(nn.Module):
#     def __init__(self, 
#                  mlp_param,
#                  latent_sdf_field,
#                 ):
#         super().__init__()
#         #########
        
#         # Use nn.Sequential to generate an mlp with select parameters:
#         self.mlp = MLP_Simple(mlp_param=mlp_param, input_dim=1+mlp_param['feature_dim'], output_dim=mlp_param['feature_dim']).to(device)
        
#         self.latent_dim = mlp_param['feature_dim']
#         self.latent_sdf_field = latent_sdf_field.module.to(device) if(isinstance(latent_sdf_field, nn.DataParallel)) else latent_sdf_field.to(device)
    
#     def forward(self, inputs):
#         #########

#         coordinate = inputs['xyz']
#         rotation = inputs['rot']
#         idcs = inputs['idx']
        
#         self.latent_sdf_field = self.latent_sdf_field.to(coordinate.device)

#         # Using the input indices 'idcs', get the respective latents
#         sdf_latents = self.latent_sdf_field.module.latents(idcs) if(isinstance(self.latent_sdf_field, nn.DataParallel)) else self.latent_sdf_field.latents(idcs)
#         sdf_latents = torch.reshape(sdf_latents, (idcs.shape[0], self.latent_dim))
        
#         # Concatenate input to rotation encoder
#         con_inputs_rotEnc = torch.cat([sdf_latents, rotation/360], dim=-1)

#         # Forward pass through the rotation encoder
#         rot_latents = self.mlp(con_inputs_rotEnc)

#         # Tile the latents to all points
#         rot_latents_r = torch.unsqueeze(rot_latents, 1)
#         rot_latents_r = rot_latents_r.repeat(1, coordinate.shape[1], 1)

#         # Concatenate points and latents for model prediction
#         con_inputs_sdf = torch.cat([rot_latents_r, coordinate], dim=2)

#         # Compute the output of decoder of the latents
#         outputs = self.latent_sdf_field.module.mlp(con_inputs_sdf) if(isinstance(self.latent_sdf_field, nn.DataParallel)) else self.latent_sdf_field.mlp(con_inputs_sdf)

#         return outputs, rot_latents
    
    
 
    
class LatentRotationEncoder(nn.Module):
    def __init__(self, 
                 mlp_param,
                 latent_sdf_field,
                 geo_ids,
                ):
        super().__init__()
        #########
        
        # Use nn.Sequential to generate an mlp with select parameters:
        self.mlp = MLP_Simple(mlp_param=mlp_param, input_dim=1+mlp_param['feature_dim'], output_dim=mlp_param['feature_dim']).to(device)
        
        self.latent_dim = mlp_param['feature_dim']
        self.latent_sdf_field = latent_sdf_field.module.to(device) if(isinstance(latent_sdf_field, nn.DataParallel)) else latent_sdf_field.to(device)
        
        self.geo_ids = geo_ids
        self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
        #print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)

    
    def forward(self, inputs):
        #########

        coordinate = inputs['xyz']
        rotation = inputs['rot']
        idcs = inputs['idx']
        geo_code = inputs['geo_code']
        
        # name_translations = np.array(['building','extrBuilding'])
        # g_numpy = geo_code.detach().cpu().numpy().astype(int)
        # print('Current geometries in batch: ' + str(np.array([name_translations[g[0]] + '_' + str(g[1]) for g in g_numpy])))
        
        
        
        self.geo_ids_geo_code = self.geo_ids_geo_code.to(coordinate.device)
        # print('geo_code',geo_code)
        #print('self.geo_ids_geo_code.shape',self.geo_ids_geo_code.shape)
        self.equiv_ids = torch.sum(((torch.sum((geo_code==self.geo_ids_geo_code).long(),-1)==2)*torch.arange(self.geo_ids_geo_code.shape[0]).to(coordinate.device).unsqueeze(0).T),0)
        # print('geo_ids',self.geo_ids)
        # print('equiv_ids',self.equiv_ids)
        
#         print('Equivalent geometries from the geometry set: ' + str(np.array(self.geo_ids)[self.equiv_ids.detach().cpu().numpy().astype(int)]))
#         print(" ")
        

        self.latent_sdf_field = self.latent_sdf_field.to(coordinate.device)
        # Using the input indices 'idcs', get the respective latents
        sdf_latents = self.latent_sdf_field.module.latents(self.equiv_ids) if(isinstance(self.latent_sdf_field, nn.DataParallel)) else self.latent_sdf_field.latents(self.equiv_ids)
        sdf_latents = torch.reshape(sdf_latents, (self.equiv_ids.shape[0], self.latent_dim))
        
        # Concatenate input to rotation encoder
        con_inputs_rotEnc = torch.cat([sdf_latents, rotation/360], dim=-1)

        # Forward pass through the rotation encoder
        rot_latents = self.mlp(con_inputs_rotEnc)

        # Tile the latents to all points
        rot_latents_r = torch.unsqueeze(rot_latents, 1)
        rot_latents_r = rot_latents_r.repeat(1, coordinate.shape[1], 1)

        # Concatenate points and latents for model prediction
        con_inputs_sdf = torch.cat([rot_latents_r, coordinate], dim=2)

        # Compute the output of decoder of the latents
        outputs = self.latent_sdf_field.module.mlp(con_inputs_sdf) if(isinstance(self.latent_sdf_field, nn.DataParallel)) else self.latent_sdf_field.mlp(con_inputs_sdf)

        return outputs, rot_latents
    
class LatentRotationEncoderCompMLP(nn.Module):
    def __init__(self, 
                 mlp_param,
                 latent_sdf_field,
                 geo_ids,
                ):
        super().__init__()
        #########
        
        # Use nn.Sequential to generate an mlp with select parameters:
        self.mlp = MLP(mlp_param=mlp_param, input_dim=1+mlp_param['feature_dim'], output_dim=mlp_param['feature_dim']).to(device)
        
        self.latent_dim = mlp_param['feature_dim']
        self.latent_sdf_field = latent_sdf_field.module.to(device) if(isinstance(latent_sdf_field, nn.DataParallel)) else latent_sdf_field.to(device)
        
        self.geo_ids = geo_ids
        self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
        #print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)

    
    def forward(self, inputs):
        #########

        coordinate = inputs['xyz']
        rotation = inputs['rot']
        idcs = inputs['idx']
        geo_code = inputs['geo_code']
        
        
        self.geo_ids_geo_code = self.geo_ids_geo_code.to(coordinate.device)
        # print('geo_code',geo_code)
        #print('self.geo_ids_geo_code.shape',self.geo_ids_geo_code.shape)
        self.equiv_ids = torch.sum(((torch.sum((geo_code==self.geo_ids_geo_code).long(),-1)==2)*torch.arange(self.geo_ids_geo_code.shape[0]).to(coordinate.device).unsqueeze(0).T),0)
        # print('geo_ids',self.geo_ids)
        # print('equiv_ids',self.equiv_ids)

        self.latent_sdf_field = self.latent_sdf_field.to(coordinate.device)
        # Using the input indices 'idcs', get the respective latents
        sdf_latents = self.latent_sdf_field.module.latents(self.equiv_ids) if(isinstance(self.latent_sdf_field, nn.DataParallel)) else self.latent_sdf_field.latents(self.equiv_ids)
        sdf_latents = torch.reshape(sdf_latents, (self.equiv_ids.shape[0], self.latent_dim))
        
        # Concatenate input to rotation encoder
        con_inputs_rotEnc = torch.cat([sdf_latents, rotation/360], dim=-1)

        # Forward pass through the rotation encoder
        rot_latents = self.mlp(con_inputs_rotEnc)

        # Tile the latents to all points
        rot_latents_r = torch.unsqueeze(rot_latents, 1)
        rot_latents_r = rot_latents_r.repeat(1, coordinate.shape[1], 1)

        # Concatenate points and latents for model prediction
        con_inputs_sdf = torch.cat([rot_latents_r, coordinate], dim=2)

        # Compute the output of decoder of the latents
        outputs = self.latent_sdf_field.module.mlp(con_inputs_sdf) if(isinstance(self.latent_sdf_field, nn.DataParallel)) else self.latent_sdf_field.mlp(con_inputs_sdf)

        return outputs, rot_latents
    
    
    
# class LatentNeuralPerformanceField(nn.Module):
#     def __init__(self, 
#                  ctxtmod_param, 
#                  perfdec_param, 
#                  equiv_ids,
#                  geo_latents,
#                  rotEnc_model = None,
#                  equiv_ids_val = None,
#                 ):
#         super().__init__()
#         #########

#         self.ctxtmod_autoenc = Autoencoder_MLP(autoenc_param = ctxtmod_param).to(device)
#         # print('ctxtmod_autoenc_model, num_parameters: ', getModelParametersCount(ctxtmod_autoenc_model))
#         # print(ctxtmod_autoenc_model)
#         # print_params(ctxtmod_autoenc_model)

#         self.perf_dec = MLP(mlp_param=perfdec_param, input_dim=perfdec_param['input_dim']+perfdec_param['geo_feature_dim']+perfdec_param['ctxt_feature_dim'], output_dim=perfdec_param['output_dim']).to(device)
#         # print('perf_dec_model, num_parameters: ', getModelParametersCount(perf_dec_model))
#         # print(perf_dec_model)
#         # print_params(perf_dec_model)
        
#         self.equiv_ids = equiv_ids.to(device)
#         self.equiv_ids_val = equiv_ids_val.to(device) if(equiv_ids_val is not None) else None
#         self.geo_latents = geo_latents.to(device)
#         if(rotEnc_model is not None):
#             self.apply_rot = True
#             self.rotation_enc = rotEnc_model.module.to(device) if(isinstance(rotEnc_model, nn.DataParallel)) else rotEnc_model.to(device)
#         else:
#             self.apply_rot = False
        
        
#     def forward(self, inputs):
#         #########
        
#         coordinate = inputs['xyz']
#         idx = inputs['idx'].flatten()
#         rotation = inputs['rot']
        
#         self.equiv_ids = self.equiv_ids.to(coordinate.device)
#         self.geo_latents = self.geo_latents.to(coordinate.device)
#         #self.rotEnc_mlp = self.rotEnc_mlp.to(coordinate.device)

#         # Get the subset of geometry latents
#         g_latents_sub = self.geo_latents[self.equiv_ids[idx]].to(coordinate.device)
        
#         # Forward pass through the rotation module
#         if(self.apply_rot):
#             if(rotation.float().mean()!=torch.tensor([0.]).to(coordinate.device)):
#                 con_input_rot = torch.cat([g_latents_sub, rotation/360], dim=-1)
#                 g_latents_sub = self.rotation_enc.module.mlp(con_input_rot) if(isinstance(self.rotation_enc, nn.DataParallel)) else self.rotation_enc.mlp(con_input_rot)
        
#         # Forward pass through the contextualizing module
#         ctxt_latent = self.ctxtmod_autoenc(g_latents_sub)
        
#         # Tile the latents for all points
#         g_latents_sub = torch.unsqueeze(g_latents_sub, 1)
#         g_latents_sub = torch.tile(g_latents_sub, (1, coordinate.shape[1], 1))
#         ctxt_latent_t = torch.unsqueeze(ctxt_latent, 1)
#         ctxt_latent_t = torch.tile(ctxt_latent_t, (1, coordinate.shape[1], 1))
                
#         # Forward pass through the perf decoder    
#         dec_input = torch.cat([g_latents_sub, ctxt_latent_t, coordinate], 2)
#         outputs = self.perf_dec(dec_input)

#         return outputs, ctxt_latent 
    
# class LatentNeuralPerformanceField_incRot(nn.Module):
#     def __init__(self, 
#                  ctxtmod_param, 
#                  perfdec_param, 
#                  geo_latents,
#                  geo_ids,
#                  rotEnc_model = None,
#                 ):
#         super().__init__()
#         #########

#         self.ctxtmod_autoenc = Autoencoder_MLP(autoenc_param = ctxtmod_param).to(device)
#         # print('ctxtmod_autoenc_model, num_parameters: ', getModelParametersCount(ctxtmod_autoenc_model))
#         # print(ctxtmod_autoenc_model)
#         # print_params(ctxtmod_autoenc_model)

#         self.perf_dec = MLP(mlp_param=perfdec_param, input_dim=perfdec_param['input_dim']+perfdec_param['geo_feature_dim']+perfdec_param['ctxt_feature_dim'], output_dim=perfdec_param['output_dim']).to(device)
#         # print('perf_dec_model, num_parameters: ', getModelParametersCount(perf_dec_model))
#         # print(perf_dec_model)
#         # print_params(perf_dec_model)
        
#         self.geo_ids = geo_ids
#         self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
#         print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
        
#         self.geo_latents = geo_latents.to(device)
#         if(rotEnc_model is not None):
#             self.apply_rot = True
#             self.rotation_enc = rotEnc_model.module.to(device) if(isinstance(rotEnc_model, nn.DataParallel)) else rotEnc_model.to(device)
#             self.rotation_enc.eval()
#         else:
#             self.apply_rot = False
        
        
#     def forward(self, inputs):
#         #########
        
#         geo_types = ['building','extrBuilding']
#         coordinate = inputs['xyz']
#         idx = inputs['idx'].flatten()
#         rotation = inputs['rot']
#         geo_code = inputs['geo_code']

#         # print('coordinate',coordinate.shape)
#         # print('idx',idx.shape)
#         # print('rotation',rotation.shape)
#         # print('geo_code',geo_code.shape)
        
#         self.geo_ids_geo_code = self.geo_ids_geo_code.to(coordinate.device)
#         #print('self.geo_ids_geo_code.shape',self.geo_ids_geo_code.shape)
#         self.equiv_ids = torch.sum(((torch.sum((geo_code==self.geo_ids_geo_code).long(),-1)==2)*torch.arange(self.geo_ids_geo_code.shape[0]).to(coordinate.device).unsqueeze(0).T),0)
#         #print('self.equiv_ids',self.equiv_ids.shape)
#         self.geo_latents = self.geo_latents.to(coordinate.device)
#         #print('self.geo_latents',self.geo_latents.shape)

#         # Get the subset of geometry latents
#         g_latents_sub = self.geo_latents[self.equiv_ids].to(coordinate.device)
#         #print('g_latents_sub',g_latents_sub.shape)
        
#         # Forward pass through the rotation module
#         if(self.apply_rot):
#             if(rotation.float().mean()!=torch.tensor([0.]).to(coordinate.device)):
#                 #print('g_latents_sub',g_latents_sub.shape)
#                 #print('rotation',rotation.shape)
#                 con_input_rot = torch.cat([g_latents_sub, rotation/360], dim=-1)
#                 g_latents_sub = self.rotation_enc.module.mlp(con_input_rot) if(isinstance(self.rotation_enc, nn.DataParallel)) else self.rotation_enc.mlp(con_input_rot)
        
#         # Forward pass through the contextualizing module
#         ctxt_latent = self.ctxtmod_autoenc(g_latents_sub)
#         #print('ctxt_latent',ctxt_latent.shape)
        
#         # Tile the latents for all points
#         g_latents_sub = torch.unsqueeze(g_latents_sub, 1)
#         g_latents_sub = torch.tile(g_latents_sub, (1, coordinate.shape[1], 1))
#         ctxt_latent_t = torch.unsqueeze(ctxt_latent, 1)
#         ctxt_latent_t = torch.tile(ctxt_latent_t, (1, coordinate.shape[1], 1))
#         #print('ctxt_latent_t',ctxt_latent_t.shape)
                
#         # Forward pass through the perf decoder    
#         dec_input = torch.cat([g_latents_sub, ctxt_latent_t, coordinate], 2)
#         outputs = self.perf_dec(dec_input)
#         #print('outputs',outputs.shape)

#         return outputs, ctxt_latent 
    
# class LatentNeuralPerformanceField(nn.Module):
#     def __init__(self, 
#                  ctxtmod_param, 
#                  perfdec_param, 
#                  geo_latents,
#                  geo_ids,
#                  rotEnc_model = None,
#                 ):
#         super().__init__()
#         #########

#         self.ctxtmod_autoenc = Autoencoder_MLP(autoenc_param = ctxtmod_param).to(device)
#         # print('ctxtmod_autoenc_model, num_parameters: ', getModelParametersCount(ctxtmod_autoenc_model))
#         # print(ctxtmod_autoenc_model)
#         # print_params(ctxtmod_autoenc_model)

#         self.perf_dec = MLP(mlp_param=perfdec_param, input_dim=perfdec_param['input_dim']+perfdec_param['geo_feature_dim']+perfdec_param['ctxt_feature_dim'], output_dim=perfdec_param['output_dim']).to(device)
#         # print('perf_dec_model, num_parameters: ', getModelParametersCount(perf_dec_model))
#         # print(perf_dec_model)
#         # print_params(perf_dec_model)
        
#         self.geo_ids = geo_ids
#         self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
#         print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
        
#         self.geo_latents = geo_latents.to(device)
#         if(rotEnc_model is not None):
#             self.apply_rot = True
#             self.rotation_enc = rotEnc_model.module.to(device) if(isinstance(rotEnc_model, nn.DataParallel)) else rotEnc_model.to(device)
#             self.rotation_enc.eval()
#         else:
#             self.apply_rot = False
        
        
#     def forward(self, inputs):
#         #########
        
#         geo_types = ['building','extrBuilding']
#         coordinate = inputs['xyz']
#         idx = inputs['idx'].flatten()
#         rotation = inputs['rot']
#         geo_code = inputs['geo_code']

#         # print('coordinate',coordinate.shape)
#         # print('idx',idx.shape)
#         # print('rotation',rotation.shape)
#         # print('geo_code',geo_code.shape)
        
#         self.geo_ids_geo_code = self.geo_ids_geo_code.to(coordinate.device)
#         #print('self.geo_ids_geo_code.shape',self.geo_ids_geo_code.shape)
#         self.equiv_ids = torch.sum(((torch.sum((geo_code==self.geo_ids_geo_code).long(),-1)==2)*torch.arange(self.geo_ids_geo_code.shape[0]).to(coordinate.device).unsqueeze(0).T),0)
#         #print('self.equiv_ids',self.equiv_ids.shape)
#         self.geo_latents = self.geo_latents.to(coordinate.device)
#         #print('self.geo_latents',self.geo_latents.shape)

#         # Get the subset of geometry latents
#         g_latents_sub = self.geo_latents[self.equiv_ids].to(coordinate.device)
#         #print('g_latents_sub',g_latents_sub.shape)
        
#         # Forward pass through the rotation module
#         if(self.apply_rot):
#             if(rotation.float().mean()!=torch.tensor([0.]).to(coordinate.device)):
#                 #print('g_latents_sub',g_latents_sub.shape)
#                 #print('rotation',rotation.shape)
#                 con_input_rot = torch.cat([g_latents_sub, rotation/360], dim=-1)
#                 g_latents_sub = self.rotation_enc.module.mlp(con_input_rot) if(isinstance(self.rotation_enc, nn.DataParallel)) else self.rotation_enc.mlp(con_input_rot)
        
#         # Forward pass through the contextualizing module
#         ctxt_latent = self.ctxtmod_autoenc(g_latents_sub)
#         #print('ctxt_latent',ctxt_latent.shape)
        
#         # Tile the latents for all points
#         g_latents_sub = torch.unsqueeze(g_latents_sub, 1)
#         g_latents_sub = torch.tile(g_latents_sub, (1, coordinate.shape[1], 1))
#         ctxt_latent_t = torch.unsqueeze(ctxt_latent, 1)
#         ctxt_latent_t = torch.tile(ctxt_latent_t, (1, coordinate.shape[1], 1))
#         #print('ctxt_latent_t',ctxt_latent_t.shape)
                
#         # Forward pass through the perf decoder    
#         dec_input = torch.cat([g_latents_sub, ctxt_latent_t, coordinate], 2)
#         outputs = self.perf_dec(dec_input)
#         #print('outputs',outputs.shape)

#         return outputs, ctxt_latent 
    

class LatentNeuralPerformanceField(nn.Module):
    def __init__(self, 
                 ctxtmod_param, 
                 perfdec_param, 
                 geo_latents,
                 geo_ids,
                 rotEnc_model = None,
                ):
        super().__init__()
        #########

        if(ctxtmod_param is not None):
            self.ctxtmod_autoenc = MLP(mlp_param=ctxtmod_param, input_dim=ctxtmod_param['input_dim'], output_dim=ctxtmod_param['output_dim']).to(device)  #Autoencoder_MLP(autoenc_param = ctxtmod_param).to(device)
            self.apply_ctxt = True
        else:
            self.apply_ctxt = False
            # print('ctxtmod_autoenc_model, num_parameters: ', getModelParametersCount(ctxtmod_autoenc_model))
            # print(ctxtmod_autoenc_model)
            # print_params(ctxtmod_autoenc_model)
        self.include_sdf_inpt = False if(('include_sdf_inpt' not in list(perfdec_param.keys())) or (not perfdec_param['include_sdf_inpt'])) else True
        input_dim = perfdec_param['input_dim']+1 if(self.include_sdf_inpt) else perfdec_param['input_dim']
        self.perf_dec = MLP(mlp_param=perfdec_param, input_dim=input_dim+perfdec_param['ctxt_feature_dim'], output_dim=perfdec_param['output_dim']).to(device) if(self.apply_ctxt) else MLP(mlp_param=perfdec_param, input_dim=input_dim+perfdec_param['geo_feature_dim']+perfdec_param['ctxt_feature_dim'], output_dim=perfdec_param['output_dim']).to(device)  
        #print(self.perf_dec)
        # print('perf_dec_model, num_parameters: ', getModelParametersCount(perf_dec_model))
        # print(perf_dec_model)
        # print_params(perf_dec_model)
        
        if(geo_ids is not None):
            self.geo_ids = geo_ids
            if(len(geo_ids[0].split('_'))==2):
                self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
                #print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
            else:
                self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1]), int(g_name.split('_')[2])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1]),int(g_name.split('_')[2])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
                #print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
      #  else:
           # print('geo_ids are not defined!')
        
        if(geo_latents is not None):
            self.geo_latents = geo_latents.to(device)
            #print(self.geo_latents.shape)
       # else:
           # print('geo_latents are not defined!')
        
        if(rotEnc_model is not None):
            self.apply_rot = True
            self.rotation_enc = rotEnc_model.module.to(device) if(isinstance(rotEnc_model, nn.DataParallel)) else rotEnc_model.to(device)
            self.rotation_enc.eval()
        else:
            self.apply_rot = False
        
        
    def forward(self, inputs):
        #########
        
        geo_types = ['building','extrBuilding']
        coordinate = inputs['xyz']
        idx = inputs['idx'].flatten()
        rotation = inputs['rot']
        geo_code = inputs['geo_code']
        if(self.include_sdf_inpt):
            sdf = inputs['sdf']

        # print('coordinate',coordinate.shape)
        # print('idx',idx.shape)
        # print('rotation',rotation.shape)
        # print('geo_code',geo_code.shape)
        
        self.geo_ids_geo_code = self.geo_ids_geo_code.to(coordinate.device)
        #print('self.geo_ids_geo_code.shape',self.geo_ids_geo_code.shape)
        geo_code_rot = torch.cat([geo_code,rotation],-1) if(self.geo_ids_geo_code.shape[-1]==3) else geo_code
        
        # name_translations = np.array(['building','extrBuilding'])
        # g_numpy = geo_code_rot.detach().cpu().numpy().astype(int) 
        # print('Current geometries in batch: ' + str(np.array([name_translations[g[0]] + '_' + str(g[1]) + '_' + str(g[2]) for g in g_numpy])))
        
        self.equiv_ids = torch.sum(((torch.sum((geo_code_rot==self.geo_ids_geo_code).long(),-1)==self.geo_ids_geo_code.shape[-1])*torch.arange(self.geo_ids_geo_code.shape[0]).to(coordinate.device).unsqueeze(0).T),0)
        #print('self.equiv_ids',self.equiv_ids.shape)
        self.geo_latents = self.geo_latents.to(coordinate.device)
        #print('self.geo_latents',self.geo_latents.shape)
        
        # print('Equivalent geometries from the geometry set: ' + str(np.array(self.geo_ids)[self.equiv_ids.detach().cpu().numpy().astype(int)]))
        # print(" ")

        # Get the subset of geometry latents
        g_latents_sub = self.geo_latents[self.equiv_ids].to(coordinate.device)
        #print('g_latents_sub',g_latents_sub.shape)
        
        # Forward pass through the rotation module
        if(self.apply_rot):
            if(rotation.float().mean()!=torch.tensor([0.]).to(coordinate.device)):
                #print('g_latents_sub',g_latents_sub.shape)
                #print('rotation',rotation.shape)
                con_input_rot = torch.cat([g_latents_sub, rotation/360], dim=-1)
                g_latents_sub = self.rotation_enc.module.mlp(con_input_rot) if(isinstance(self.rotation_enc, nn.DataParallel)) else self.rotation_enc.mlp(con_input_rot)
        
        # Forward pass through the contextualizing module
        ctxt_latent = None
        if(self.apply_ctxt):
            ctxt_latent = self.ctxtmod_autoenc(g_latents_sub)
            #print('ctxt_latent',ctxt_latent.shape)
        
        # Tile the latents for all points
        # g_latents_sub = torch.unsqueeze(g_latents_sub, 1)
        # g_latents_sub = torch.tile(g_latents_sub, (1, coordinate.shape[1], 1))
        if(self.apply_ctxt):
            ctxt_latent_t = torch.unsqueeze(ctxt_latent, 1)
            ctxt_latent_t = torch.tile(ctxt_latent_t, (1, coordinate.shape[1], 1))
        else:
            g_latents_sub = torch.unsqueeze(g_latents_sub, 1)
            g_latents_sub = torch.tile(g_latents_sub, (1, coordinate.shape[1], 1))
        #print('ctxt_latent_t',ctxt_latent_t.shape)
                
        # Forward pass through the perf decoder    
        if(self.include_sdf_inpt):
            dec_input = torch.cat([ctxt_latent_t, coordinate, sdf], -1) if(self.apply_ctxt) else torch.cat([g_latents_sub, coordinate, sdf], -1)
        else:
            dec_input = torch.cat([ctxt_latent_t, coordinate], -1) if(self.apply_ctxt) else torch.cat([g_latents_sub, coordinate], -1)
        outputs = self.perf_dec(dec_input)
        #print('outputs',outputs.shape)

        return outputs, ctxt_latent 
    
    

class LatentNeuralPerformanceFieldAggr_CDF(nn.Module):
    def __init__(self, 
                 ctxtmod_param, 
                 perf_com_param, 
                 perf_sep_param,
                 geo_latents,
                 geo_ids,
                 rotEnc_model = None,
                ):
        super().__init__()
        #########

        if(ctxtmod_param is not None):
            self.ctxtmod_autoenc = MLP(mlp_param=ctxtmod_param, input_dim=ctxtmod_param['input_dim'], output_dim=ctxtmod_param['output_dim']).to(device)
            self.apply_ctxt = True
        else:
            self.apply_ctxt = False
            # print('ctxtmod_autoenc_model, num_parameters: ', getModelParametersCount(ctxtmod_autoenc_model))
            # print(ctxtmod_autoenc_model)
            # print_params(ctxtmod_autoenc_model)
            
        self.perf_com = MLP(mlp_param=perf_com_param, input_dim=perf_com_param['input_dim']+perf_com_param['geo_feature_dim']+perf_com_param['ctxt_feature_dim'], output_dim=perf_com_param['output_dim']).to(device)
        
        self.perf_sep_1 = MLP(mlp_param=perf_sep_param, input_dim=perf_sep_param['input_dim']+perf_sep_param['com_output_dim'], output_dim=perf_sep_param['output_dim']).to(device)
        if(perf_sep_param['n_networks']==2):
            self.perf_sep_2 = MLP(mlp_param=perf_sep_param, input_dim=perf_sep_param['input_dim']+perf_sep_param['com_output_dim'], output_dim=perf_sep_param['output_dim']).to(device)
            self.perf_sep = [self.perf_sep_1,self.perf_sep_2]
        else:
            self.perf_sep = [self.perf_sep_1]
            
        self.inc_var_com = True if(perf_com_param['input_dim'] != 0) else False
        self.inc_var_sep = True if(perf_sep_param['input_dim'] != 0) else False

        # print('perf_dec_model, num_parameters: ', getModelParametersCount(perf_dec_model))
        # print(perf_dec_model)
        # print_params(perf_dec_model)
        
        self.geo_ids = geo_ids
        if(len(geo_ids[0].split('_'))==2):
            self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
            #print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
        else:
            self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1]), int(g_name.split('_')[2])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1]),int(g_name.split('_')[2])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
            #print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
        
        self.geo_latents = geo_latents.to(device)
        #print(self.geo_latents.shape)
        
        if(rotEnc_model is not None):
            self.apply_rot = True
            self.rotation_enc = rotEnc_model.module.to(device) if(isinstance(rotEnc_model, nn.DataParallel)) else rotEnc_model.to(device)
            self.rotation_enc.eval()
        else:
            self.apply_rot = False
        
        
    def forward(self, inputs):
        #########
        
        geo_types = ['building','extrBuilding']
        idx = inputs['idx'].flatten()
        rotation = inputs['rot']
        geo_code = inputs['geo_code']
        ref_val = inputs['ref_val'].to(rotation.dtype) if(inputs['ref_val'] is not None) else inputs['ref_val']
        
        self.geo_ids_geo_code = self.geo_ids_geo_code.to(rotation.device)
        geo_code_rot = torch.cat([geo_code,rotation],-1) if(self.geo_ids_geo_code.shape[-1]==3) else geo_code
        
        # name_translations = np.array(['building','extrBuilding'])
        # g_numpy = geo_code_rot.detach().cpu().numpy().astype(int) 
        # print('Current geometries in batch: ' + str(np.array([name_translations[g[0]] + '_' + str(g[1]) + '_' + str(g[2]) for g in g_numpy])))
        
        self.equiv_ids = torch.sum(((torch.sum((geo_code_rot==self.geo_ids_geo_code).long(),-1)==self.geo_ids_geo_code.shape[-1])*torch.arange(self.geo_ids_geo_code.shape[0]).to(rotation.device).unsqueeze(0).T),0)
        #print('self.equiv_ids',self.equiv_ids.shape)
        self.geo_latents = self.geo_latents.to(rotation.device)
        #print('self.geo_latents',self.geo_latents.shape)
        
        # print('Equivalent geometries from the geometry set: ' + str(np.array(self.geo_ids)[self.equiv_ids.detach().cpu().numpy().astype(int)]))
        # print(" ")

        # Get the subset of geometry latents
        g_latents_sub = self.geo_latents[self.equiv_ids].to(rotation.device)
        #print('g_latents_sub',g_latents_sub.shape)
        
        # Forward pass through the rotation module
        if(self.apply_rot):
            if(rotation.float().mean()!=torch.tensor([0.]).to(coordinate.device)):
                #print('g_latents_sub',g_latents_sub.shape)
                #print('rotation',rotation.shape)
                con_input_rot = torch.cat([g_latents_sub, rotation/360], dim=-1)
                g_latents_sub = self.rotation_enc.module.mlp(con_input_rot) if(isinstance(self.rotation_enc, nn.DataParallel)) else self.rotation_enc.mlp(con_input_rot)
        
        # Forward pass through the contextualizing module
        ctxt_latent = None
        if(self.apply_ctxt):
            ctxt_latent = self.ctxtmod_autoenc(g_latents_sub)
            #print('ctxt_latent',ctxt_latent.shape)
                
        # Forward pass through the common perf decoder    
        dec_input = torch.cat([g_latents_sub, ctxt_latent], -1) if(self.apply_ctxt) else g_latents_sub
        outputs_com = self.perf_com(torch.cat([ref_val, dec_input],-1)) if(self.inc_var_com) else self.perf_com(dec_input)
        #print('outputs_com',outputs_com.shape)
        
        #print(torch.cat([torch.unsqueeze(self.perf_sep[i](outputs_com),1) for i in range(len(self.perf_sep))],1).shape)
        outputs = torch.cat([torch.unsqueeze(self.perf_sep[i](torch.cat([ref_val, outputs_com],-1)),1) for i in range(len(self.perf_sep))],1) if(self.inc_var_sep) else torch.cat([torch.unsqueeze(self.perf_sep[i](outputs_com),1) for i in range(len(self.perf_sep))],1)
        #print('outputs',outputs.shape)

        return outputs, ctxt_latent 
    
# class LatentNeuralPerformanceField(nn.Module):
#     def __init__(self, 
#                  ctxtmod_param, 
#                  perfdec_param, 
#                  geo_latents,
#                  geo_ids,
#                  rotEnc_model = None,
#                 ):
#         super().__init__()
#         #########

#         self.ctxtmod_autoenc = Autoencoder_MLP(autoenc_param = ctxtmod_param).to(device)
#         # print('ctxtmod_autoenc_model, num_parameters: ', getModelParametersCount(ctxtmod_autoenc_model))
#         # print(ctxtmod_autoenc_model)
#         # print_params(ctxtmod_autoenc_model)

#         self.perf_dec = MLP(mlp_param=perfdec_param, input_dim=perfdec_param['input_dim']+perfdec_param['geo_feature_dim']+perfdec_param['ctxt_feature_dim'], output_dim=perfdec_param['output_dim']).to(device)
#         # print('perf_dec_model, num_parameters: ', getModelParametersCount(perf_dec_model))
#         # print(perf_dec_model)
#         # print_params(perf_dec_model)
        
#         self.geo_ids = geo_ids
#         if(len(geo_ids[0].split('_'))==2):
#             self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
#             print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
#         else:
#             self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1]), int(g_name.split('_')[2])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1]),int(g_name.split('_')[2])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
#             print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
        
#         self.geo_latents = geo_latents.to(device)
#         print(self.geo_latents.shape)
        
#         if(rotEnc_model is not None):
#             self.apply_rot = True
#             self.rotation_enc = rotEnc_model.module.to(device) if(isinstance(rotEnc_model, nn.DataParallel)) else rotEnc_model.to(device)
#             self.rotation_enc.eval()
#         else:
#             self.apply_rot = False
        
        
#     def forward(self, inputs):
#         #########
        
#         geo_types = ['building','extrBuilding']
#         coordinate = inputs['xyz']
#         idx = inputs['idx'].flatten()
#         rotation = inputs['rot']
#         geo_code = inputs['geo_code']

#         # print('coordinate',coordinate.shape)
#         # print('idx',idx.shape)
#         # print('rotation',rotation.shape)
#         # print('geo_code',geo_code.shape)
        
#         # name_translations = np.array(['building','extrBuilding'])
#         # g_numpy = geo_code.detach().cpu().numpy().astype(int) 
#         # print('Current geometries in batch: ' + str(np.array([name_translations[g[0]] + '_' + str(g[1]) for g in g_numpy])))
        
#         self.geo_ids_geo_code = self.geo_ids_geo_code.to(coordinate.device)
#         #print('self.geo_ids_geo_code.shape',self.geo_ids_geo_code.shape)
#         geo_code_rot = torch.cat([geo_code,rotation],-1) if(self.geo_ids_geo_code.shape[-1]==3) else geo_code
#         self.equiv_ids = torch.sum(((torch.sum((geo_code_rot==self.geo_ids_geo_code).long(),-1)==self.geo_ids_geo_code.shape[-1])*torch.arange(self.geo_ids_geo_code.shape[0]).to(coordinate.device).unsqueeze(0).T),0)
#         #print('self.equiv_ids',self.equiv_ids.shape)
#         self.geo_latents = self.geo_latents.to(coordinate.device)
#         #print('self.geo_latents',self.geo_latents.shape)
        
#         # print('Equivalent geometries from the geometry set: ' + str(np.array(self.geo_ids)[self.equiv_ids.detach().cpu().numpy().astype(int)]))
#         # print(" ")

#         # Get the subset of geometry latents
#         g_latents_sub = self.geo_latents[self.equiv_ids].to(coordinate.device)
#         #print('g_latents_sub',g_latents_sub.shape)
        
#         # Forward pass through the rotation module
#         if(self.apply_rot):
#             if(rotation.float().mean()!=torch.tensor([0.]).to(coordinate.device)):
#                 #print('g_latents_sub',g_latents_sub.shape)
#                 #print('rotation',rotation.shape)
#                 con_input_rot = torch.cat([g_latents_sub, rotation/360], dim=-1)
#                 g_latents_sub = self.rotation_enc.module.mlp(con_input_rot) if(isinstance(self.rotation_enc, nn.DataParallel)) else self.rotation_enc.mlp(con_input_rot)
        
#         # Forward pass through the contextualizing module
#         ctxt_latent = self.ctxtmod_autoenc(g_latents_sub)
#         #print('ctxt_latent',ctxt_latent.shape)
        
#         # Tile the latents for all points
#         g_latents_sub = torch.unsqueeze(g_latents_sub, 1)
#         g_latents_sub = torch.tile(g_latents_sub, (1, coordinate.shape[1], 1))
#         ctxt_latent_t = torch.unsqueeze(ctxt_latent, 1)
#         ctxt_latent_t = torch.tile(ctxt_latent_t, (1, coordinate.shape[1], 1))
#         #print('ctxt_latent_t',ctxt_latent_t.shape)
                
#         # Forward pass through the perf decoder    
#         dec_input = torch.cat([g_latents_sub, ctxt_latent_t, coordinate], 2)
#         outputs = self.perf_dec(dec_input)
#         #print('outputs',outputs.shape)

#         return outputs, ctxt_latent 
    
    
    


    

# class LatentNeuralField_TwoParallelFields(nn.Module):
#     def __init__(self, 
#                  num_latents, 
#                  mlp_param_common,
#                  mlp_param_sdf, 
#                  mlp_param_perf, 
#                  in_dim = 3, 
#                  out_dim_sdf = 1, 
#                  out_dim_perf = 1,
#                 ):
#         super().__init__()
#         #########

#         # Generate an nn.Embedding with num_latents each of dimension as latent_dim.
#         self.latents = nn.Embedding(num_embeddings = num_latents, embedding_dim = mlp_param_common['feature_dim'])

#         # Initialize the embedding using a normal distribution.
#         self.latents.weight.data.normal_(0, mlp_param_common['latent_init'])   # training very sensitive to initialization so try changing the std

#         # Use nn.Sequential to generate an mlp with select parameters:
#         self.mlp_sdf = MLP(mlp_param=mlp_param_sdf, input_dim=in_dim+mlp_param_common['feature_dim'], output_dim=out_dim_sdf).to(device)
#         self.mlp_perf = MLP(mlp_param=mlp_param_perf, input_dim=in_dim+mlp_param_common['feature_dim'], output_dim=out_dim_perf).to(device)

#         self.latent_dim = mlp_param_common['feature_dim']
    
#     def forward(self, inputs):
#         #########

#         coordinate = inputs['xyz']
#         #print('coordinate',coordinate.shape)
#         idcs = inputs['idx']

#         # Using the input indices 'idcs', get the respective latents
#         latents = self.latents(idcs)
        
#         # Reshape the latents to (batch_size, latent_dim)
#         latents = torch.reshape(latents, (idcs.shape[0], self.latent_dim))
#         #print('latents',latents.shape)

#         # Tile the latents to all points
#         latents_r = torch.unsqueeze(latents, 1)
#        # print('latents',latents_r.shape)
#         latents_r = latents_r.repeat(1, coordinate.shape[1], 1)

#         # Concatenate points and latents for model prediction
#        # print('latents',latents.shape)
#        # print('coordinate',coordinate.shape)
#         con_inputs = torch.cat([latents_r, coordinate], dim=2)
#         #print('con_inputs',con_inputs.shape)

#         # Compute the output of decoder of the latents
#         outputs_sdf = self.mlp_sdf(con_inputs)
#         outputs_perf = self.mlp_perf(con_inputs)
#         #print('outputs',outputs.shape)

#         return outputs_sdf, outputs_perf, latents    


def instModelfromDir(model_dir, model_category='LatentNeuralField', model_type = "train", device = 'cpu', include_val=False, include_rot=False, return_ids=False, override_latent=None, geo_model_params = None, include_ctxt=True, dtype='train',otherdataset=""):   #model categories: LatentNeuralField/LatentRotationEncoder/LatentNeuralPerformanceField
    
    parameters_dir = model_dir + '/' + 'parameters'
    model = None
    geo_dataset_ids = None
   
    # Initialize model
    if(model_category=='LatentNeuralField'):
        val_add = 'val' if(include_val) else ''
        rot_add = 'rot' if(include_rot) else ''
        d_name = 'dataset_'+model_type+val_add+rot_add+'_param' if(model_type=='train') else 'dataset_param_'+otherdataset if(otherdataset != "") else'dataset_'+model_type+'_param'
        #print('d_name',d_name)
        mlp_param, dataset_param = loadModelParam(model_dir, param_ar = ['mlp_param', d_name])
        geo_dataset_ids = dataset_param['dataset_ids']
        num_latents = override_latent if(override_latent is not None) else len(geo_dataset_ids)
        model = LatentNeuralField(num_latents=num_latents, mlp_param = mlp_param).to(device)
    if(model_category=='LatentRotationEncoder'):
        mlp_param = loadModelParam(model_dir, param_ar = ['mlp_param'])[0]
        #print(loadModel(model_dir = mlp_param['geo_model_dir'], model_category='LatentNeuralField', model_type = mlp_param['geo_model_type'], load_type = mlp_param['geo_load_type'], epoch = mlp_param['geo_epoch'], device = device, include_val=include_val, return_ids=True))
        geo_model, geo_dataset_ids = loadModel(model_dir = mlp_param['geo_model_dir'].replace('/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF','/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF'), model_category='LatentNeuralField', model_type = mlp_param['geo_model_type'], load_type = mlp_param['geo_load_type'], epoch = mlp_param['geo_epoch'], device = device, include_val=include_val, return_ids=True, override_latent = mlp_param['override_latent_geo'])
        geo_model = geo_model.module if(isinstance(geo_model, nn.DataParallel)) else geo_model
        model = LatentRotationEncoder(mlp_param = mlp_param, latent_sdf_field = geo_model, geo_ids=geo_dataset_ids).to(device)
    if(model_category=='LatentNeuralPerformanceField'):
        ctxtmod_param, perfdec_param, dataset_param = loadModelParam(model_dir, param_ar = ['ctxtmod_param', 'perfdec_param', 'dataset_'+dtype+'_param'])
        geo_model_type = dataset_param['geo_model_type'] if(dtype != 'test') else 'test'
        geo_load_type = dataset_param['geo_load_type'] if(dtype != 'test') else 'final'
        geo_epoch = dataset_param['geo_epoch'] if(dtype != 'test') else 0
        geo_latents, geo_dataset_ids = loadLatents(dataset_param['geo_model_dir'].replace('/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF','/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF'), model_category='LatentNeuralField', model_type = geo_model_type, load_type = geo_load_type, epoch = geo_epoch, device = device, include_val=dataset_param['geo_include_val'], include_rot=dataset_param['geo_include_rot'], return_ids=True, override_latent=dataset_param['geo_override_latent'])
        if(dataset_param['rot_model_dir'] is not None):
            rotEnc_model = loadModel(dataset_param['rot_model_dir'].replace('/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF','/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF'), model_category='LatentRotationEncoder', model_type = 'train', load_type = dataset_param['rot_load_type'], epoch = dataset_param['rot_epoch'], device = device, override_latent=override_latent)
            rotEnc_model = rotEnc_model.module if(isinstance(rotEnc_model, nn.DataParallel)) else rotEnc_model
        else:
            rotEnc_model = None
        ctxtmod_param = ctxtmod_param if(include_ctxt) else None
        #print('ctxtmod_param',ctxtmod_param)
        model = LatentNeuralPerformanceField(ctxtmod_param, perfdec_param, geo_latents, list(geo_dataset_ids), rotEnc_model).to(device)
        #print(model)
    if(model_category=='LatentNeuralPerformanceFieldAggr_CDF'):
        ctxtmod_param, perf_com_param, perf_sep_param, dataset_param = loadModelParam(model_dir, param_ar = ['ctxtmod_param', 'perf_com_param', 'perf_sep_param', 'dataset_'+model_type+'_param'])
        geo_model_type = dataset_param['geo_model_type'] if(dtype != 'test') else 'test'
        geo_load_type = dataset_param['geo_load_type'] if(dtype != 'test') else 'final'
        geo_epoch = dataset_param['geo_epoch'] if(dtype != 'test') else 0
        geo_latents, geo_dataset_ids = loadLatents(dataset_param['geo_model_dir'].replace('/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF','/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF'), model_category='LatentNeuralField', model_type = geo_model_type, load_type = geo_load_type, epoch = geo_epoch, device = device, include_val=dataset_param['geo_include_val'], include_rot=dataset_param['geo_include_rot'], return_ids=True, override_latent=dataset_param['geo_override_latent'])
        if(dataset_param['rot_model_dir'] is not None):
            rotEnc_model = loadModel(dataset_param['rot_model_dir'].replace('/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF','/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF'), model_category='LatentRotationEncoder', model_type = 'train', load_type = dataset_param['rot_load_type'], epoch = dataset_param['rot_epoch'], device = device, override_latent=override_latent)
            rotEnc_model = rotEnc_model.module if(isinstance(rotEnc_model, nn.DataParallel)) else rotEnc_model
        else:
            rotEnc_model = None
        ctxtmod_param = ctxtmod_param if(include_ctxt) else None
        #print('ctxtmod_param',ctxtmod_param)
        model = LatentNeuralPerformanceFieldAggr_CDF(ctxtmod_param, perf_com_param, perf_sep_param, geo_latents, list(geo_dataset_ids), rotEnc_model).to(device)
        #print(model)
    
    if(return_ids):
        return model, geo_dataset_ids
    else:
        return model

    
def loadModel(model_dir, model_category='LatentNeuralField', model_type = 'train', load_type = 'current', epoch = None, device = 'cpu', include_val=False, include_rot=False, return_ids=False, override_latent=None, model_instance = None, include_ctxt=True, dtype='train', train_epoch=None,otherdataset=""): #model categories: LatentNeuralField/LatentRotationEncoder/LatentNeuralPerformanceField
    
    if(model_instance is None):
        outputs = instModelfromDir(model_dir, model_category, model_type, device, include_val, include_rot, return_ids, override_latent, include_ctxt=include_ctxt, dtype=dtype,otherdataset=otherdataset)
        if(return_ids):
            model, geo_dataset_ids = outputs
        else:
            model = outputs
    else:
        model = model_instance

    prefix_model = {'LatentNeuralField':'', 'LatentRotationEncoder':'rot_', 'LatentNeuralPerformanceField':'perf_', 'LatentNeuralPerformanceFieldAggr_CDF':'perf_'}
    model_dir =  model_dir + '/' + otherdataset if(otherdataset != "") else model_dir + '/'+ model_type +'/' if(('test' in model_type)and (model_category=='LatentNeuralField') and (train_epoch == None)) else model_dir + '/'+ 'test_sets/epoch_' + str(train_epoch) + '/' if(('test' in model_type)and (model_category=='LatentNeuralField')) else model_dir

    checkpoints_dir = model_dir + '/' + 'checkpoints'

    # Load the model
    epoch = epoch if(epoch is not None) else -1
    ext_str = {'current':'', 'final':'', 'epoch':'_%04d' % epoch}
    fol_base = {'current':model_dir, 'final':model_dir, 'epoch':checkpoints_dir}
    # print('load_type',load_type)
    # print('epoch',epoch)
    # print('model',model)
    file_to_load = os.path.join(fol_base[load_type], prefix_model[model_category] + 'model_'+load_type+ext_str[load_type]+'.pth')
    try:
        model.load_state_dict(torch.load(file_to_load)) if(str(device) != 'cpu') else model.load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu')))    
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(file_to_load)) if(str(device) != 'cpu') else model.load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu')))  
        
    if(return_ids):
        return model, geo_dataset_ids
    else:
        return model


def loadLatents(model_dir, model_category='LatentNeuralField', model_type = 'train', load_type = 'current', epoch = None, device = 'cpu', include_val=False, include_rot=False, return_ids = True, override_latent=None):
    
    
    outputs = loadModel(model_dir, model_category, model_type, load_type, epoch, device, include_val, include_rot, True, override_latent)
    model, geo_dataset_ids = outputs
    model.eval()
        
    latents = model.module.latents.weight.detach() if(isinstance(model, nn.DataParallel)) else model.latents.weight.detach()
    
    if(return_ids):
        return latents, geo_dataset_ids
    else:
        return latents





    

# def loadModel(model_dir, model_type = 'train', load_type = 'current', epoch_to_view = None, model_end = 'geo', device = 'cpu'):

#     parameters_dir = model_dir + '/' + 'parameters'
#     model = None
#     # Read parameter files
#     mlp_param, dataset_param = loadModelParam(model_dir, param_ar = ['mlp_param', 'dataset_'+model_type+'_param'])
#     if((mlp_param is None) or (dataset_param is None)):
#         print('Some parameters files do not exist.')
   
#     # Initialize model
#     max_num_instances = dataset_param['max_num_instances'] if(dataset_param['max_num_instances'] != -1) else dataset_param['total_instances']
#     model = LatentNeuralField(num_latents=max_num_instances, mlp_param = mlp_param).to(device)
#     model_dir = model_dir + '/test/' if(model_type != 'train') else model_dir
#     checkpoints_dir = model_dir + '/' + 'checkpoints'

#     # Load the model
#     epoch_to_view = epoch_to_view if(epoch_to_view is not None) else -1
#     ext_str = {'current':'', 'final':'', 'epoch':'_%04d' % epoch_to_view}
#     fol_base = {'current':model_dir, 'final':model_dir, 'epoch':checkpoints_dir}
#     file_to_load = os.path.join(fol_base[load_type], 'model_'+load_type+ext_str[load_type]+'.pth')
#     try:
#         model.load_state_dict(torch.load(file_to_load)) if(device != 'cpu') else model.load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu')))    
#     except:
#         model = torch.nn.DataParallel(model)
#         model.load_state_dict(torch.load(file_to_load)) if(device != 'cpu') else model.load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu')))  
        
#     return model



def instModelfromDir_OLD(model_dir, model_category='LatentNeuralField', model_type = "train", device = 'cpu', include_val=False, return_ids=False):   #model categories: LatentNeuralField/LatentRotationEncoder/LatentNeuralPerformanceField
    
    parameters_dir = model_dir + '/' + 'parameters'
    model = None
    geo_dataset_ids = None
   
    # Initialize model
    if(model_category=='LatentNeuralField'):
        mlp_param, dataset_param, dataset_val_param = loadModelParam(model_dir, param_ar = ['mlp_param', 'dataset_'+model_type+'_param', 'dataset_val_param'])
        geo_dataset_ids = dataset_param['dataset_ids'] 
        max_num_instances = dataset_param['max_num_instances'] if(dataset_param['max_num_instances'] != -1) else dataset_param['total_instances']
        if(include_val):
            xyz_sdf_dataset = XYZ_SDF_Dataset(directory=dataset_param['dataset_directory'],
                                    dataset_type=dataset_param['dataset_type'],
                                    subsets=dataset_param['dataset_subsets'],
                                    geo_types=dataset_param['dataset_geotypes'],
                                    sampling_types=dataset_param['sampling_types'],
                                    sampling_distr=dataset_param['sampling_distr'],
                                    distr=dataset_param['distr'],
                                    num_points=dataset_param['num_points'],
                                    s_range = dataset_param['s_range'],
                                    split_type = ['train','val'],
                                    max_instances = dataset_param['max_num_instances'])
            geo_dataset_ids = xyz_sdf_dataset.__getUniqueNames__()
            max_num_instances = len(xyz_sdf_dataset)
            model = LatentNeuralField(num_latents=max_num_instances, mlp_param = mlp_param).to(device)
    if(model_category=='LatentRotationEncoder'):
        mlp_param = loadModelParam(model_dir, param_ar = ['mlp_param'])[0]
        geo_model = loadModel(model_dir = mlp_param['geo_model_dir'], model_category='LatentNeuralField', model_type = mlp_param['geo_model_type'], load_type = mlp_param['geo_load_type'], epoch = mlp_param['geo_epoch'], device = device)
        geo_model = geo_model.module if(isinstance(geo_model, nn.DataParallel)) else geo_model
        model = LatentRotationEncoder(mlp_param = mlp_param, latent_sdf_field = geo_model).to(device)
    if(model_category=='LatentNeuralPerformanceField'):
        ctxtmod_param, perfdec_param, dataset_param = loadModelParam(model_dir, param_ar = ['ctxtmod_param', 'perfdec_param', 'dataset_'+model_type+'_param'])
        geo_latents, geo_dataset_ids = loadLatents(dataset_param['geo_model_dir'], model_category='LatentNeuralField', model_type = dataset_param['geo_model_type'], load_type = dataset_param['geo_load_type'], epoch = dataset_param['geo_epoch'], device = device, include_val=include_val, return_ids=True)
        rotEnc_model = loadModel(dataset_param['rot_model_dir'], model_category='LatentRotationEncoder', model_type = 'train', load_type = dataset_param['rot_load_type'], epoch = dataset_param['rot_epoch'], device = device)
        rotEnc_model = rotEnc_model.module if(isinstance(rotEnc_model, nn.DataParallel)) else rotEnc_model
        model = LatentNeuralPerformanceField(ctxtmod_param, perfdec_param, geo_latents, list(geo_dataset_ids), rotEnc_model).to(device)
    
    return model, geo_dataset_ids