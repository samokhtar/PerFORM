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
from building_sdf.learning.network_modules import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
    
#### PERF

class SpatialGroundLatentNeuralPerformanceField(nn.Module):
    def __init__(self, 
                 spatenc_param, 
                 perfdec_param, 
                 geo_ids,
                 geo_model_grid,
                 mode='bilinear',
                ):
        super().__init__()
        #########
        
        # Define the spatial encoder model
        self.mode = 'bilinear'
        input_shape = (2,spatenc_param['feature_dim'],spatenc_param['grid_size'],spatenc_param['grid_size'])
        #print('input_shape',input_shape)
        self.encoder_type = spatenc_param['encoder_type']
        params = spatenc_param['encoder_params']
        #print('params',params)

        if(self.encoder_type == 'unet'):
            norm_layer = get_norm_layer(norm_type=params['norm_layer_type'])
            self.encoder_model =  UnetGenerator(params['input_nc'], params['output_nc'], int(np.log(spatenc_param['grid_size'])/np.log(2)), params['ngf'], norm_layer=norm_layer, use_dropout=params['use_dropout'])
            #print('encoder_model',self.encoder_model)
        #else:
            #print('The encoder model does not exist: ' + self.encoder_type)
        #print('input_shape',input_shape)
        #print('input_shape',torch.rand(input_shape).shape)
        output_shape = self.encoder_model(torch.rand(input_shape)).shape
        #print('output_shape',output_shape)

        #print('The encoder model '+self.encoder_type+', num_parameters: ', getModelParametersCount(self.encoder_model))
        
        input_dim = 1 
        # Use GaussFFT for input encoding - if True
        self.apply_encoding = bool(perfdec_param['gauss_fft']) and (input_dim != 0)
        if(self.apply_encoding):
            #print('Input encoding ON')
            self.input_encoding = GaussianFourierFeatureTransform(input_dim, mapping_size=perfdec_param['gauss_fft_param'][0], scale=perfdec_param['gauss_fft_param'][1], seed=perfdec_param['seed'])
            input_dim = perfdec_param['gauss_fft_param'][0]*2
        
        self.perf_dec = MLP(mlp_param=perfdec_param, input_dim=input_dim+output_shape[1], output_dim=perfdec_param['output_dim']).to(device)
        #print('perf_dec_model, num_parameters: ', getModelParametersCount(self.perf_dec))
        #print(self.perf_dec)
      #  print_params(self.perf_dec)
        
        if(geo_ids is not None):
            self.geo_ids = geo_ids
            if(len(geo_ids[0].split('_'))==2):
                self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
                #print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
            else:
                self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1]), int(g_name.split('_')[2])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1]),int(g_name.split('_')[2])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
              #  print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
       # else:
         #   print('geo_ids are not defined')
        
        if(geo_model_grid is not None):
            self.geo_model_grid = geo_model_grid.to(device)
            self.geo_model_grid.eval()
        #else:
           # print('geo_model_grid is not defined!')
        
        
    def forward(self, inputs):
        #########
        
        geo_types = ['building','extrBuilding']
        coordinate = inputs['xyz']
        idx = inputs['idx'].flatten()
        rotation = inputs['rot']
        geo_code = inputs['geo_code']
        
        xy = coordinate[:,:,[0,2]]   # Note that in the dataset the Y and Z are interchanged
        xy = torch.unsqueeze(xy,-2)
        z = coordinate[:,:,1]     # Note that in the dataset the Y and Z are interchanged
        z = torch.unsqueeze(z, -1) 

        # print('coordinate',coordinate.shape)
        # print('idx',idx.shape)
        # print('rotation',rotation.shape)
        # print('geo_code',geo_code.shape)
        
        # Apply z fft input encoding
        if(self.apply_encoding):
            z = self.input_encoding(z)
        #print('z',z.shape)
        
        self.geo_ids_geo_code = self.geo_ids_geo_code.to(coordinate.device)
        #print('self.geo_ids_geo_code.shape',self.geo_ids_geo_code.shape)
        geo_code_rot = torch.cat([geo_code,rotation],-1) if(self.geo_ids_geo_code.shape[-1]==3) else geo_code
        
        # name_translations = np.array(['building','extrBuilding'])
        # g_numpy = geo_code_rot.detach().cpu().numpy().astype(int) 
        # print('Current geometries in batch: ' + str(np.array([name_translations[g[0]] + '_' + str(g[1]) + '_' + str(g[2]) for g in g_numpy])))
        
        self.equiv_ids = torch.sum(((torch.sum((geo_code_rot==self.geo_ids_geo_code).long(),-1)==self.geo_ids_geo_code.shape[-1])*torch.arange(self.geo_ids_geo_code.shape[0]).to(coordinate.device).unsqueeze(0).T),0)
        #print('self.equiv_ids',self.equiv_ids.shape)
        #print('self.geo_latents',self.geo_latents.shape)
        
        # print('Equivalent geometries from the geometry set: ' + str(np.array(self.geo_ids)[self.equiv_ids.detach().cpu().numpy().astype(int)]))
        # print(" ")

        # Get the grid latents    
        grid_output, param_latent = self.geo_model_grid({'xyz': coordinate,'idx': self.equiv_ids.to(device)})
        #print('grid_output',grid_output.shape)
        #print('param_latent',param_latent.shape)
        
        # Forward pass through the encoder
        perf_grid = self.encoder_model(grid_output)
        #print('perf_grid',perf_grid.shape)
        
        # Use grid_sample to get the values for the coordinates in the grid        
        values = nn.functional.grid_sample(perf_grid, xy, mode=self.mode, align_corners=True)
        
        # Use squeeze and permute to ensure the shape of values is (B, -1, L) where B is the batch size and L is the feature_dim
        values = torch.squeeze(torch.squeeze(values, -1), -1)
        values = torch.permute(values, (0,-1, 1))
        
        # Concatenate with z coordinate and infer the mlp on values and store it in variable values.
        dec_input = torch.cat((values, z),-1)

        # Forward pass for perf decoder
        outputs = self.perf_dec(dec_input)

        return outputs, perf_grid
    
    

class SpatialLatentNeuralPerformanceField(nn.Module):
    def __init__(self, 
                 spatenc_param, 
                 perfdec_param, 
                 geo_ids,
                 geo_model_grid,
                ):
        super().__init__()
        #########
        
        # Define the spatial encoder model
        input_shape = (2,spatenc_param['feature_dim'],spatenc_param['grid_size'],spatenc_param['grid_size']) if(spatenc_param['grid_type']=='ground') else (1,spatenc_param['feature_dim'],spatenc_param['grid_size'],spatenc_param['grid_size'])
        #print('input_shape',input_shape)
        self.encoder_type = spatenc_param['encoder_type']
        params = spatenc_param['encoder_params']

        if(self.encoder_type == 'transformer'):
            self.encoder_model = Transformer(grid_size=params['grid_size'],nhead=params['nhead'],dim_feedforward=params['dim_feedforward'], 
                                    num_layers=params['num_layers'],dropout=params['dropout'],norm_first=params['norm_first'])
        elif(self.encoder_type == 'densenet'):
            self.encoder_model = DenseNet(n_channels=params['n_channels'],growth_rate=params['growth_rate'],block_config=params['block_config'], 
                 num_init_features=params['num_init_features'],bn_size=params['bn_size'], drop_rate=params['drop_rate'])
        elif(self.encoder_type == 'resnet'):
            self.encoder_model = ResNet(n_channels=params['n_channels'],block_type=params['block_type'],layers=params['layers'], 
                   groups=params['groups'],width_per_group=params['width_per_group'])
        elif(self.encoder_type == 'ViT'):
            self.encoder_model = VisionTransformer(n_channels=params['n_channels'],image_size=params['image_size'],patch_size=params['patch_size'], 
                              num_layers=params['num_layers'],num_heads=params['num_heads'],hidden_dim=params['hidden_dim'], 
                              mlp_dim=params['mlp_dim'],dropout=params['dropout'],attention_dropout=params['attention_dropout'])
        #else:
            #print('The encoder model does not exist: ' + self.encoder_type)
        #print(self.encoder_model)
        output_shape = self.encoder_model(torch.rand(input_shape)).shape
        #print('output_shape',output_shape)

        #print('The encoder model '+self.encoder_type+', num_parameters: ', getModelParametersCount(self.encoder_model))
        
        input_dim = perfdec_param['input_dim']
        
        # Use GaussFFT for input encoding - if True
        self.apply_encoding = bool(perfdec_param['gauss_fft'])
        if(self.apply_encoding):
            #print('Input encoding ON')
            self.input_encoding = GaussianFourierFeatureTransform(input_dim, mapping_size=perfdec_param['gauss_fft_param'][0], scale=perfdec_param['gauss_fft_param'][1], seed=perfdec_param['seed'])
            input_dim = perfdec_param['gauss_fft_param'][0]*2
        
        self.perf_dec = MLP(mlp_param=perfdec_param, input_dim=input_dim+output_shape[-1], output_dim=perfdec_param['output_dim']).to(device)
        #print('perf_dec_model, num_parameters: ', getModelParametersCount(self.perf_dec))
        #print(self.perf_dec)
        #print_params(self.perf_dec)
        
        mode = '2d' if(spatenc_param['grid_type']=='ground') else '3d'
        self.pos_enc = CoordCat(mode)
        
        
        if(geo_ids is not None):
            self.geo_ids = geo_ids
            if(len(geo_ids[0].split('_'))==2):
                self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
                #print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
            else:
                self.geo_ids_geo_code = torch.from_numpy(np.concatenate([[[0, int(g_name.split('_')[1]), int(g_name.split('_')[2])]] if(g_name.split('_')[0]=='building') else [[1,int(g_name.split('_')[1]),int(g_name.split('_')[2])]] for g_name in self.geo_ids],0)).unsqueeze(0).permute(1,0,2)  
                #print('self.geo_ids_geo_code',self.geo_ids_geo_code.shape)
       # else:
          #  print('geo_ids are not defined!')
        
        if(geo_model_grid is not None):
            self.geo_model_grid = geo_model_grid.to(device)
            self.geo_model_grid.eval()
        #else:
           # print('geo_model_grid is not defined!')

        
        
    def forward(self, inputs):
        #########
        
        geo_types = ['building','extrBuilding']
        coordinate = inputs['xyz']
        idx = inputs['idx'].flatten()
        rotation = inputs['rot']
        geo_code = inputs['geo_code']

        # print('coordinate',coordinate.shape)
        # print('idx',idx.shape)
        # print('rotation',rotation.shape)
        # print('geo_code',geo_code.shape)
        
        # Apply coordinate fft input encoding
        if(self.apply_encoding):
            coordinate = self.input_encoding(coordinate)
        #print('coordinate',coordinate.shape)
        
        self.geo_ids_geo_code = self.geo_ids_geo_code.to(coordinate.device)
        #print('self.geo_ids_geo_code.shape',self.geo_ids_geo_code.shape)
        geo_code_rot = torch.cat([geo_code,rotation],-1) if(self.geo_ids_geo_code.shape[-1]==3) else geo_code
        
        # name_translations = np.array(['building','extrBuilding'])
        # g_numpy = geo_code_rot.detach().cpu().numpy().astype(int) 
        # print('Current geometries in batch: ' + str(np.array([name_translations[g[0]] + '_' + str(g[1]) + '_' + str(g[2]) for g in g_numpy])))
        
        self.equiv_ids = torch.sum(((torch.sum((geo_code_rot==self.geo_ids_geo_code).long(),-1)==self.geo_ids_geo_code.shape[-1])*torch.arange(self.geo_ids_geo_code.shape[0]).to(coordinate.device).unsqueeze(0).T),0)
        #print('self.equiv_ids',self.equiv_ids.shape)
        #print('self.geo_latents',self.geo_latents.shape)
        
        # print('Equivalent geometries from the geometry set: ' + str(np.array(self.geo_ids)[self.equiv_ids.detach().cpu().numpy().astype(int)]))
        # print(" ")

        # Get the grid latents    
        self.geo_model_grid.eval()
        grid_output, param_latent = self.geo_model_grid({'xyz': coordinate,'idx': self.equiv_ids.to(device)})
        #print('grid_output',grid_output.shape)
        #print('param_latent',param_latent.shape)
        
        # # Tile the latents for all points
        # param_latent = param_latent.permute(0,2,1)
        # param_latent = torch.tile(param_latent, (1, coordinate.shape[1], 1))
        # print('param_latent',param_latent.shape)
        
        # #Add positional encodings
        # if(self.encoder_type == 'transformer'):
        #     grid_output = self.pos_enc(grid_output)
        #     #print('grid_output',grid_output.shape)
        #     grid_output = grid_output.reshape(grid_output.shape[0],grid_output.shape[1],-1)
        #     #print('grid_output',grid_output.shape)
        
        # Forward pass through the encoder
        features = self.encoder_model(grid_output)
        #print('features',features.shape)
        
        # Tile the latents for all points
        features_t = torch.unsqueeze(features, 1)
        #print('features_t',features_t.shape)
        features_t = torch.tile(features_t, (1, coordinate.shape[1], 1))
        #print('features_t',features_t.shape)
        
        # Forward pass through the perf decoder    
        dec_input = torch.cat([features_t, coordinate], -1)
        #print('dec_input',dec_input.shape)
        outputs = self.perf_dec(dec_input)
        #print('outputs',outputs.shape)

        return outputs, features
    
    
    
    
    
#### VOXEL

class HybridVoxelLatentGrid_Grid(nn.Module):
    def __init__(self, grid_param):
        super().__init__()

        # Generate an nn.Parameter that stores a tensor of shape (1, feature_dim, resolution_per_dim[0], resolution_per_dim[1], resolution_per_dim[2]).
        self.latent_grid = nn.Parameter(torch.rand(1, grid_param['feature_dim'], grid_param['grid_size'], grid_param['grid_size'], grid_param['grid_size'])*grid_param['grid_latent_init'])
    
    def forward(self, inputs):

        # Get coordinates and reshape to match grid shape
        coordinate = inputs['xyz']

        coord = torch.unsqueeze(torch.unsqueeze(coordinate, -2), -3) 
        
        # Extract grid from the model parameters
        grid = self.latent_grid
        
        # Tile the latent grid in case we're fitting a single object but have a batch size larger than 1.
        if self.latent_grid.shape[0] != coord.shape[0]: 
            grid = self.latent_grid.repeat(coord.shape[0], 1, 1, 1, 1)
        
        return grid
    
class LatenttoHybridVoxelLatentGrid_ConvDecoder(nn.Module):
    def __init__(self, grid_param):
        super().__init__()

        # Generate a feature grid that stores a tensor of shape (1, feature_dim, resolution_per_dim[0], resolution_per_dim[1], resolution_per_dim[2]).
        num_up = int(np.log2(grid_param['grid_size']))
        self.grid = LatentFeatureGrid(latent_sidelength=1, latent_ch=grid_param['feature_dim'], num_up=num_up, out_ch=grid_param['feature_dim'], latent_init=grid_param['grid_latent_init'], mode='3d', neg_slope=grid_param['neg_slope'])
        
    
    def forward(self, inputs):
        
        # Get coordinates and reshape to match grid shape
        coordinate = inputs['xyz']
        
        coord = torch.unsqueeze(torch.unsqueeze(coordinate, -2), -3) 
        
        # Call self.grid to decode latent code into 3D grid.
        grid = self.grid()

        # Tile the latent grid in case we're fitting a single object but have a batch size larger than 1.
        if grid.shape[0] != coord.shape[0]: 
            grid = grid.repeat(coord.shape[0], 1, 1, 1, 1)
        
        return grid
    
class LatentHybridVoxelNeuralField(nn.Module):   # From voxel latent grid to features
    def __init__(self, grid_param, mode='bilinear'):
        super().__init__()
        
        # Set the interpolation mode
        self.mode = mode

        # Define the mlp parameters dictionary     
        mlp_param = {}
        mlp_param['n_hidden_layers'] = 0
        mlp_param['n_hidden_neurons'] = grid_param['feature_dim']
        mlp_param['nonlinearity'] = grid_param['grid_mlp_nonlinearity']
        mlp_param['outermost_nonlinearity'] = grid_param['grid_mlp_outermost_nonlinearity']
        mlp_param['outermost_linear'] = grid_param['grid_mlp_outermost_linear']
        self.mlp = MLP_Simple(mlp_param, input_dim=grid_param['feature_dim'], output_dim=grid_param['feature_dim'])
    
    def forward(self, inputs, grid):

        coordinate = inputs['xyz']
        
        coord = coordinate
        coord = torch.unsqueeze(torch.unsqueeze(coordinate, -2), -3) 

        # Use grid_sample to get the values for the coordinates in the grid
        values = nn.functional.grid_sample(grid, coord, mode=self.mode, align_corners=True)    

        # Use squeeze and permute to ensure the shape of values is (B, -1, L) where B is the batch size and L is the feature_dim
        values = torch.squeeze(torch.squeeze(values, -1), -1)
        values = torch.permute(values, (0,-1, 1))

        # Compute features by evaluating the mlp on the values
        features = self.mlp(values)
        return features
    

#### GROUND

class HybridGroundLatentGrid_Grid(nn.Module):
    def __init__(self, grid_param):
        super().__init__()

        # Generate an nn.Parameter that stores a tensor of shape (1, feature_dim, resolution_per_dim[0], resolution_per_dim[1]).
        self.latent_grid = nn.Parameter(torch.rand(1, grid_param['feature_dim'], grid_param['grid_size'], grid_param['grid_size'])*grid_param['grid_latent_init'])
        
    
    def forward(self, inputs):

        # Get coordinates, project coordinate onto xy-plane and reshape to match grid shape
        coordinate = inputs['xyz']
                
        xy = coordinate[:,:,[0,2]]   # Note that in the dataset the Y and Z are interchanged
        xy = torch.unsqueeze(xy,-1)

        # Extract grid from the model parameters
        grid = self.latent_grid

        # Tile the latent grid in case we're fitting a single object but have a batch size larger than 1.
        if self.latent_grid.shape[0] != coord.shape[0]: 
            grid = self.latent_grid.repeat(coord.shape[0], 1, 1, 1)
        
        return grid
    
    
class LatenttoHybridGroundLatentGrid_ConvDecoder(nn.Module):
    def __init__(self, grid_param):
        super().__init__()

        # Generate a feature grid that stores a tensor of shape (1, feature_dim, resolution_per_dim[0], resolution_per_dim[1]).
        num_up = int(np.log2(grid_param['grid_size']))
        self.grid = LatentFeatureGrid(latent_sidelength=1, latent_ch=grid_param['feature_dim'], num_up=num_up, out_ch=grid_param['feature_dim'], latent_init=grid_param['grid_latent_init'], mode='2d')
        
    
    def forward(self, inputs):

        # Get coordinates, project coordinate onto xy-plane and reshape to match grid shape
        coordinate = inputs['xyz']
        
        xy = coordinate[:,:,[0,2]]   # Note that in the dataset the Y and Z are interchanged
        xy = torch.unsqueeze(xy,-1)
        
        # Call self.grid to decode latent code into 2D grid.
        grid = self.grid()

        # Tile the latent grid in case we're fitting a single object but have a batch size larger than 1.
        if grid.shape[0] != xy.shape[0]: 
            grid = grid.repeat(xy.shape[0], 1, 1, 1)
        
        return grid


class LatentHybridVoxelNeuralField(nn.Module):   # From voxel latent grid to features
    def __init__(self, grid_param, mode='bilinear'):
        super().__init__()
        
        # Set the interpolation mode
        self.mode = mode

        # Define the mlp parameters dictionary     
        mlp_param = {}
        mlp_param['n_hidden_layers'] = 0
        mlp_param['n_hidden_neurons'] = grid_param['feature_dim']
        mlp_param['nonlinearity'] = grid_param['grid_mlp_nonlinearity']
        mlp_param['outermost_nonlinearity'] = grid_param['grid_mlp_outermost_nonlinearity']
        mlp_param['outermost_linear'] = grid_param['grid_mlp_outermost_linear']
        self.mlp = MLP_Simple(mlp_param, input_dim=grid_param['feature_dim'], output_dim=grid_param['feature_dim'])

    
    def forward(self, inputs, grid):

        coordinate = inputs['xyz']
        
        coord = coordinate
        coord = torch.unsqueeze(torch.unsqueeze(coordinate, -2), -3) 

        # Use grid_sample to get the values for the coordinates in the grid
        values = nn.functional.grid_sample(grid, coord, mode=self.mode, align_corners=True)    

        # Use squeeze and permute to ensure the shape of values is (B, -1, L) where B is the batch size and L is the feature_dim
        values = torch.squeeze(torch.squeeze(values, -1), -1)
        values = torch.permute(values, (0,-1, 1))

        # Compute features by evaluating the mlp on the values
        features = self.mlp(values)
        return features
    
    
class LatentHybridGroundNeuralField(nn.Module):
    def __init__(self, grid_param, mode='bilinear'):
        super().__init__()
        
        # Set the interpolation mode
        self.mode = mode

        # Define the mlp parameters dictionary     
        mlp_param = {}
        mlp_param['n_hidden_layers'] = 0
        mlp_param['n_hidden_neurons'] = np.maximum(grid_param['feature_dim'],128)
        mlp_param['nonlinearity'] = grid_param['grid_mlp_nonlinearity']
        mlp_param['outermost_nonlinearity'] = grid_param['grid_mlp_outermost_nonlinearity']
        mlp_param['outermost_linear'] = grid_param['grid_mlp_outermost_linear']
        self.mlp = MLP_Simple(mlp_param, input_dim=grid_param['feature_dim'] + 1, output_dim=np.maximum(grid_param['feature_dim'],128))
        
    
    def forward(self, inputs, grid):

        coordinate = inputs['xyz']
        
        xy = coordinate[:,:,[0,2]]   # Note that in the dataset the Y and Z are interchanged
        xy = torch.unsqueeze(xy,-2)
        z = coordinate[:,:,1]     # Note that in the dataset the Y and Z are interchanged
        z = torch.unsqueeze(z, -1) 

        # Use grid_sample to get the values for the coordinates in the grid        
        values = nn.functional.grid_sample(grid, xy, mode=self.mode, align_corners=True)
        
        # Use squeeze and permute to ensure the shape of values is (B, -1, L) where B is the batch size and L is the feature_dim
        values = torch.squeeze(torch.squeeze(values, -1), -1)
        values = torch.permute(values, (0,-1, 1))
        
        # Concatenate with z coordinate and infer the mlp on values and store it in variable values.
        values = torch.cat((values, z),-1)

        # Compute features by evaluating the mlp on the values
        features = self.mlp(values)
        return features
    

#### COMMON

class LatentField(nn.Module):
    def __init__(self, grid_param, field_param):
        super().__init__()

        # Define the scene_rep which takes the latent grid as an input and outputs feature channels
        if(grid_param['grid_type']=='voxel'):
            self.scene_rep = LatentHybridVoxelNeuralField(grid_param, mode='bilinear')
        elif(grid_param['grid_type']=='ground'):
            self.scene_rep = LatentHybridGroundNeuralField(grid_param, mode='bilinear')
        else:
            self.scene_rep = None
            
        # Define the field mlp which takes the grid features and outputs scalar value(s) 
        mlp_param = {}
        mlp_param['n_hidden_layers'] = 0
        mlp_param['n_hidden_neurons'] =  np.maximum(grid_param['feature_dim'],128)
        mlp_param['nonlinearity'] = field_param['field_mlp_nonlinearity']
        mlp_param['outermost_nonlinearity'] = field_param['field_mlp_outermost_nonlinearity']
        mlp_param['outermost_linear'] = field_param['field_mlp_outermost_linear']
        self.field = MLP_Simple(mlp_param, input_dim=np.maximum(grid_param['feature_dim'],128), output_dim=field_param['field_mlp_out_dim'])


    def forward(
        self, 
        inputs: torch.Tensor,
        grid:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

      # Do a forward pass through the scene representation neural field.
        features = self.scene_rep(inputs, grid)
      # Do a forward pass through both the self.field to yield the sdf
        field = self.field(features) 

        return field
    
    
class LatentFeatureGrid(nn.Module):
    def __init__(self, latent_sidelength, latent_ch, num_up, out_ch, latent_init, mode='2d', neg_slope=0.2):
        super().__init__()

        # Define the feature grid mode (dimension)
        if mode == '2d':
            latent_shape = 1, latent_ch, latent_sidelength, latent_sidelength
        elif mode == '3d':
            latent_shape = 1, latent_ch, latent_sidelength, latent_sidelength, latent_sidelength

        # Generate an nn.Parameter with shape (1, out_dim, *resolution_per_dim)
        self.latent = nn.Parameter(torch.rand(latent_shape)* latent_init)

        # Instatiate decoder as ConvDecoder which takes in latent_ch channels as
        # input and output, and the num_up and mode specified in the arguments.
        self.decoder = ConvDecoder(latent_ch, latent_ch, num_up, out_ch, mode, neg_slope=neg_slope)


    def forward(self, input=None):
        '''
        coordinate: (batch_size, num_points, 2)
        '''
        return(self.decoder(self.latent))
        # return the output of the decoder on input of self.latent. 
        
        
        
def instModelfromDir(model_dir, model_category='HybridLatentNeuralField', model_type = "train", device = 'cpu', include_val=False, include_rot=False, return_ids=False, override_latent=None, dtype='train',otherdataset=""):   #model categories: LatentNeuralField/LatentRotationEncoder/LatentNeuralPerformanceField
    
    parameters_dir = model_dir + '/' + 'parameters'
    model = None
    geo_dataset_ids = None
   
    # Initialize model
    if(model_category=='HybridLatentNeuralField'):
        #print('Why!, HybridLatentNeuralField')
        val_add = 'val' if(include_val) else ''
        rot_add = 'rot' if(include_rot) else ''
        #d_name = 'dataset_'+model_type+val_add+rot_add+'_param' if(model_type=='train') else 'dataset_'+model_type+'_param'
        d_name = 'dataset_'+model_type+val_add+rot_add+'_param' if(model_type=='train') else 'dataset_param_'+otherdataset if(otherdataset != "") else'dataset_'+model_type+'_param'
        #print('d_name',d_name)
        grid_param, field_param, dataset_param = loadModelParam(model_dir, param_ar = ['grid_param', 'field_param', d_name])
        geo_dataset_ids = dataset_param['dataset_ids']
        num_latents = override_latent if(override_latent is not None) else len(geo_dataset_ids)
        model_grid = LatenttoHybridGroundLatentGrid_ConvDecoder(grid_param=grid_param).to(device) if(grid_param['grid_type'] == 'ground') else LatenttoHybridVoxelLatentGrid_ConvDecoder(grid_param=grid_param).to(device)
        max_num_instances = dataset_param['max_num_instances'] if(dataset_param['max_num_instances']!=-1) else dataset_param['total_instances']
        model_grid = AutoDecoderWrapper(max_num_instances, model_grid, param_name='grid.latent', in_wgt=grid_param['latent_init']).to(device)
        model_field = LatentField(grid_param=grid_param, field_param=field_param).to(device)
        models = [model_grid, model_field]
    if(model_category=='SpatialPerformanceField'):  
        #print('HEYYYYYYYY!, SpatialPerformanceField')
        spatenc_param, perfdec_param, dataset_param = loadModelParam(model_dir, param_ar = ['spatenc_param', 'perfdec_param', 'dataset_'+dtype+'_param'])
        #print('dataset_param',dataset_param.keys())
        geo_model_type = dataset_param['geo_model_type'] if(dtype != 'test') else 'test'
        load_type = dataset_param['geo_load_type'] if(dtype != 'test') else 'final'
        epoch = dataset_param['geo_epoch'] if(dtype != 'test') else 0
        geo_latents, geo_dataset_ids = loadLatents(dataset_param['geo_model_dir'].replace('/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF','/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF'), model_type = geo_model_type, load_type = load_type, epoch = epoch, device = device, include_val=dataset_param['geo_include_val'], include_rot=dataset_param['geo_include_rot'], return_ids=True, model_category='HybridLatentNeuralField')
        model_grid, model_field = loadModels(dataset_param['geo_model_dir'].replace('/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF','/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF'), model_type = dataset_param['geo_model_type'], load_type = dataset_param['geo_load_type'], epoch = dataset_param['geo_epoch'], include_val=dataset_param['geo_include_val'], include_rot=dataset_param['geo_include_rot'], device = device, model_category='HybridLatentNeuralField')
        model_grid = model_grid.module if(isinstance(model_grid, nn.DataParallel)) else model_grid
        model_grid.eval()
        perf_type = 'global' if('perf_type' not in spatenc_param.keys()) else spatenc_param['perf_type']
        models = [SpatialLatentNeuralPerformanceField(spatenc_param, perfdec_param, list(geo_dataset_ids), model_grid).to(device)] if(perf_type == 'global') else [SpatialGroundLatentNeuralPerformanceField(spatenc_param, perfdec_param, list(geo_dataset_ids), model_grid).to(device)]
    
    if(return_ids):
        return models, geo_dataset_ids
    else:
        return models

def loadModels(model_dir, model_category='HybridLatentNeuralField', model_type = 'train', load_type = 'current', epoch = None, include_val=False, include_rot=False, model_end = 'geo', device = 'cpu', return_ids = False, dtype = 'train',otherdataset="", train_epoch=None):
    
    #print('model_category',model_category)
    #print('device',device)
    
    # Initialize models
    models, geo_dataset_ids = instModelfromDir(model_dir, model_category = model_category, model_type = model_type, device = device, include_val=include_val, include_rot=include_rot, return_ids=True, dtype=dtype, otherdataset=otherdataset)
    #print('initialized_models',models)

    # Set model directories
    prefix_model = {'HybridLatentNeuralField':'', 'SpatialPerformanceField':'perf_'}
    #model_dir =  model_dir + '/' + otherdataset if(otherdataset != "") else model_dir + '/'+ model_type +'/' if(('test' in model_type)and (train_epoch == None)) else model_dir + '/'+ 'test_sets/epoch_' + str(train_epoch) + '/' if(('test' in model_type)) else model_dir
    model_dir =  model_dir + '/' + otherdataset if(otherdataset != "") else model_dir + '/'+ model_type +'/' if(('test' in model_type)and (model_category=='HybridLatentNeuralField') and (train_epoch == None)) else model_dir + '/'+ 'test_sets/epoch_' + str(train_epoch) + '/' if(('test' in model_type)and (model_category=='HybridLatentNeuralField')) else model_dir
    #model_dir = model_dir + '/test/' if(model_type != 'train') else model_dir
    checkpoints_dir = model_dir + '/' + 'checkpoints'

    # Load the models
    model_names = ['model_grid', 'model_field'] if(model_category=='HybridLatentNeuralField') else ['perf_model']
    ext_str = {'current':'', 'final':'', 'epoch':'_%04d' % epoch}
    fol_base = {'current':model_dir, 'final':model_dir, 'epoch':checkpoints_dir}
    
    for i in range(len(models)):
        file_to_load = os.path.join(fol_base[load_type], model_names[i]+'_'+load_type+ext_str[load_type]+'.pth')

        try:
            models[i].load_state_dict(torch.load(file_to_load)) if(str(device) != 'cpu') else models[i].load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu'))) 
        except:
            models[i] = torch.nn.DataParallel(models[i])
            models[i].load_state_dict(torch.load(file_to_load)) if(str(device) != 'cpu') else models[i].load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu')))  

        if(dtype == 'test'):
            dataset_param = loadModelParam(model_dir, param_ar = ['dataset_'+dtype+'_param'])[0]
           # geo_latents, geo_dataset_ids = loadLatents(dataset_param['geo_model_dir'], model_type = 'test', load_type = 'final', epoch = 0, device = device, include_val=True, include_rot=True, return_ids=True,model_category='HybridLatentNeuralField')
            model_grid, model_field = loadModels(dataset_param['geo_model_dir'].replace('/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF','/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF'), model_type = 'test', load_type = 'final', epoch = 0, include_val=True, include_rot=True, device = device, model_category='HybridLatentNeuralField')
            model_grid = model_grid.module if(isinstance(model_grid, nn.DataParallel)) else model_grid
            model_grid.eval()
           # models[i].geo_ids = list(geo_dataset_ids)
            models[i].geo_model_grid = model_grid

        
    if(return_ids):
        return models, geo_dataset_ids
    else:
        return models
    

    
def loadLatents(model_dir, model_category='HybridLatentNeuralField', model_type = 'train', load_type = 'current', epoch = None, device = 'cpu', include_val=False, include_rot=False, return_ids = True):
    
    outputs = loadModels(model_dir, model_category, model_type, load_type, epoch, include_val, include_rot, device=device, return_ids=True)
    models, geo_dataset_ids = outputs
    for i in range(len(models)):
        models[i].eval()
        
    latents = models[0].module.latents.weight.detach() if(isinstance(models[0], nn.DataParallel)) else models[0].latents.weight.detach()
    
    if(return_ids):
        return latents, geo_dataset_ids
    else:
        return latents
    
    # try:
    #     model.load_state_dict(torch.load(file_to_load)) if(str(device) != 'cpu') else model.load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu')))    
    # except:
    #     model = torch.nn.DataParallel(model)
    #     model.load_state_dict(torch.load(file_to_load)) if(str(device) != 'cpu') else model.load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu')))  