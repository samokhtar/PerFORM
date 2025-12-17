# Visualization libraries
import numpy as np
from plotly.subplots import make_subplots
import plotly
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib
from matplotlib import pyplot as plt
from IPython.display import display, clear_output
import matplotlib.image as mpimg
import ipywidgets as widgets
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.cm as cm
import seaborn as sns
from building_sdf.sampling import *
import trimesh
import torch
import plotly.io as pio
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
# Define colors and color maps
colors_set_4 = [(12,44,132),(28,80,158),(31,118,179),(36,152,192)]
colors_set_8 = [(12,44,132),(28,80,158),(31,118,179),(36,152,192),(63,180,195),(107,197,189),(157,216,184),(204,235,179)] 
colors_set_8_long = [(12,44,132),(20,62,145),(28,80,158),(30,99,169),(31,118,179),(33,135,186),(36,152,192),(49,166,193),(63,180,195),(107,197,189),(155,216,184),(204,235,179),(180,225,181),(157,216,184),(255,255,255)] 
colors_set_3_unique = [(12,44,132),(63,180,195),(157,216,184)]
colors_set_utci_9 = [(249,65,68),(243,114,44),(248,150,30),(249,199,79),(144,190,109),(67,170,139),(77,144,172),(87,117,144),(39,125,161)]
color_maps = [colors_set_4, colors_set_8, colors_set_8_long, colors_set_3_unique, colors_set_utci_9]
color_map_names = ['colors_set_4', 'colors_set_8', 'colors_set_8_long', 'colors_set_3_unique', 'colors_set_utci_9']
remap_color_maps = {}
for i in range(len(color_maps)):
    remap_color_maps[color_map_names[i]]=[(color_maps[i][j][0]/255, color_maps[i][j][1]/255, color_maps[i][j][2]/255) for j in range(len(color_maps[i]))]
    
blueGreen = ListedColormap(remap_color_maps['colors_set_8'])
blueGreen = LinearSegmentedColormap.from_list(name='blueGreen',colors=remap_color_maps['colors_set_8'], N=256)
RdBu = plt.colormaps['RdBu'].resampled(256)


def visualize_mesh(vertices: np.ndarray, faces: np.ndarray, flatshading: bool = False, height=800, width=800, color="pink"):
    x, y, z = vertices.T
    i, j, k = faces.T
    fig = go.Figure(
        [
            go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                color=color
            ),
        ]
    )

    # styling
    fig.update_traces(
        flatshading=flatshading, lighting=dict(specular=1.0), selector=dict(type="mesh3d")
    )
    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5))
    fig.update_layout(scene_aspectmode="data")
    fig.update_layout(
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        scene_camera=dict(eye=dict(x=0.5, y=4, z=1.5)),
        width=width, height=height
    )

    return fig

# Function to plot sampled points
def plotSDFPoints(points, SDF_values, distance_visibility_threshold_norm, bbox):
    scatter_plot = go.Scatter3d(x = points[:,0][np.abs(SDF_values)<distance_visibility_threshold_norm], 
                                y = points[:,2][np.abs(SDF_values)<distance_visibility_threshold_norm], 
                                z = points[:,1][np.abs(SDF_values)<distance_visibility_threshold_norm], 
                                mode='markers',
                                marker=dict(
                                    color=np.abs(SDF_values[np.abs(SDF_values)<distance_visibility_threshold_norm]),
                                    colorscale='Plasma',
                                    size=3,
                                    opacity=0.8,
                                    cmin = 0,
                                    cmax = distance_visibility_threshold_norm,
                                    showscale = True
                                ),
                                )
    fig = go.Figure(scatter_plot)
    fig.update_scenes(xaxis_title_text='X',  
                      yaxis_title_text='Y',  
                      zaxis_title_text='Z',
                      xaxis_showbackground=False,
                      yaxis_showbackground=False,
                      zaxis_showbackground=False,
                      )
    fig.update_layout(
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=True, range=bbox[0]),
            yaxis=dict(visible=True, range=bbox[2]),
            zaxis=dict(visible=True, range=bbox[1]),
        ),
    )
    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5),height=700,showlegend=True)
    return fig

# Get mesh points subsample
def getMeshPointsSubSample(num_points, mesh_file, bbox, path_meshes, path_samples, sampling_type):

    bldg_mesh = trimesh.load_mesh(path_meshes + '/' +  mesh_file)

    # Generate centered mesh
    mesh_box = get_buffered_bbox(bldg_mesh, bbox_relative_buffer=0)
    bldg_c = (np.mean(mesh_box, axis=1))
    b_center_shift = np.expand_dims(np.array([-bldg_c[0], 0, -bldg_c[2]]), axis=0)
    centred_mesh = trimesh.Trimesh((bldg_mesh.vertices+b_center_shift), bldg_mesh.faces, process=False)

    bldg_sdf = path_samples + '/' + sampling_type + '/' + mesh_file.split('.')[0] + '_'+sampling_type+'.npz'

    s_sdf = np.load(bldg_sdf)
    d_sdf = s_sdf['allv']
    total_num_points = len(d_sdf)
    random_indices = np.unique(np.random.randint(total_num_points, size=num_points))
    sub_sample_to_plot = d_sdf[random_indices]

    # Calculate point density
    c = density(sub_sample_to_plot[:,0:3])

    # Define input parameters
    XYZ = (sub_sample_to_plot[:,0:3]*(bbox[:,1]-bbox[:,0]))+bbox[:,0]
    D = sub_sample_to_plot[:,3]

    return centred_mesh, XYZ, D, c

# Function to plot sampled points and mesh
def plotPointsMesh(mesh, points, values, bbox, cmin_v = None, cmax_v = None, flatshading: bool = False):

    x, y, z = mesh.vertices.T
    i, j, k = mesh.faces.T
    maxValue = np.max(values)
    cmin_v = cmin_v if(cmin_v is not None) else 0
    cmax_v = cmax_v if(cmax_v is not None) else 1
    
    mesh_plot = go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                color='white',
                opacity=0.2,
            )
    
    scatter_plot = go.Scatter3d(x = points[:,0], 
                                y = points[:,2], 
                                z = points[:,1], 
                                mode='markers',
                                marker=dict(
                                    color=density/maxDensity,
                                    colorscale='Plasma',
                                    size=2,
                                    opacity=0.8,
                                    cmin = cmin_v,
                                    cmax = cmax_v,
                                    showscale = True
                                ),
                                )

    fig = go.Figure([mesh_plot, scatter_plot])

    # styling
    fig.update_traces(
        flatshading=flatshading, lighting=dict(specular=1.0), selector=dict(type="mesh3d")
    )
    fig.update_scenes(xaxis_title_text='X',  
                      yaxis_title_text='Y',  
                      zaxis_title_text='Z',
                      xaxis_showbackground=False,
                      yaxis_showbackground=False,
                      zaxis_showbackground=False,
                      )
    fig.update_layout(
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=True, range=bbox[0]),
            yaxis=dict(visible=True, range=bbox[2]),
            zaxis=dict(visible=True, range=bbox[1]),
        ),
    )
    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5),height=700,showlegend=True)
    return fig

def renderGeo(mesh, saved_loc, image_size = 1024, camera_dist = 500, elevation = 30, azim_angle = -45, ptLightLoc = [[-100, 100, -100]], show = True):

    # Extract vertices and faces from mesh
    v = torch.tensor(mesh.vertices.astype(float), dtype=torch.float32).unsqueeze(0).to(device)
    f = torch.tensor(mesh.faces.astype(float), dtype=torch.float32).unsqueeze(0).to(device)

    # Create white texture
    verts_rgb = torch.ones_like(torch.tensor(mesh.vertices.astype(float), dtype=torch.float32))[None] # (1, V, 3)
    t = Textures(verts_rgb=verts_rgb.to(device)).to(device)

    # Generate Pytorch3d mesh 
    mesh_py3d = Meshes(verts=v,faces=f,textures=t).to(device)

    # Initialize the camera with camera distance, elevation, and azimuth angle
    R, T = look_at_view_transform(dist = camera_dist, elev = 
                                    elevation, azim = azim_angle) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Here we set the output image to be of size 256 x 256 based on config.json 
    # raster_settings = RasterizationSettings(
    #       image_size = image_size, 
    #       blur_radius = 0.0, 
    #       faces_per_pixel = 1, 
    #       max_faces_per_bin = 3
    # )
    raster_settings = RasterizationSettings(
      image_size = image_size, 
      blur_radius = 0.0, 
      faces_per_pixel = 1, 
    )

    # Initialize rasterizer by using a MeshRasterizer class
    rasterizer = MeshRasterizer(
              cameras=cameras, 
              raster_settings=raster_settings
    )

    # The textured phong shader interpolates the texture uv coordinates for 
    # each vertex, and samples from a texture image.
    shader = HardPhongShader(device = device, cameras = cameras)

    # Change specular color to green and change material shininess 
    materials = Materials(
          device=device,
          specular_color=[[0.0, 0, 0.0]],
          shininess=10
    )

    lights = PointLights(device=device, location=ptLightLoc)

    # Create a mesh renderer by composing a rasterizer and a shader
    renderer = MeshRenderer(rasterizer, shader)

    # Render Meshes object
    image = renderer(mesh_py3d, lights=lights, materials=materials)
    # Plot rendered image
    plt.figure(figsize=(5, 5))
    plt.imshow(image[0, 0:768, 128:896].cpu().numpy())
    plt.grid("off")
    plt.axis("off")
    plt.savefig(saved_loc, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    if(show is False):
        plt.close()

def plotImageMatrix(im_files_ar, im_titles, fig_title = '', text_size = 12, title_size = 20, col_n = None, save_dir = None, show_titles = True, plot_size = 18):

    if(col_n is None):
        col_n = len(im_files_ar)
    row_n = int(len(im_files_ar)/col_n)

    fig = plt.figure(figsize=(plot_size*col_n/row_n+30,plot_size))
    grid =  ImageGrid(fig, 111, 
                        nrows_ncols=(row_n, col_n),
                        axes_pad=0.1,
                        )

    for ax, im, t in zip(grid, im_files_ar, im_titles):
        im_r = plt.imread(im)
        ax.imshow(im_r)
        if(show_titles):
            ax.set_title(t,fontsize=text_size)
        ax.grid("off")
        ax.axis("off")

    if(show_titles):
        plt.suptitle(fig_title, size=title_size)

    if(save_dir is not None):
        plt.savefig(save_dir, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)

    plt.show()
    

# Function to plot two meshes side by side
def visualize_meshes_side_by_side(mesh_1_vertices: np.ndarray, mesh_1_faces: np.ndarray, mesh_2_vertices: np.ndarray, mesh_2_faces: np.ndarray, mesh_1_color='pink', mesh_2_color='pink', mesh_1_opacity=1, mesh_2_opacity=1, flatshading: bool = False, height=700):

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"},{"type": "scene"}]])

    x, y, z = mesh_1_vertices.T
    i, j, k = mesh_1_faces.T
    mesh_1 = go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                color=mesh_1_color,
                opacity=mesh_1_opacity
            )
    
    x, y, z = mesh_2_vertices.T
    i, j, k = mesh_2_faces.T
    mesh_2 = go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                color=mesh_2_color,
                opacity=mesh_2_opacity
            )
    
    
    fig.add_trace(mesh_1, row=1, col=1)
    fig.add_trace(mesh_2, row=1, col=2)

    # styling
    fig.update_traces(
        flatshading=flatshading, lighting=dict(specular=1.0), selector=dict(type="mesh3d")
    )
    fig.update_scenes(xaxis_showbackground=False,
                      yaxis_showbackground=False,
                      zaxis_showbackground=False,
                      xaxis=dict(visible=False, range=[0,1]),
                      yaxis=dict(visible=False, range=[0,1]),
                      zaxis=dict(visible=False, range=[0,1]),
                      )

    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5),height=height)

    return fig

# Function to plot two meshes one overlayed on the other
def visualize_meshes_overlayed(mesh_1_vertices: np.ndarray, mesh_1_faces: np.ndarray, mesh_2_vertices: np.ndarray, mesh_2_faces: np.ndarray, mesh_1_color='grey', mesh_2_color='pink', mesh_1_opacity=0.8, mesh_2_opacity=0.1, flatshading: bool = False, height=700):

    x, y, z = mesh_1_vertices.T
    i, j, k = mesh_1_faces.T
    mesh_1 = go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                color = mesh_1_color,
                opacity = mesh_1_opacity
            )
    
    x, y, z = mesh_2_vertices.T
    i, j, k = mesh_2_faces.T
    mesh_2 = go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                color = mesh_2_color,
                opacity = mesh_2_opacity
            )
    
    mesh_list = [mesh_1, mesh_2]
    fig = go.Figure(mesh_list)

    # styling
    fig.update_traces(
        flatshading=flatshading, lighting=dict(specular=1.0), selector=dict(type="mesh3d")
    )
    fig.update_scenes(xaxis_showbackground=False,
                      yaxis_showbackground=False,
                      zaxis_showbackground=False,
                      xaxis=dict(visible=False, range=[0,1]),
                      yaxis=dict(visible=False, range=[0,1]),
                      zaxis=dict(visible=False, range=[0,1]),
                      )

    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5),height=height)

    return fig

def visualize_mesh_gradient(vertices: np.ndarray, faces: np.ndarray, metric, flatshading: bool = False, height=800):
    x, y, z = vertices.T
    i, j, k = faces.T
    fig = go.Figure(
        [
            go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                colorbar_title='z',
                colorscale='Plasma',
                intensity = metric,
                #intensitymode='cell',
                showscale=True, 
                cmin = 0.0,
                cmax = 1.0
            ),
        ]
    )

    # styling
    fig.update_traces(
        flatshading=flatshading, lighting=dict(specular=1.0), selector=dict(type="mesh3d")
    )
    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5))
    fig.update_layout(scene_aspectmode="data")
    fig.update_layout(
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        scene_camera=dict(eye=dict(x=0.1, y=3, z=0.1)),
        height=height
    )

    return fig

# Function to plot sampled points
def plotPointswithValues(points, vals, bbox = None):
    scatter_plot = go.Scatter3d(x = points[:,0], 
                                y = points[:,2], 
                                z = points[:,1], 
                                mode='markers',
                                marker=dict(
                                    color=vals,
                                    colorscale='Plasma',
                                    size=3,
                                    opacity=1,
                                    cmin = np.min(vals),
                                    cmax = np.max(vals),
                                    showscale = True
                                ),
                                )
    fig = go.Figure(scatter_plot)
    fig.update_scenes(xaxis_title_text='X',  
                      yaxis_title_text='Y',  
                      zaxis_title_text='Z',
                      xaxis_showbackground=False,
                      yaxis_showbackground=False,
                      zaxis_showbackground=False,
                      )
    fig.update_layout(
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=True, range=bbox[0]) if(bbox is not None) else dict(visible=True),
            yaxis=dict(visible=True, range=bbox[2]) if(bbox is not None) else dict(visible=True),
            zaxis=dict(visible=True, range=bbox[1]) if(bbox is not None) else dict(visible=True),
        ),
    )
    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5),height=700,showlegend=True)
    return fig

# Function to plot sampled points and mesh
def plotPointsMesh(mesh, points, density, bbox, flatshading: bool = False, visible_axes=True):

    x, y, z = mesh.vertices.T
    i, j, k = mesh.faces.T
    maxDensity = np.max(density)

    mesh_plot = go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                color='white',
                opacity=0.05,
            )
    
    scatter_plot = go.Scatter3d(x = points[:,0], 
                                y = points[:,2], 
                                z = points[:,1], 
                                mode='markers',
                                marker=dict(
                                    color=density/maxDensity,
                                    colorscale='Plasma',
                                    size=2,
                                    opacity=0.8,
                                    cmin = 0,
                                    cmax = 1,
                                    showscale = True
                                ),
                                )

    fig = go.Figure([mesh_plot, scatter_plot])

    # styling
    fig.update_traces(
        flatshading=flatshading, lighting=dict(specular=1.0), selector=dict(type="mesh3d")
    )
    fig.update_scenes(xaxis_title_text='X',  
                      yaxis_title_text='Y',  
                      zaxis_title_text='Z',
                      xaxis_showbackground=False,
                      yaxis_showbackground=False,
                      zaxis_showbackground=False,
                      )
    fig.update_layout(
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=visible_axes, range=bbox[0]),
            yaxis=dict(visible=visible_axes, range=bbox[2]),
            zaxis=dict(visible=visible_axes, range=bbox[1]),
        ),
    )
    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5),height=700,showlegend=True)
    return fig

# From https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = np.array(points)[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

# Plot performance distribution among dataset
def plotPerfDist_perExp(perf_df, color = cm.nipy_spectral(float(9)/10)):

    row_n = 1
    col_n = len(perf_df.columns)
    bin_n = 20
    alpha_v = 0.8
    colsC = perf_df.columns

    fig, ax = plt.subplots(row_n, col_n,  figsize=(20,2))
    for i in range(row_n*col_n):
        ax[i].hist(perf_df[colsC[i]].astype(float), bins = bin_n, alpha = alpha_v, color = color)
        ax[i].set_title(colsC[i],fontsize=10)
        ax[i].tick_params(labelsize=8)
        ax[i].tick_params(labelsize=8)
    plt.show()
    
# Plot performance distribution among dataset
def plotPerfDist_TrainTest(perf_df_train, perf_df_test, color = cm.nipy_spectral(float(9)/10)):

    row_n = 2
    col_n = len(perf_df_train.columns)
    bin_n = 20
    alpha_v = 0.8
    colsC = perf_df_train.columns
    perf_comb = [perf_df_train, perf_df_test]

    fig, ax = plt.subplots(row_n, col_n,  figsize=(20,6))
    for i in range(row_n):
        for j in range(col_n):
            ax[i][j].hist(perf_comb[i][colsC[j]].astype(float), bins = bin_n, alpha = alpha_v, color = color)
            ax[i][j].set_title(colsC[j],fontsize=10)
            ax[i][j].tick_params(labelsize=8)
            ax[i][j].tick_params(labelsize=8)
    plt.show()

# Plot performance distribution among dataset
def plotPerfDist(perf_df, set_idx, title = None, colors = None, color_ind = None, x_lim = None, y_lim = None, bin_n = 20, exclude_outliers=False, file_path=None):

    col_n = 1
    alpha_v = 0.8
    colsC = perf_df.columns
    #print(colsC)
    width = 18
    vals = perf_df.values.astype(float)
    vals_sep = vals
    if(exclude_outliers):
        vals = [v[~is_outlier(v)] for v in vals.T]
        vals = [item for row in vals for item in row]
    bins=np.histogram(vals, bins=bin_n)[1]
    y_max = int(np.max([np.histogram(v, bins=bins)[0] for v in np.array(vals_sep, dtype=object).T])*1.1)
    x_max = np.ceil(np.max(vals)*100000)*1.1/100000
    print('x_max',x_max)
    print('y_max',y_max)
    un_sets = len(np.unique(set_idx))

    for j in range(un_sets):
        row_n = np.sum([int(n == j) for n in set_idx])
        fig, ax = plt.subplots(row_n, col_n,  figsize=(width,width*row_n/5))
        cur_ids = np.array([id_n for id_n in range(len(set_idx)) if(int(set_idx[id_n] == j))])
        #print(cur_ids)
        for i in range(row_n):
            vals = perf_df[colsC[cur_ids][i]].astype(float)
            #print(vals)
            if(exclude_outliers):
                vals = vals[~is_outlier(vals)]
            if((colors is not None) and (color_ind is not None)):
                ax[i].hist(vals, bins = bins, alpha = alpha_v, color = colors[color_ind[cur_ids][i]])
            else:
                ax[i].hist(vals, bins = bins, alpha = alpha_v)
            ax[i].set_title(colsC[cur_ids][i],fontsize=10)
            #ax[i].tick_params(labelsize=8)
            if(y_lim is not None):
                ax[i].set_ylim(y_lim)
            else:
                ax[i].set_ylim([0,y_max])
            if(x_lim is not None):
                ax[i].set_xlim(x_lim)
            else:
                ax[i].set_xlim([0,x_max])
    
        if(title is not None):
            plt.suptitle(title + '_'+str(j), size=10)
        else:
            plt.tight_layout()
        if(file_path is not None):
            plt.savefig(file_path.replace('.png','_'+str(j)+'.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close() 
        clear_output(wait = True)
        plt.pause(0.1)
        
    return x_max, y_max


def plotBoxPlot(perf_df, x_tick_labels = None, title = None, ylim = None, colors = None, color_ind = None, file_path=None):
    sns.set(
            style="ticks",                   # The 'ticks' style
            rc={"figure.figsize": (30, 6),      # width = 6, height = 9
                "figure.facecolor": "white",  # Figure colour
                "axes.facecolor": "white"})  # Axes colour
    my_pal = np.array(colors)[color_ind]
    if((colors is not None) and (color_ind is not None)):
        b = sns.boxplot(data = perf_df.values,    
                                width = 0.4,        # The width of the boxes
                                palette = list(my_pal),
                                linewidth = 1,      # Thickness of the box lines
                                showfliers = False)  # Sop showing the fliers
    else:
        b = sns.boxplot(data = perf_df.values,    
                                width = 0.4,        # The width of the boxes
                                linewidth = 1,      # Thickness of the box lines
                                showfliers = False)  # Sop showing the fliers
    if(x_tick_labels is not None):
        b.set_xticks(np.arange(len(x_tick_labels)))
        b.set_xticklabels(x_tick_labels)
    if(y_lim is not None):
        b.set(ylim=ylim)
    if(title is not None):
        b.set_title(title, fontsize = 11)
    b.get_figure()
    if(file_path is not None):
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() 
    clear_output(wait = True)
    plt.pause(0.5)
    
    
    
def plotMtplBars(perf_arr, x_lab = None, y_lab = None, title = None, color_list = cm.nipy_spectral(np.linspace(0,1,10)), file_path=None):

    fig = plt.figure(figsize=(20,4))
    X = x_lab
    Y = perf_arr
    diff = np.linspace(0,1,len(Y))-np.linspace(0,1,len(Y))[int(len(Y)/2)]
    sp_r = np.arange(len(X))*0.25
    c = color_list[0:len(X)]
    X_axis = np.arange(len(X)) 
    for i in range(len(Y)):
        plt.bar(X_axis + diff[i] + sp_r, Y[i], width=(diff[1]-diff[0])*0.8, label = y_lab[i], color = c)
    plt.title(title)
    plt.xticks(X_axis + sp_r, X) 
    if(file_path is not None):
        plt.savefig(file_path, dpi=300)
    plt.show()
    plt.close() 
    clear_output(wait = True)
    plt.pause(0.5)
    
    # Function to plot multiple meshes side-by-side
def plotMultipleMeshes(meshes, mesh_names, flatshading: bool = False, height=800, color="pink", opacity=0.8):
    num = len(meshes)
    fig = make_subplots(rows=1, cols=num, subplot_titles=mesh_names, specs=[list(np.repeat({"type": "scene"}, num))])

    mesh_plots = []
    for p in range(num):
        x, y, z = np.array(meshes[p].vertices).T
        i, j, k = np.array(meshes[p].faces).T
        mesh_plot = go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                color=color,
                opacity=opacity
            )
        mesh_plots.append(mesh_plot)
        fig.add_trace(mesh_plot, row=1, col=(p+1))

    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=4, y=4, z=4)
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
        height=height
    )
    fig.update_annotations(font_size=8)

    return fig


# Function to plot multiple meshes side-by-side
def plotMultipleMeshesonGrid(meshes, mesh_names, flatshading: bool = False, height=800, color="pink", opacity=0.8):
    fig = make_subplots(rows=meshes.shape[0], cols=meshes.shape[1], subplot_titles=mesh_names.flatten(), specs=[list(l) for l in list(np.repeat({"type": "scene"}, meshes.shape[0]*meshes.shape[1]).reshape((meshes.shape[0],meshes.shape[1])))])

    mesh_plots = []
    for p in range(meshes.shape[0]):
        for r in range(meshes.shape[1]):
            x, y, z = np.array(meshes[p][r].vertices).T
            i, j, k = np.array(meshes[p][r].faces).T
            mesh_plot = go.Mesh3d(
                    x=x,
                    y=z,
                    z=y,
                    i=i,
                    j=j,
                    k=k,
                    color=color,
                    opacity=opacity
                )
            mesh_plots.append(mesh_plot)
            fig.add_trace(mesh_plot, row=(p+1), col=(r+1))

    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=4, y=4, z=4)
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
        height=height
    )
    fig.update_annotations(font_size=8)

    return fig

# Function to plot sampled points and mesh
def plotPointsMesh(mesh, points, values, bbox, cmin_v = None, cmax_v = None, mesh_opacity = 0.2, pts_opacity=0.8, points_size=2,flatshading: bool = False, colorscale=None, visible_axes=False, eye_x=2, eye_y=2, eye_z=1.5, center_x=0, center_y=0, center_z=0, mesh_above=True):

    x, y, z = mesh.vertices.T
    i, j, k = mesh.faces.T
    maxValue = np.max(values)
    cmin_v = cmin_v if(cmin_v is not None) else 0
    cmax_v = cmax_v if(cmax_v is not None) else 1
    
    mesh_plot = go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                color='white',
                opacity=mesh_opacity,
            )
    
    colorscale = colorscale if(isinstance(colorscale, str)) else ['rgb' + str(colorscale[i]) for i in range(len(colorscale))] if(colorscale is not None) else 'Plasma'
    scatter_plot = go.Scatter3d(x = points[:,0], 
                                y = points[:,2], 
                                z = points[:,1], 
                                mode='markers',
                                marker=dict(
                                    color=values,
                                    colorscale=colorscale,#'Plasma',
                                    size=points_size,
                                    opacity=pts_opacity,
                                    cmin = cmin_v,
                                    cmax = cmax_v,
                                    showscale = True
                                ),
                                )

    fig = go.Figure([scatter_plot, mesh_plot]) if(mesh_above) else go.Figure([mesh_plot, scatter_plot])

    # styling
    fig.update_traces(
        flatshading=flatshading, lighting=dict(specular=1.0), selector=dict(type="mesh3d")
    )
    fig.update_scenes(xaxis_title_text='X',  
                      yaxis_title_text='Y',  
                      zaxis_title_text='Z',
                      xaxis_showbackground=False,
                      yaxis_showbackground=False,
                      zaxis_showbackground=False,
                      )
    fig.update_layout(
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=visible_axes, range=bbox[0]),
            yaxis=dict(visible=visible_axes, range=bbox[2]),
            zaxis=dict(visible=visible_axes, range=bbox[1]),
        ),
        scene_camera=dict(
            eye=dict(x=eye_x, y=eye_y, z=eye_z),
            center=dict(x=center_x, y=center_y, z=center_z),
        ),
    )
    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5),height=700,showlegend=True)
    fig.update_layout(scene_aspectmode="cube")

    return fig

# Function to plot sampled points and mesh
def plotVectorsMesh(mesh, points, values, bbox, cmin_v = None, cmax_v = None, mesh_opacity = 0.2, pts_opacity=0.8, points_size=10, flatshading: bool = False, colorscale=None, visible_axes=False, eye_x=2, eye_y=2, eye_z=1.5, center_x=0, center_y=0, center_z=0, mesh_above=True):

    x, y, z = mesh.vertices.T
    i, j, k = mesh.faces.T
    maxValue = np.max(values)
    cmin_v = cmin_v if(cmin_v is not None) else 0
    cmax_v = cmax_v if(cmax_v is not None) else 1
    
    mesh_plot = go.Mesh3d(
                x=x,
                y=z,
                z=y,
                i=i,
                j=j,
                k=k,
                color='white',
                opacity=mesh_opacity,
            )
    
    colorscale = colorscale if(isinstance(colorscale, str)) else ['rgb' + str(colorscale[i]) for i in range(len(colorscale))] if(colorscale is not None) else 'Plasma'
    vector_plot = go.Cone(x = points[:,0], 
                          y = points[:,2], 
                          z = points[:,1],
                          u = values[:,1]*values[:,0].flatten(),
                          v = values[:,3]*values[:,0].flatten(),
                          w = values[:,2]*values[:,0].flatten(),
                          sizemode="absolute",
                          sizeref=points_size,
                          colorscale=colorscale,
                          cmin=cmin_v,
                          cmax=cmax_v,
                          opacity=pts_opacity
                                )

    fig = go.Figure([mesh_plot, vector_plot])  if(mesh_above) else go.Figure([mesh_plot, vector_plot])

    # styling
    fig.update_traces(
        flatshading=flatshading, lighting=dict(specular=1.0), selector=dict(type="mesh3d")
    )
    fig.update_scenes(xaxis_title_text='X',  
                      yaxis_title_text='Y',  
                      zaxis_title_text='Z',
                      xaxis_showbackground=False,
                      yaxis_showbackground=False,
                      zaxis_showbackground=False,
                      )
    fig.update_layout(
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=visible_axes, range=bbox[0]),
            yaxis=dict(visible=visible_axes, range=bbox[2]),
            zaxis=dict(visible=visible_axes, range=bbox[1]),
        ),
        scene_camera=dict(
            eye=dict(x=eye_x, y=eye_y, z=eye_z),
            center=dict(x=center_x, y=center_y, z=center_z)
        ),
    )
    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5),height=700,showlegend=True)
    fig.update_layout(scene_aspectmode="cube")

    return fig


# Function to plot sampled points
def plotPointswithValuesDirection(points, vals, minV = None, maxV = None, bbox = None, mesh_opacity = 0.2, pts_opacity=0.8, flatshading: bool = False, colorscale=None, visible_axes=False, eye_x=2, eye_y=2, eye_z=1.5):

    minV = np.min(vals[:,0].flatten()) if (minV == None) else minV
    maxV = np.max(vals[:,0].flatten()) if (maxV == None) else maxV

    pts_toPlot = points
    vals_toPlot = vals
    colorscale = ['rgb' + str(colorscale[i]) for i in range(len(colorscale))] if(colorscale is not None) else 'Plasma'
        
    vector_plot = go.Cone(x = pts_toPlot[:,0], 
                          y = pts_toPlot[:,2], 
                          z = pts_toPlot[:,1],
                          u = vals[:,1]*vals[:,0].flatten(),
                          v = vals[:,3]*vals[:,0].flatten(),
                          w = vals[:,2]*vals[:,0].flatten(),
                          sizemode="absolute",
                          sizeref=10,
                          colorscale=colorscale,
                          cmin=minV,
                          cmax=maxV,
                                )
    fig = go.Figure(vector_plot)
    fig.update_scenes(xaxis_title_text='X',  
                      yaxis_title_text='Y',  
                      zaxis_title_text='Z',
                      xaxis_showbackground=False,
                      yaxis_showbackground=False,
                      zaxis_showbackground=False,
                      )
    fig.update_layout(
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=visible_axes, range=bbox[0]) if(bbox is not None) else dict(visible=True),
            yaxis=dict(visible=visible_axes, range=bbox[2]) if(bbox is not None) else dict(visible=True),
            zaxis=dict(visible=visible_axes, range=bbox[1]) if(bbox is not None) else dict(visible=True),
        ),
        scene_camera=dict(
            eye=dict(x=eye_x, y=eye_y, z=eye_z),
           # center=dict(x=0, y=0, z=0.7)
        ),
    )
    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5),height=700,showlegend=True)
    fig.update_layout(scene_aspectmode="cube")
    return fig


# Function to plot sampled points
def plotPointswithValues(points, vals, minV = None, maxV = None, bbox = None):
    minV = np.min(vals) if (minV == None) else minV
    maxV = np.max(vals) if (maxV == None) else maxV

    pts_toPlot = points
    vals_toPlot = vals
        
    scatter_plot = go.Scatter3d(x = pts_toPlot[:,0], 
                                y = pts_toPlot[:,2], 
                                z = pts_toPlot[:,1], 
                                mode='markers',
                                marker=dict(
                                    color=vals_toPlot,
                                    colorscale='Plasma',
                                    size=3,
                                    opacity=0.8,
                                    cmin = minV,
                                    cmax = maxV,
                                    showscale = True
                                ),
                                )
    fig = go.Figure(scatter_plot)
    fig.update_scenes(xaxis_title_text='X',  
                      yaxis_title_text='Y',  
                      zaxis_title_text='Z',
                      xaxis_showbackground=False,
                      yaxis_showbackground=False,
                      zaxis_showbackground=False,
                      )
    fig.update_layout(
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=True, range=bbox[0]) if(bbox is not None) else dict(visible=True),
            yaxis=dict(visible=True, range=bbox[2]) if(bbox is not None) else dict(visible=True),
            zaxis=dict(visible=True, range=bbox[1]) if(bbox is not None) else dict(visible=True),
        ),
    )
    fig.update_layout(margin=dict(r=5, l=5, b=5, t=5),height=700,showlegend=True)
    fig.update_layout(scene_aspectmode="cube")
    return fig

def plotHistogramMtx(values, bins_n, mtx_shape, v_names = None, colormap = remap_color_maps['colors_set_8'], c_ind = None, c_type='byrow', x_max_t = 'byind', y_max_t = 'byind', bins_t = 'byind', flip_matrix=False, showTitles=True, title_fontsize=8, showRightAxes=False, showLeftAxes=True, showTopAxes=False, removeLeftRepAxes = False, removeBottRepAxes = False, xticks_split = 8, yticks_split = 6, xticks_round = 4, returnFig = False, ticksize=6, width=6, height=2.5):

    values = np.array(values, dtype=object).reshape(mtx_shape)
    values = values if(not flip_matrix) else values.T
    
    if(v_names is not None):
        v_names = np.array(v_names).reshape(mtx_shape)
        v_names = v_names if(not flip_matrix) else v_names.T
    
    rows_n = values.shape[0]
    cols_n = values.shape[1]
    
    plt.rcParams['axes.spines.right'] = showRightAxes
    plt.rcParams['axes.spines.top'] = showTopAxes
    plt.rcParams['axes.spines.left'] = showLeftAxes
    
    fig, ax = plt.subplots(rows_n,cols_n, figsize=(width*cols_n,height*rows_n))
    
    y_max = np.array([np.max(np.histogram(v, bins=bins_n)[0]) for v in values.reshape(np.prod(mtx_shape))]).reshape(mtx_shape)
    x_max = np.array([np.max(np.histogram(v, bins=bins_n)[1]) for v in values.reshape(np.prod(mtx_shape))]).reshape(mtx_shape)
    print('x_max',x_max)
    print('y_max',y_max)
    bins = np.array([np.histogram(v, bins=bins_n)[1] for v in values.reshape(np.prod(mtx_shape))]).reshape(mtx_shape[0],mtx_shape[1],-1)
    x_max_v = {'bytot':[np.max(x_max)],'byrow':np.max(x_max,axis=1),'bycol':np.max(x_max,axis=0), 'byind':x_max.flatten()}
    y_max_v = {'bytot':[np.max(y_max)],'byrow':np.max(y_max,axis=1),'bycol':np.max(y_max,axis=0), 'byind':y_max.flatten()}
    bins_v = {'bytot':[bins.reshape(mtx_shape[0]*mtx_shape[1],-1).mean(1)],'byrow':bins.mean(axis=1),'bycol':bins.mean(axis=0), 'byind':bins.reshape(mtx_shape[0]*mtx_shape[1],-1)}
    
    colors = np.array(colormap)[c_ind] if(c_ind is not None) else colormap

    for i in range(rows_n):
        for j in range(cols_n):
            ax_c = ax[i] if((mtx_shape[0]==1) or (mtx_shape[1]==1)) else ax[i][j]
            id_set = {'bytot':0,'byrow':i,'bycol':j,'byind':j+(i*cols_n)}
            c_ind = j+(i*cols_n) if(c_type=='byind') else i if(c_type=='byrow') else j
            ax_c.hist(values[i][j], bins = bins_v[bins_t][id_set[bins_t]], color=colormap[c_ind])
            #ax[i][j].hist(values[i][j], bins = bins_n) 
            ax_c.set_xlim([0,x_max_v[x_max_t][id_set[x_max_t]]])
            ax_c.set_xticks(np.linspace(0,x_max_v[x_max_t][id_set[x_max_t]],xticks_split),labels=[format(n, '.'+str(xticks_round)) for n in np.linspace(0,x_max_v[x_max_t][id_set[x_max_t]],xticks_split)])
            ax_c.set_yticks(np.linspace(0,y_max_v[y_max_t][id_set[y_max_t]],yticks_split),labels=np.linspace(0,y_max_v[y_max_t][id_set[y_max_t]],yticks_split).astype(int))
            ax_c.set_ylim([0,y_max_v[y_max_t][id_set[y_max_t]]])
            ax_c.tick_params(labelsize=ticksize)
            if(((j>0) and removeLeftRepAxes) or (not showLeftAxes)):
                ax_c.get_yaxis().set_visible(False)
                ax_c.spines[['left']].set_visible(False)
            if((i!=rows_n-1) and removeBottRepAxes):
                ax_c.get_xaxis().set_visible(False)
                ax_c.spines[['bottom']].set_visible(False)
            if((v_names is not None) and showTitles):
                ax_c.set_title(v_names[i][j], fontsize = title_fontsize)
    plt.subplots_adjust(hspace=0.6, wspace=0.2)  
    plt.show()
    
    
def getGeoAllPerfPar(xyz_sdf_perf_sel, i, perf_metric, p_sets=['srf_20','XYgrid_256_30_15'], p_set_names=['srf','grd'], print_stats=False):
    p_list_n = ['xyz', 'sdf_gt', 'perf_gt', 'pred_p', 'perf_dif', 'perf_difP','perf_gt_vec', 'perf_gt_vec_norm', 'pred_p_vec', 'pred_p_vec_norm']
    params = {}
    for j in range(len(p_sets)):
        model_input, sdf, perf = xyz_sdf_perf_sel.__getitembySet__(i, p_sets[j], fil_bBox = True, include_sdf = True, formatforModel = True)  
        if(sdf is not None):
            multip = 20 if(perf_metric == 'U') else 1
            if(print_stats):
                print('Point set: ' + p_sets[j])
                print('Sample ' + str(model_input['idx'][0]) + ', name: ' + model_input['item_n'][0] + ', rot: ' + str(model_input['rot'][0]))
            model_output = model.to('cpu')(model_input)
            xyz = model_input['xyz'][0].cpu().numpy()
            sdf_gt = sdf.cpu().numpy().flatten()
            perf_gt = perf[:,0].cpu().numpy().flatten() * multip
            perf_gt_vec = perf.cpu().numpy()
            perf_gt_vec[:,0] = perf_gt_vec[:,0]* multip
            perf_gt_vec_norm = perf[:,1:].cpu().numpy()
            pred_p = model_output[0][0][:,0].detach().numpy().flatten() * multip
            pred_p_vec = model_output[0][0].detach().numpy()
            pred_p_vec[:,0] = pred_p_vec[:,0]* multip
            pred_p_vec_norm = model_output[0][0][:,1:].detach().numpy()
            perf_dif = pred_p-perf_gt
            p_list = [xyz, sdf_gt, perf_gt, pred_p, perf_dif, perf_dif/(perf_gt+0.00000000001), perf_gt_vec, perf_gt_vec_norm, pred_p_vec, pred_p_vec_norm]
            for k in range(len(p_list_n)):
                params[p_set_names[j] + '_' + p_list_n[k]] = p_list[k]
            if(print_stats):
                print('Shape: '+ str(xyz.shape))
                print('GT - SDF value bounds: ' + str(np.min(sdf_gt)) + ' and ' + str(np.max(sdf_gt)))
                print('GT - PERF value bounds: ' + str(np.min(perf_gt)) + ' and ' + str(np.max(perf_gt)))
                print('PRED - PERF value bounds: ' + str(np.min(pred_p)) + ' and ' + str(np.max(pred_p)))
                print('DIFF - PERF value bounds: ' + str(np.min(perf_dif)) + ' and ' + str(np.max(perf_dif)))
        else:
            print('The element does not exist in the dataset.')
    
    params['item_n'] = model_input['item_n'][0]
    params['geo_n'] = '_'.join(model_input['item_n'][0].split('_')[0:2])
    return params