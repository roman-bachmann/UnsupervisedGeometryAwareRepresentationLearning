import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import os
import torch
import numpy as np
import scipy.linalg as la
from sklearn.manifold import TSNE


def random_color():
    return tuple(np.random.choice(range(256), size=3) / 255)

def plot_scatter_3d(d, idxs=None, complement=False, save_path=None, save_iter=0):
    from mayavi import mlab
    mlab.options.offscreen = True
    figure = mlab.figure()
    if idxs is None:
        idxs = np.arange(d.shape[1])
        xmin,xmax,ymin,ymax,zmin,zmax = d[:,:,0].min(), d[:,:,0].max(), d[:,:,1].min(), d[:,:,1].max(), d[:,:,2].min(), d[:,:,2].max()
        d = d.reshape(-1,3)
        pts = mlab.points3d(d[:,0], d[:,1], d[:,2], figure=figure, scale_mode='none', scale_factor=0.03)
    else:
        idxs = np.array(idxs)
        if complement:
            idxs = np.setdiff1d(np.arange(d.shape[1]), idxs)
        for idx in idxs:
            pts = mlab.points3d(d[:,idx,0], d[:,idx,1], d[:,idx,2], figure=figure, scale_mode='none', scale_factor=0.03, color=random_color())
        xmin,xmax,ymin,ymax,zmin,zmax = d[:,idxs,0].min(), d[:,idxs,0].max(), d[:,idxs,1].min(), d[:,idxs,1].max(), d[:,idxs,2].min(), d[:,idxs,2].max()
    mlab.axes(figure=figure, xlabel='x', ylabel='y', zlabel='z', ranges=(xmin,xmax,ymin,ymax,zmin,zmax))
    if save_path is not None:
        mlab.savefig(os.path.join(save_path, 'scatter3D_outl{}_i{}.pdf'.format(idxs!=None, save_iter)))
    else:
        mlab.show()
    mlab.close()

def reparameterize(mu, logvar):
    std = np.exp(0.5*logvar)
    eps = np.random.randn(*std.shape) # Sample from N(0,1)
    return mu + eps*std

def plot_samples(mus_3d, logvars_3d, img_idx, point_idxs=None, num_samples=10):
    from mayavi import mlab
    mlab.options.offscreen = True
    if point_idxs is not None:
        mu = mus_3d[img_idx, point_idxs]
        logvar = logvars_3d[img_idx, point_idxs]
    else:
        mu = mus_3d[img_idx]
        logvar = logvars_3d[img_idx]
    figure = mlab.figure()
    for i in range(num_samples):
        sample = reparameterize(mu, logvar)
        pts = mlab.points3d(sample[:,0], sample[:,1], sample[:,2], scale_mode='none', scale_factor=0.03, color=(0,0,0.5))
    pts = mlab.points3d(mu[:,0], mu[:,1], mu[:,2], scale_mode='none', scale_factor=0.07, color=(1,0,0))
    mlab.axes()
    mlab.show()

def plot_mu_std_hists(mus_fg, logvars_fg, mus_3d, logvars_3d, save_path=None, save_iter=0):
    fig, ax = plt.subplots(2, 2)

    if mus_fg is not None:
        ax[0,0].set_title('$\mu_{FG}$ histogram')
        ax[0,0].hist(mus_fg.flatten(), bins=1000)

    if logvars_fg is not None:
        ax[0,1].set_title('$\sigma_{FG}^2$ histogram')
        std_fg = np.exp(0.5*logvars_fg.flatten())
        ax[0,1].hist(std_fg, bins=1000)

    if mus_3d is not None:
        ax[1,0].set_title('$\mu_{3D}$ histogram')
        ax[1,0].hist(mus_3d.flatten(), bins=1000)

    if logvars_3d is not None:
        ax[1,1].set_title('$\sigma_{3D}^2$ histogram')
        std_3d = np.exp(0.5*logvars_3d.flatten())
        ax[1,1].hist(std_3d, bins=1000)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'hists_i{}.pdf'.format(save_iter)))
    else:
        plt.show()
    plt.close()

def analyze_mu_std(mus_fg, logvars_fg, mus_3d, logvars_3d, save_path=None, save_iter=0):
    if mus_fg is not None:
        plt.figure(figsize=(20,20))
        plt.title('$\mu_{FG}$ mean per feature')
        plt.stem(np.arange(mus_fg.shape[1]), mus_fg.mean(axis=0), bottom=0)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'mu_fg_mean_i{}.pdf'.format(save_iter)))
        else:
            plt.show()
        plt.close()

    if logvars_fg is not None:
        plt.figure(figsize=(20,20))
        plt.title('$\sigma_{FG}^2$ mean per feature')
        std_fg = np.exp(0.5*logvars_fg)
        plt.stem(np.arange(std_fg.shape[1]), std_fg.mean(axis=0), bottom=1)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'std_fg_mean_i{}.pdf'.format(save_iter)))
        else:
            plt.show()
        plt.close()

    if mus_3d is not None:
        plt.figure(figsize=(20,20))
        plt.title('$\mu_{3D}$ mean per feature')
        mus_3d_mean = np.linalg.norm(mus_3d, axis=2).mean(axis=0)
        plt.stem(np.arange(mus_3d.shape[1]), mus_3d_mean, bottom=0)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'mu_3d_norm_mean_i{}.pdf'.format(save_iter)))
        else:
            plt.show()
        plt.close()

        plt.figure(figsize=(20,20))
        plt.title('$\mu_{3D}$ mean per feature')
        mus_3d_mean = mus_3d.reshape(-1,mus_3d.shape[1]*mus_3d.shape[2]).mean(axis=0)
        plt.stem(np.arange(mus_3d.shape[1]*mus_3d.shape[2]), mus_3d_mean, bottom=0)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'mu_3d_mean_i{}.pdf'.format(save_iter)))
        else:
            plt.show()
        plt.close()

    if logvars_3d is not None:
        plt.figure(figsize=(20,20))
        plt.title('$\sigma_{3D}^2$ mean per feature')
        std_3d = np.exp(0.5*logvars_3d)
        std_3d_mean = np.linalg.norm(std_3d, axis=2).mean(axis=0)
        plt.stem(np.arange(std_3d.shape[1]), std_3d_mean, bottom=1)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'std_3d_norm_mean_i{}.pdf'.format(save_iter)))
        else:
            plt.show()
        plt.close()

        plt.figure(figsize=(20,20))
        plt.title('$\sigma_{3D}^2$ mean per feature, v2')
        std_3d = np.exp(0.5*logvars_3d)
        std_3d_mean = std_3d.reshape(-1,std_3d.shape[1]*std_3d.shape[2]).mean(axis=0)
        plt.stem(np.arange(std_3d.shape[1]*std_3d.shape[2]), std_3d_mean, bottom=1)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'std_3d_mean_i{}.pdf'.format(save_iter)))
        else:
            plt.show()
        plt.close()

def get_3d_outlier_idxs(logvars_3d):
    std_3d = np.exp(0.5*logvars_3d)
    std_3d_mean = np.linalg.norm(std_3d, axis=2).mean(axis=0)
    cutoff = std_3d_mean.mean() - std_3d_mean.std()
    outlier_idxs = np.where(std_3d_mean < cutoff)[0]
    return outlier_idxs

def plot_tsne(data, save_path=None, save_iter=0, data_name=''):
    if len(data.shape) == 3:
        data = data.reshape(-1,data.shape[1]*data.shape[2])
    data_emb = TSNE(n_components=2).fit_transform(data)
    plt.scatter(data_emb[:,0], data_emb[:,1], s=2)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'tsne_{}_i{}.pdf'.format(data_name, save_iter)))
    else:
        plt.show()
    plt.close()


def plot_2Dpose(ax, pose_2d, bones, bones_dashed=[], bones_dashdot=[], colormap='hsv',
                linewidth=1, limits=None, color_order=[0, 5, 9, 15, 2, 10, 12, 4, 14, 13, 11, 3, 7, 8, 6, 1]):
    cmap = plt.get_cmap(colormap)

    plt.axis('equal')
    maximum = max(color_order) #len(bones)
    for i, bone in enumerate(bones):
        colorIndex = (color_order[i] * cmap.N / float(maximum))
#        color = cmap(int(colorIndex))
#        colorIndex = i / len(bones)
        color = cmap(int(colorIndex))
        ax.plot(pose_2d[0, bone], pose_2d[1, bone], '-', color=color, linewidth=linewidth)
    for bone in bones_dashed:
        ax.plot(pose_2d[0, bone], pose_2d[1, bone], ':', color=color, linewidth=linewidth)
    for bone in bones_dashdot:
        ax.plot(pose_2d[0, bone], pose_2d[1, bone], '--', color=color, linewidth=linewidth)

    if not limits==None:
        ax.set_xlim(limits[0],limits[2])
        ax.set_ylim(limits[1],limits[3])

def plot3Dsphere(ax, p, radius=5, color=(0.5, 0.5, 0.5)):
    num_samples = 8
    u = np.linspace(0, 2 * np.pi, num_samples)
    v = np.linspace(0, np.pi, num_samples)

    x = p[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = p[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = p[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    c = np.ones( (list(x.shape)+[len(color)]) )
    c[:,:] = color
    #ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, alpha=1)
    return x, y, z, c

def plot3Dcylinder(ax, p0, p1, radius=5, color=(0.5, 0.5, 0.5)):
    num_samples = 8
    origin = np.array([0, 0, 0])
    #vector in direction of axis
    v = p1 - p0
    mag = la.norm(v)
    if mag==0: # prevent division by 0 for bones of length 0
        return np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0))
    #unit vector in direction of axis
    v = v / mag
    #make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    eps = 0.00001
    if la.norm(v-not_v)<eps:
        not_v = np.array([0, 1, 0])
    #make vector perpendicular to v
    n1 = np.cross(v, not_v)
    n1 /= la.norm(n1)
    #make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    #surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, 2)
    theta = np.linspace(0, 2 * np.pi, num_samples)
    #use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    #generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + radius * np.sin(theta) * n1[i] + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    #ax.plot_surface(X, Y, Z, color=color, alpha=0.25, shade=True)
    c = np.ones( (list(X.shape)+[4]) )
    c[:,:] = color #(1,1,1,0) #color
    return X, Y, Z, c

def plot_3Dpose(ax, pose_3d, bones, radius=10, colormap='gist_rainbow', color_order=[0, 5, 9, 15, 2, 10, 12, 4, 14, 13, 11, 3, 7, 8, 6, 1], set_limits=True, flip_yz=True, fixed_color = False, transparentBG=False):
    pose_3d = np.reshape(pose_3d, (3, -1))

    ax.view_init(elev=0, azim=-90)
    #plt.colormap(ax,'hsv') #'jet', 'nipy_spectral',tab20
    cmap = plt.get_cmap(colormap)

    if flip_yz:
        X,Y,Z = np.squeeze(np.array(pose_3d[0,:])), np.squeeze(np.array(pose_3d[2,:])), np.squeeze(np.array(pose_3d[1,:]))
    else:
        X,Y,Z = np.squeeze(np.array(pose_3d[0,:])), np.squeeze(np.array(pose_3d[1,:])), np.squeeze(np.array(pose_3d[2,:]))
    XYZ = np.vstack([X,Y,Z])

    # dummy bridge that connects different components (set to transparent)
    def bridge_vertices(xs,ys,zs,cs, x,y,z,c):
        num_samples = x.shape[0]
        if num_samples == 0: # don't build a bridge if there is no data
            return
        if len(cs) > 0:
            x_bridge = np.hstack([xs[-1][:,-1].reshape(num_samples,1), x[:,0].reshape(num_samples,1)])
            y_bridge = np.hstack([ys[-1][:,-1].reshape(num_samples,1), y[:,0].reshape(num_samples,1)])
            z_bridge = np.hstack([zs[-1][:,-1].reshape(num_samples,1),z[:,0].reshape(num_samples,1)])
            c_bridge = np.ones( (num_samples,2,4) )
            c_bridge[:,:] = np.array([0,0,0,0])
            xs.append(x_bridge)
            ys.append(y_bridge)
            zs.append(z_bridge)
            cs.append(c_bridge)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        cs.append(c)
        return

    maximum = max(color_order) #len(bones)
    xs = []
    ys = []
    zs = []
    cs = []
    for i, bone in enumerate(bones):
        assert i < len(color_order)
        colorIndex = None
        if fixed_color == True:
            colorIndex = ( cmap.N / 4)
        else:
            colorIndex = (color_order[i] * cmap.N / float(maximum))
        color = cmap(int(colorIndex))
        x,y,z,c = plot3Dcylinder(ax, XYZ[:,bone[0]], XYZ[:,bone[1]], radius=radius, color=color)
        bridge_vertices(xs,ys,zs,cs, x,y,z,c)
        #x,y,z,c = plot3Dsphere(ax, XYZ[:,bone[0]], radius=radius*1.2, color=color)
        #bridge_vertices(xs,ys,zs,cs, x,y,z,c)
        #x,y,z,c = plot3Dsphere(ax, XYZ[:,bone[1]], radius=radius*1.2, color=color)
        #bridge_vertices(xs,ys,zs,cs, x,y,z,c)

    if len(xs) == 0:
        return

    # merge all sufaces together to one big one
    x_full = np.hstack(xs)
    y_full = np.hstack(ys)
    z_full = np.hstack(zs)
    c_full = np.hstack(cs)

    ax.plot_surface(x_full, y_full, z_full, rstride=1, cstride=1, facecolors=c_full, linewidth=0, antialiased=True)

    # maintain aspect ratio
    #if set_limits_fixed:
    #    ax.set_xlim(-1.5, 1.5)
    #    ax.set_ylim(-1.5, 1.5)
    #    ax.set_zlim(-1.5, 1.5)

    if set_limits:
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if transparentBG:
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off',
            labelsize=8) # labels along the bottom edge are off
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off',
            labelsize=8) # labels along the bottom edge are off
        ax.tick_params(
            axis='z',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off',
            labelsize=8) # labels along the bottom edge are off

        # make the bg white
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


def plot_3Dpose_simple(ax, pose_3d, bones, linewidth=1, colormap='gist_rainbow',
                       color_order=[0, 5, 9, 15, 2, 10, 12, 4, 14, 13, 11, 3, 7, 8, 6, 1], plot_handles=None):
    pose_3d = np.reshape(pose_3d, (3, -1))

    X, Y, Z = np.squeeze(np.array(pose_3d[0, :])), np.squeeze(np.array(pose_3d[2, :])), np.squeeze(
        np.array(pose_3d[1, :]))
    XYZ = np.vstack([X, Y, Z])

    if not plot_handles is None:
        for i, bone in enumerate(bones):
            plot_handles['lines'][i].set_data(XYZ[0:2, bone])
            plot_handles['lines'][i].set_3d_properties(XYZ[2, bone])
    else:
        ax.view_init(elev=0, azim=-90)
        cmap = plt.get_cmap(colormap)

        plot_handles = {'lines': [], 'points': []}
        plt.axis('equal')
        maximum = len(bones)  # max(color_order) #len(bones)
        for i, bone in enumerate(bones):
            assert i < len(color_order)
            # colorIndex = (color_order[i] * cmap.N / float(maximum))
            colorIndex = (i * cmap.N / float(maximum))
            color = cmap(int(colorIndex))
            line_handle = ax.plot(XYZ[0, bone], XYZ[1, bone], XYZ[2, bone], color=color, linewidth=linewidth, alpha=0.5,
                                  solid_capstyle='round')
            plot_handles['lines'].extend(line_handle)  # for whatever reason plot already returns a list

        # maintain aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return plot_handles
