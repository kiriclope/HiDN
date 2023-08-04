import copy
import pickle as pkl

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection


def save_fig(fig, figname, path='.', format='svg', dpi='figure'):

    fig = plt.figure(fig.number)
    if not os.path.isdir(path):
        os.makedirs(path)

    plt.savefig(path + '/' + figname + '.' + format, dpi=dpi, format=format)


def pkl_save(obj, name, path='.'):
    pkl.dump(obj, open(path + '/' + name + '.pkl', 'wb'))


def pkl_load(name, path='.'):
    return pkl.load(open(path + '/' + name, 'rb'))

def copy_fig(fig, ax):

    ax0 = fig.axes[0]

    # labels
    x_label = ax0.xaxis.get_label().get_text()
    y_label = ax0.yaxis.get_label().get_text()
    # print(x_label, y_label)

    # lines
    lines = ax0.get_lines()
    for line in lines:
        xydata = line.get_xydata()
        ax.plot(xydata[:, 0], xydata[:, 1])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # ticks
    x_ticks = ax0.get_xaxis().properties()['ticklocs']
    y_ticks = ax0.get_yaxis().properties()['ticklocs']

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # limits
    x_lim = ax0.get_xaxis().properties()['view_interval']
    y_lim = ax0.get_yaxis().properties()['view_interval']

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # add collections (fill_between and errorbars)
    # polys = fig.findobj(matplotlib.collections)
    collects = ax0.collections
    # print(collects)
    for collect in collects:
        collect_cpy = copy.copy(collect)
        collect_cpy.axes = None
        collect_cpy.figure = None
        collect_cpy.set_transform(ax.transData)
        ax.add_collection(collect_cpy)

    # # fill_between
    # polys = fig.findobj(matplotlib.collections.PolyCollection)
    # # print(polys)
    # for poly in polys:
    #     poly_cpy = copy.copy(poly)
    #     poly_cpy.axes = None
    #     poly_cpy.figure = None
    #     poly_cpy.set_transform(ax.transData)
    #     ax.add_collection(poly_cpy)
    #     # for path in poly.get_paths():
    #     #     points = path.vertices
    #     #     # print(points)
    #     #     ax.fill_between(points[:,0], points[:,1], alpha=.25)

    # # errorbars
    # errorbars = fig.findobj(matplotlib.collections.LineCollection)
    # # errorbars = ax0.collections
    # for errorbar in errorbars:
    #     errorbar_cpy = copy.copy(errorbar)
    #     errorbar_cpy.axes = None
    #     errorbar_cpy.figure = None
    #     errorbar_cpy.set_transform(ax.transData)
    #     ax.add_collection(errorbar_cpy)
    #     # for path in errorbar.get_paths():
    #     #     error = path.vertices
    #     #     # print(error)

    ax.autoscale()

def concat_fig(figname, fig_list, dim=[1,2]):

    fig, ax = plt.subplots(dim[0], dim[1],
                           figsize=(2.427*dim[1], 1.5*dim[0])
                           num=figname)

    fig_iter = iter(fig_list)
    for col in range(ax.shape[0]):
        for row in range(ax.shape[1]):
            copy_fig(next(fig_iter), ax[col][row])


if __name__ == '__main__':

    fig = plt.figure()
    x = np.linspace(0,2*np.pi)

    y = np.sin(x)
    # plt.plot(x,y)
    # plt.ylabel('sin(x)')
    # plt.xlabel('x')
    plt.errorbar(x, y, yerr=.5)
    y = np.cos(x)
    plt.errorbar(x, y, yerr=.5)
    plt.xticks([0,1,2,3])
    plt.yticks([])
    plt.xlim([0,3])

    pkl_save(fig, 'sin')
    plt.close(fig)

    fig = plt.figure()
    y = np.cos(x)
    plt.plot(x,y)
    plt.fill_between(x, y-1, y+1, alpha=.25)
    plt.ylabel('cos(x)')
    plt.xlabel('x')
    plt.xticks([0,1,2,3])
    plt.yticks([])
    plt.xlim([0,3])

    pkl_save(fig, 'cos')
    plt.close(fig)

    cos = pkl_load('cos.pkl')
    sin = pkl_load('sin.pkl')

    plt.close('all')

    fig_list = [cos, sin, cos, sin]
    concat_fig('summary', fig_list, dim=[2,2])
