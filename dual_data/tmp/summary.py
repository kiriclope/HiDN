import glob
import numpy as np
import matplotlib.pyplot as plt

import common.constants as gv
from common.plot_utils import save_fig, pkl_load, concat_fig

if __name__ == '__main__':

    files = glob.glob(gv.figdir+'/*.pkl')
    files.sort()
    files.reverse()

    figlist = [pkl_load(file, path='') for file in files]
    plt.close('all')

    figname = 'summary'

    fig = concat_fig(figname, figlist, dim=[1,3])

    save_fig(fig, figname, path=gv.figdir)
