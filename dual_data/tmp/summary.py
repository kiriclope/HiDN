import glob

import common.constants as gv
import matplotlib.pyplot as plt
import numpy as np
from common.plot_utils import concat_fig, pkl_load, save_fig

if __name__ == '__main__':

    files = glob.glob(gv.figdir+'/*.pkl')
    files.sort()
    files.reverse()

    figlist = [pkl_load(file, path='') for file in files]
    plt.close('all')

    figname = 'summary'

    fig = concat_fig(figname, figlist, dim=[1,3])

    save_fig(fig, figname, path=gv.figdir)
