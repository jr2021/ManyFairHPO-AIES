import seaborn as sns
import matplotlib
from loguru import logger
logger.info('Init plotting tools')

# Set the things for seaborn and matplotlib
sns.set(color_codes=True, context='paper', font_scale=1)

pyplot_global_fonts = {
    # 'font.family'      : 'normal',
    'font.size'        : 16,
    'font.weight'      : 'bold',

    'axes.titlesize'   : 22,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 16,
    'axes.labelweight' : 'bold',

    # 'axes.facecolor'   : (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    # 'savefig.facecolor': (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),

    'grid.linestyle'   : '--',
    'grid.linewidth'   : 1.4,
    'grid.alpha'       : 1,

    'xtick.labelsize'  : 16,
    'ytick.labelsize'  : 16,

    'lines.linewidth'  : 3,
    'lines.markersize' : 12,

    'legend.fontsize'  : 14,
}

matplotlib.rcParams.update(pyplot_global_fonts)

legend_size_large_plots = {'size': 22}

step_function_fonts = {'color': 'black'}

bar_plot_fonts = {
    'linewidth':  0.8,
    'edgecolor':  'white',
    'width':      0.5,
    'legend':     False
}
