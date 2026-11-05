"""Common figure style for the manuscript (Times-like serif, validated categorical palette, markers as secondary encoding)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#dcdbd7', '#ffffff'
# validated categorical slots (light mode): blue, orange, aqua, violet, magenta, yellow
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', violet='#6250d6', magenta='#e87ba4', yellow='#eda100', gray='#7a7975')
# one colour and one marker per entity, the same in every figure
ENT = {
    'FFP':           dict(color=C['gray'],   marker='s', label='Local'),
    'MPC':           dict(color=C['blue'],   marker='o', label='Predictive'),
    'Fixed':         dict(color=C['aqua'],   marker='^', label='Fixed shift'),
    'FixedBest':     dict(color=C['yellow'], marker='X', label='Best fixed shift'),
    'Morph-nominal': dict(color=C['magenta'], marker='v', label='Morphology, nominal'),
    'Morph-learn':   dict(color=C['orange'], marker='D', label='Morphology, learned'),
    'Morph-oracle':  dict(color=C['violet'], marker='P', label='Morphology, true'),
}


def setup():
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'legend.fontsize': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120, 'savefig.dpi': 300,
        'lines.linewidth': 1.4, 'lines.markersize': 4.5, 'mathtext.fontset': 'stix', 'figure.facecolor': SURF, 'axes.facecolor': SURF,
        'pdf.fonttype': 42,
    })


def panel(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left')


def eqn(name, src='../md/manuscript.md'):
    """Equation number of a label (@name) in the manuscript source (same rule as md/prep.py)."""
    import re
    t = open(src, encoding='utf-8').read()
    defs = re.findall(r'\\qquad \(@(\w+)\)', t)
    return defs.index(name) + 1
