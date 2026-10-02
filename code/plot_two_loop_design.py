"""Plot the analytical two-loop value and optimal bank angle."""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
tstar=brentq(lambda t:(1-t*t)*(np.pi+2*np.arctan(t))-2*t,0,1)
threshold=(tstar+np.sqrt(1+tstar*tstar))**2
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(7.1,2.55),layout='constrained')
phi=np.linspace(.0001,np.pi/2,600)
for ratio in [2,4,10,100]:
    t=(ratio-1)*np.sin(phi)/(2*np.sqrt(ratio))
    gap=1+2*t/((1+t*t)*(np.pi+2*np.arctan(t)))
    ax[0].plot(np.degrees(phi),gap,label=f'$b/a={ratio}$',lw=1.4)
ax[0].axhline(2/(1+tstar*tstar),color='black',ls='--',lw=.9)
ax[0].set(xlabel='Angle between loops (degrees)',ylabel='Optimal risk ratio',xlim=(0,90))
ax[0].legend(fontsize=7.5,frameon=False,loc='lower right')
ratios=np.geomspace(1.01,200,800)
optimal=np.degrees(np.arcsin(np.minimum(1,2*np.sqrt(ratios)*tstar/(ratios-1))))
ax[1].semilogx(ratios,optimal,lw=1.6,color='#245b8f')
ax[1].axvline(threshold,color='black',ls='--',lw=.9)
ax[1].text(threshold*1.12,80,'$b/a=4.342$',fontsize=8)
ax[1].set(xlabel='Damping ratio $b/a$',ylabel='Optimal angle (degrees)',ylim=(0,95),xlim=(1,200))
for a in ax:a.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'figures/two_loop_design_full.pdf')
plt.close(fig)

fig,ax=plt.subplots(figsize=(3.35,2.4),layout='constrained')
for ratio in [2,4,10,100]:
    t=(ratio-1)*np.sin(phi)/(2*np.sqrt(ratio))
    gap=1+2*t/((1+t*t)*(np.pi+2*np.arctan(t)))
    line,=ax.plot(np.degrees(phi),gap,label=f'$b/a={ratio}$',lw=1.4)
    angle=np.arcsin(min(1,2*np.sqrt(ratio)*tstar/(ratio-1)))
    opt_t=(ratio-1)*np.sin(angle)/(2*np.sqrt(ratio))
    opt_gap=1+2*opt_t/((1+opt_t*opt_t)*(np.pi+2*np.arctan(opt_t)))
    ax.plot(np.degrees(angle),opt_gap,'o',ms=3.5,color=line.get_color(),clip_on=False)
ax.axhline(2/(1+tstar*tstar),color='black',ls='--',lw=.9)
ax.set(xlabel='Angle between loops (degrees)',ylabel='Optimal risk ratio',xlim=(0,90),ylim=(.995,1.23))
ax.legend(fontsize=7.5,frameon=False,loc='lower right')
ax.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'figures/two_loop_design.pdf')
plt.close(fig)
