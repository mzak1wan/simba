import numpy as np
import matplotlib.pyplot as plt

def plot_random(names, mses, ylim=[3.], grid=[0.25], savename=None, scale=2):

    all_mses = np.concatenate(mses, axis=1)
    if len(all_mses) != len(names):
        all_mses = all_mses.T
        
    todel = []
    for i,k in enumerate(names):
        if ('LLS' in k) or ('ARX' in k) or ('mat' in k):
            todel.append(i)
    for k in todel:
        all_mses[k,:] = np.inf
    all_mses = all_mses / all_mses.min(axis=0)
    mses = {name: all_mses[i,:] for i, name in enumerate(names)}
    mses_ = sorted(mses.items(), key=lambda x:np.median(x[1]))[:-len(todel)]
    # Reorder
    mses = [] 
    if mses_[0][0][:3] == 'PAR':
        mses = mses + mses_[1:-2]
        mses.append(mses_[0])
        mses = mses + mses_[-2:]
    elif mses_[1][0][:3] == 'PAR':
        if mses_[-1][0][:3] == 'SIM':
            mses.append(mses_[0])
            mses = mses + mses_[2:-4]
            mses.append(mses_[-3])
            mses.append(mses_[-1]) # Add bad SIMBa
            mses.append(mses_[1])
            mses.append(mses_[4])
            mses.append(mses_[-2])
        else:
            mses.append(mses_[0])
            mses = mses + mses_[2:-2]
            mses.append(mses_[1])
            mses = mses + mses_[-2:]
    elif (mses_[5][0][:3] == 'PAR') and (mses_[-1][0][:3] == 'SIM'):
        mses = mses_[:5]
        mses = mses + mses_[6:-3]
        mses.append(mses_[-1]) # Add bad SIMBa
        mses.append(mses_[5])
        mses = mses + mses_[-3:-1]

    if not isinstance(ylim, list):
        ylim = [ylim]
    if not isinstance(grid, list):
        grid = [grid]

    fig, ax = plt.subplots(len(ylim),1,figsize=(16,8*len(ylim)+4), sharex=True)   
    if len(ylim) == 1:
        ax = [ax]
    print('Method\t\t&\tQ1\t&\tMedian\t&\tQ3') 
    for m in range(len(ylim)):
        for i in range(len(mses)):
            if m == 0:
                if len(mses[i][0]) < 8:
                    print(f'{mses[i][0]}\t\t&\t{np.quantile(mses[i][1], q=0.25):.2f}\t&\t{np.median(mses[i][1]):.2f}\t&\t{np.quantile(mses[i][1], q=0.75):.2f}\t\\\\')
                else:
                    print(f'{mses[i][0]}\t&\t{np.quantile(mses[i][1], q=0.25):.2f}\t&\t{np.median(mses[i][1]):.2f}\t&\t{np.quantile(mses[i][1], q=0.75):.2f}\t\\\\')
            ticks = list(np.array([i+1]*all_mses.shape[1]) + (np.random.rand(all_mses.shape[1])*0.3 - 0.15))
            c = 'tab:green' if 'SIMBa' in mses[i][0] else ('tab:red' if 'PARSIM' in mses[i][0] else ('tab:orange' if 'mat-' in mses[i][0] else 'tab:blue'))
            ax[m].scatter(ticks, mses[i][1], s=200*scale, alpha=0.25, c=c)
            ax[m].boxplot(mses[i][1], positions=[i+1], sym='', widths=0.5, 
                        boxprops=dict(color=c, lw=2.5*scale), medianprops=dict(color=c, lw=3.5*scale),
                        whiskerprops=dict(color=c, lw=2*scale), capprops=dict(color=c, lw=2*scale))#, patch_artist=True, boxprops=dict(color=c))
        ax[m].set_xticks(np.arange(len(mses))+1)
        ax[m].set_xticklabels([x[0] for x in mses], rotation = 90, size=20*scale)
        ax[m].vlines(len(mses) - 2.5, 1-grid[m]/5, ylim[m]+grid[m]/5, color='black', ls='--', lw=3*scale, clip_on=False)
        ax[m].set_yticks(np.arange((ylim[m]-1) * (1/grid[m]) + 1)*grid[m]+1)
        ax[m].set_yticklabels(ax[m].get_yticks(), size=20*scale)
        ax[m].set_ylim(1-grid[m]/5, ylim[m]+grid[m]/5)
        ax[m].set_ylabel('Normalized MSE', size=20*scale)
        ax[m].grid()
    ax[-1].set_xlabel('Method', size=20*scale)
    if len(mses) == 8:
        ax[0].set_title('Always stable                      Potentially unstable  ', size=20*scale, loc='right', fontweight='bold')
    elif len(mses) == 7:
        ax[0].set_title('Always stable                      Potentially unstable     ', size=20*scale, loc='right', fontweight='bold')
    else:
        ax[0].set_title('Always                                    Potentially  \n stable                                       unstable    ', size=20*scale, loc='right', fontweight='bold')
    fig.tight_layout()
    if savename is not None:
        fig.savefig(f'figures/{savename}.pdf', format='pdf')
    plt.show()

def plot_erros_many(names, mses, ylim=[3.], grid=[0.25], savename=None, scale=2):
        
    todel = []
    for i,k in enumerate(names):
        if ('LLS' in k) or ('ARX' in k) or ('mat' in k):
            todel.append(i)
    for k in todel:
        for i in range(len(mses)):
            mses[i][k] = np.inf

    mins = []
    for i in range(len(mses)):
        min1 = np.min(mses[i][:-2])
        min2 = min(np.min(mses[i][-2]), np.min(mses[i][-1]))
        mins.append(min(min1,min2))
        mses[i][:-2] = np.array(mses[i][:-2]) / mins[-1]
        mses[i][-2] = np.array(mses[i][-2]) / mins[-1]
        mses[i][-1] = np.array(mses[i][-1]) / mins[-1]

    msesdic = {name: [] for i, name in enumerate(names)}
    for i, name in enumerate(names):
        for k in range(len(mses)):
            msesdic[name].append(mses[k][i])
    msesdic['SIMBa-3'] = list(np.concatenate(msesdic['SIMBa-3']).reshape(-1,))
    msesdic['SIMBa-4'] = list(np.concatenate(msesdic['SIMBa-4']).reshape(-1,))
    mses_ = sorted(msesdic.items(), key=lambda x:np.median(x[1]))[:-len(todel)]

    # Reorder
    mses = [] 
    mses.append(mses_[0])
    mses = mses + mses_[2:4]
    mses.append(mses_[5])
    mses.append(mses_[-1])
    mses.append(mses_[1])
    mses.append(mses_[4])
    mses.append(mses_[6])

    if not isinstance(ylim, list):
        ylim = [ylim]
    if not isinstance(grid, list):
        grid = [grid]

    fig, ax = plt.subplots(len(ylim),1,figsize=(16,7*len(ylim)+5), sharex=True)   
    if len(ylim) == 1:
        ax = [ax]
    print('Method\t\t&\tQ1\t&\tMedian\t&\tQ3') 
    for m in range(len(ylim)):
        for i in range(len(mses)):
            if m == 0:
                    if len(mses[i][0]) < 8:
                        print(f'{mses[i][0]}\t\t&\t${np.quantile(mses[i][1], q=0.25):.2f}$\t&\t${np.median(mses[i][1]):.2f}$\t&\t${np.quantile(mses[i][1], q=0.75):.2f}$\t\\\\')
                    else:
                        print(f'{mses[i][0]}\t&\t${np.quantile(mses[i][1], q=0.25):.2f}$\t&\t${np.median(mses[i][1]):.2f}$\t&\t${np.quantile(mses[i][1], q=0.75):.2f}$\t\\\\')
            ticks = list(np.array([i+1]*len(mses[i][1])) + (np.random.rand(len(mses[i][1]))*0.3 - 0.15))
            c = 'tab:green' if 'SIMBa' in mses[i][0] else ('tab:red' if 'PARSIM' in mses[i][0] else ('tab:orange' if 'mat-' in mses[i][0] else 'tab:blue'))
            ax[m].scatter(ticks, mses[i][1], s=200*scale, alpha=0.25, c=c)
            ax[m].boxplot(mses[i][1], positions=[i+1], sym='', widths=0.5, 
                        boxprops=dict(color=c, lw=2.5*scale), medianprops=dict(color=c, lw=3.5*scale),
                        whiskerprops=dict(color=c, lw=2*scale), capprops=dict(color=c, lw=2*scale))#, patch_artist=True, boxprops=dict(color=c))
        ax[m].set_xticks(np.arange(len(mses))+1)
        ax[m].set_xticklabels([x[0] for x in mses], rotation = 90, size=20*scale)
        ax[m].vlines(len(mses) - 2.5, 1-grid[m]/5, ylim[m]+grid[m]/5, color='black', ls='--', lw=3*scale, clip_on=False)
        ax[m].set_yticks(np.arange((ylim[m]-1) * (1/grid[m]) + 1)*grid[m]+1)
        ax[m].set_yticklabels(ax[m].get_yticks(), size=20*scale)
        ax[m].set_ylim(1-grid[m]/5, ylim[m]+grid[m]/5)
        ax[m].set_ylabel('Normalized MSE', size=20*scale)
        ax[m].grid()
    ax[-1].set_xlabel('Method', size=20*scale)
    ax[0].set_title('Always stable                      Potentially unstable  ', size=20*scale, loc='right', fontweight='bold')
    fig.tight_layout()
    if savename is not None:
        fig.savefig(f'figures/{savename}.pdf', format='pdf')
    plt.show()


def plot_daisy(nxs, perf, savename=None, scale=2, normalize=False, improvement=False, ylim=None, toplot=20):

    todel = []
    for k in perf.keys():
        if ('mat-' not in k) and ('SIMB' not in k):
            todel.append(k)
    for k in todel:
        del perf[k]

    best = {}
    per = {}
    per['SIMBa'] = []
    per['SIMBa_l'] = []
    best['SIMBa'] = []
    best['SIMBa_l'] = []
    for i in range(len(perf['SIMBa_i'])):
        l = []
        l_l = []
        for k,v in perf.items():
            if 'SIMBa_l' in k:
                l_l.append(v[i])
            elif ('SIMBa_' in k) and ('SIMBa_i' not in k):
                l.append(v[i])
        per['SIMBa'].append(l)
        per['SIMBa_l'].append(l_l)
        best['SIMBa'].append(np.min(l))
        best['SIMBa_l'].append(np.min(l_l))
    perf['SIMBa'] = np.array(per['SIMBa'])
    perf['SIMBa_l'] = np.array(per['SIMBa_l'])

    for k in ['mat-N4SID', 'mat-PEM', 'SIMBa_i', 'SIMBa_il']:
        best[k] = perf[k]

    bests = []
    for i in range(len(best['SIMBa'])):
        l = []
        for v in best.values():
            l.append(v[i])
        bests.append(np.min(l))
    bests = np.array(bests)

    if normalize:
        for k,v in perf.items():
            if len(v.shape) == 1:
                perf[k] = v/bests
            else:
                perf[k] = v/bests.reshape(-1,1)
    
    elif improvement:
        base = np.array([min(a,b) for a,b in zip(perf['mat-N4SID'], perf['mat-PEM'])])
        for k,v in perf.items():
            if len(v.shape) == 1:
                perf[k] = v/base
            else:
                perf[k] = v/base.reshape(-1,1)

    for k,v in perf.items():
        perf[k] = v[:toplot]

    fig, ax = plt.subplots(1,1,figsize=(16,12))
    imps = {'SIMBa': perf['SIMBa'], 'SIMBa_l': perf['SIMBa_l']}
    colors = {'SIMBa': 'black', 'SIMBa_l': 'tab:green'}
    width = 0.35  # the width of the bars
    multiplier = 0

    x = np.arange(len(nxs[:toplot]))+width/2+1
    if improvement:
        ax.hlines(1, x[0]-width/2-0.5, x[-1]+width/2+0.5, color='tab:red', lw=3*scale, ls='--')

    bps = [4,4]
    print('Nx\t&\tMin\t&\tQ1\t&\tMedian\t&\tQ3') 
    for method, imp in imps.items():
        i = multiplier % 2
        if i == 0:
            offset = width * multiplier - 0.025
        else:
            offset = width * multiplier + 0.025
        print(method)
        for i in range(imp.shape[0]):
            print(f'{nxs[i]}\t&\t${np.min(imp[i,:]):.2f}$\t&\t${np.quantile(imp[i,:], q=0.25):.2f}$\t&\t${np.median(imp[i,:]):.2f}$\t&\t${np.quantile(imp[i,:], q=0.75):.2f}$\t\\\\')
            ticks = list(np.array([i+1]*imp.shape[1]) + (np.random.rand(imp.shape[1])*0.3 - 0.15) + offset)
            c = colors[method]
            ax.scatter(ticks, imp[i,:], s=200*scale, alpha=0.25, c=c, label=method if i == 0 else None)
            bps[multiplier] = ax.boxplot(imp[i,:], positions=[i+1+offset], sym='', widths=0.3, labels=[method],
                        boxprops=dict(color=c, lw=2.5*scale), medianprops=dict(color=c, lw=3.5*scale),
                        whiskerprops=dict(color=c, lw=2*scale), capprops=dict(color=c, lw=2*scale))
        multiplier += 1

    ax.scatter(x, perf['SIMBa_i'], s=200*scale, marker='x', c='blue', lw=3*scale, zorder=122, label='SIMBa_i')
    ax.scatter(x, perf['SIMBa_il'], s=200*scale, marker='^', c='tab:blue', lw=3*scale, zorder=121, label='SIMBa_iL')
    ax.scatter(x, perf['mat-N4SID'], s=200*scale, marker='x', c='tab:red', lw=3*scale, zorder=122, label='mat-N4SID')
    ax.scatter(x, perf['mat-PEM'], s=200*scale, marker='^', c='red', lw=3*scale, zorder=122, label='mat-PEM')
    ax.set_xticks(x)
    ax.tick_params(axis='both', which='major', width=1*scale, length=5*scale)
    ax.tick_params(axis='both', which='minor', width=1*scale, length=2.5*scale)
    ax.set_xticklabels(nxs[:toplot], rotation = 30, size=20*scale)
    if normalize:
        ax.set_ylim(0.9,5.1)
    elif ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    ax.vlines(np.arange(len(nxs[:toplot])+1)+width/2+1-0.5, ax.get_ylim()[0], ax.get_ylim()[1], colors='grey',alpha=0.5)
    ax.set_yticklabels(ax.get_yticks(), size=20*scale)
    if normalize or improvement:
        ax.set_ylabel('Normalized MSE', size=20*scale)
    else:
        ax.set_ylabel('MSE', size=20*scale)
    ax.set_xlabel('State dimension ($n$)', size=20*scale)
    ax.grid(axis='y')
    ax.legend(prop={'size':20*scale}, ncols=3, bbox_to_anchor=(0.5,-.15), loc='upper center')
    ax.legend([x["boxes"][0] for x in bps]+ax.get_legend().legend_handles[2:], 
              ['SIMBa', 'SIMBa_L']+[x._text for x in ax.get_legend().texts[2:]], 
              prop={'size':20*scale}, ncols=3, bbox_to_anchor=(0.47,-.15), loc='upper center')
    fig.tight_layout()
    if savename is not None:
        fig.savefig(f'figures/{savename}.pdf', format='pdf')
    plt.show()

def plot_franka(mses_init, mses, trajs_, savename=None, scale=2):
    
    fig, ax = plt.subplots(1,1,figsize=(16,14))

    imps = {'SIMBa_i': np.array([x[1:] for x in mses_init]), 'SIMBa': np.array([x[1:] for x in mses])}
    colors = {'SIMBa_i': 'black', 'SIMBa': 'tab:green'}
    width = 0.35  # the width of the bars
    multiplier = 0


    ax.scatter(np.arange(len(mses))+width/2, [x[0] for x in mses], c='tab:orange', s=200*scale, label='LS', marker='^', lw=3*scale)
    ax.scatter(np.arange(len(mses))+width/2, [x[0] for x in mses_init], c='tab:red', s=200*scale, label='SOC', marker='x', lw=3*scale)
    
    bps = [4,4]
    print('Nx\t&\tMin\t&\tQ1\t&\tMedian\t&\tQ3') 
    for method, imp in imps.items():
        i = multiplier % 2
        if i == 0:
            offset = width * multiplier - 0.025
        else:
            offset = width * multiplier + 0.025
        print(method)
        for i in range(imp.shape[0]):
            print(f'{trajs_[i]}\t&\t${np.min(imp[i,:]):.2f}$\t&\t${np.quantile(imp[i,:], q=0.25):.2f}$\t&\t${np.median(imp[i,:]):.2f}$\t&\t${np.quantile(imp[i,:], q=0.75):.2f}$\t\\\\')
            ticks = list(np.array([i]*imp.shape[1]) + (np.random.rand(imp.shape[1])*0.3 - 0.15) + offset)
            c = colors[method]
            ax.scatter(ticks, imp[i,:], s=200*scale, alpha=0.25, c=c, label=method if i == 0 else None)
            bps[multiplier] = ax.boxplot(imp[i,:], positions=[i+offset], sym='', widths=0.3, labels=[method],
                        boxprops=dict(color=c, lw=2.5*scale), medianprops=dict(color=c, lw=3.5*scale),
                        whiskerprops=dict(color=c, lw=2*scale), capprops=dict(color=c, lw=2*scale))
        multiplier += 1
    
    ax.set_xticks(np.arange(len(mses))+width/2)
    ax.tick_params(axis='both', which='major', width=1*scale, length=5*scale)
    ax.tick_params(axis='both', which='minor', width=1*scale, length=2.5*scale)
    ax.set_xticklabels(trajs_, rotation = 45, size=20*scale)
    ax.set_yticklabels(ax.get_yticks(), size=20*scale)
    ax.set_ylabel('Test MSE', size=20*scale)
    ax.set_xlabel('Training trajectories', size=20*scale)
    ax.grid(axis='y')
    ax.vlines(np.arange(len(mses)+1)+width/2-0.5,0,100,colors='grey',alpha=0.5)
    ax.set_yscale('log')
    ax.set_ylim(0.0009, 0.2)
    ax.legend(prop={'size':20*scale}, ncols=4, bbox_to_anchor=(0.45,-.75), loc='upper center')
    ax.legend([x["boxes"][0] for x in bps]+ax.get_legend().legend_handles[:2], 
              ['SIMBa_i', 'SIMBa']+[x._text for x in ax.get_legend().texts[:2]], 
              prop={'size':20*scale}, ncols=4, bbox_to_anchor=(0.45,-.4), loc='upper center')
    
    fig.tight_layout()
    if savename is not None:
        fig.savefig(f'figures/{savename}.pdf', format='pdf')
    plt.show()

def plot_franka_boxplots(mses_init, mses, trajs_, savename=None, scale=2):

    fig, ax = plt.subplots(1,1,figsize=(16,12))
    x = np.arange(len(trajs_))  # the label locations
    width = 0.35  # the width of the bars
    multiplier = 0

    imp = (1-np.array([x[1:] for x in mses]) / np.array([x[0] for x in mses_init]).reshape(-1,1))*100
    imp_init = (1-np.array([x[1:] for x in mses_init]) / np.array([x[0] for x in mses_init]).reshape(-1,1))*100
    
    imps = {'SIMBa_i': imp_init, 'SIMBa': imp}
    colors = {'SIMBa_i': 'black', 'SIMBa': 'tab:green'}

    bps = [4,4]
    print('Nx\t&\tQ1\t&\tMedian\t&\tQ3') 
    for method, imp in imps.items():
        i = multiplier % 2
        if i == 0:
            offset = width * multiplier - 0.025
        else:
            offset = width * multiplier + 0.025
        print(method)
        for i in range(imp.shape[0]):
            print(f'{trajs_[i]}\t&\t${np.quantile(imp[i,:], q=0.25):.2f}$\t&\t${np.median(imp[i,:]):.2f}$\t&\t${np.quantile(imp[i,:], q=0.75):.2f}$\t\\\\')
            ticks = list(np.array([i+1]*imp.shape[1]) + (np.random.rand(imp.shape[1])*0.3 - 0.15) + offset)
            c = colors[method]
            ax.scatter(ticks, imp[i,:], s=200*scale, alpha=0.25, c=c)#, label=method if i==0 else None)
            bps[multiplier] = ax.boxplot(imp[i,:], positions=[i+offset+1], sym='', widths=0.3, labels=[method],
                        boxprops=dict(color=c, lw=2.5*scale), medianprops=dict(color=c, lw=3.5*scale),
                        whiskerprops=dict(color=c, lw=2*scale), capprops=dict(color=c, lw=2*scale))
        multiplier += 1

    ax.vlines(np.arange(len(mses)+1)+width/2+1-0.5,0,100,colors='grey',alpha=0.5)
    ax.set_xticks(np.arange(len(mses))+width/2+1)
    ax.tick_params(axis='both', which='major', width=1*scale, length=5*scale)
    ax.tick_params(axis='both', which='minor', width=1*scale, length=2.5*scale)
    ax.set_xticklabels(trajs_, rotation = 45, size=20*scale)
    ax.set_ylim(0,100)
    ax.set_yticklabels([int(x) for x in ax.get_yticks()], size=20*scale)
    ax.set_ylabel('Improvement [%]', size=20*scale)
    ax.set_xlabel('Training trajectories', size=20*scale)
    ax.grid(axis='y')
    ax.legend([x["boxes"][0] for x in bps], ['SIMBa_i', 'SIMBa'], prop={'size':20*scale})
    #ax.legend(prop={'size':20*scale}) #, ncols=2, bbox_to_anchor=(0.5,-.35), loc='upper center')
    fig.tight_layout()
    if savename is not None:
        fig.savefig(f'figures/{savename}.pdf', format='pdf')
    plt.show()

def plot_times(ts, nxs, savename=None, scale=2):

    fig, ax = plt.subplots(1,1,figsize=(16,12))
    imps = {'SIMBa_i': ts.iloc[10:,:].transpose().values, 'SIMBa': ts.iloc[:10,:].transpose().values}
    colors = {'SIMBa_i': 'black', 'SIMBa': 'tab:green'}
    x = np.arange(len(nxs))  # the label locations
    width = 0.35  # the width of the bars
    multiplier = 0

    bps = [4,4]
    print('Nx\t&\tQ1\t&\tMedian\t&\tQ3') 
    for j, (method, imp) in enumerate(imps.items()):
        i = multiplier % 2
        if i == 0:
            offset = width/2 #width * multiplier - 0.025
        else:
            offset = width/2 #width * multiplier + 0.025
        print(method)
        for i in range(imp.shape[0]):
            print(f'{nxs[i]}\t&\t{np.quantile(imp[i,:], q=0.25):.2f}\t&\t${np.median(imp[i,:]):.2f}\t&\t{np.quantile(imp[i,:], q=0.75):.2f}\t\\\\')
            ticks = list(np.array([i+1]*imp.shape[1]) + (np.random.rand(imp.shape[1])*0.3 - 0.15) + offset)
            c = colors[method]
            print(imp[i,:])
            ax.scatter(ticks, imp[i,:], s=200*scale, alpha=0.25, c=c)
        multiplier += 1
        ax.plot(x+1+width/2, imp.mean(axis=1), lw=2*scale, c=c, ls='--', label=method)
    
    print('')
    print('', end='\t')
    for nx in nxs:
        print(f'{nx}', end="\t")
    print('\\\\', end='\n')
    for method, imp in imps.items():
        print(method, end='\t')
        for i,nx in enumerate(nxs):
            if np.median(imp[i,:])<1:
                print(f'{np.median(imp[i,:]):.2f}', end="\t")
            else:
                print(f'{int(np.median(imp[i,:]))}', end="\t")
        print('\\\\', end='\n')

    ax.set_xticks(np.arange(len(nxs))+width/2+1)
    ax.tick_params(axis='both', which='major', width=1*scale, length=5*scale, labelsize=20*scale)
    ax.tick_params(axis='both', which='minor', width=1*scale, length=2.5*scale, labelsize=20*scale)
    ax.set_xticklabels(nxs, rotation=45, size=20*scale)
    ax.set_ylabel('Time [s]', size=20*scale)
    ax.set_xlabel('Training trajectories', size=20*scale)
    ax.grid(axis='y')
    ax.legend(prop={'size':20*scale})
    lims = ax.get_ylim()
    ax.vlines(np.arange(len(nxs)+1)+width/2+1-0.5,ax.get_ylim()[0]-10000,ax.get_ylim()[1]+10000,colors='grey',alpha=0.5)
    ax.set_ylim(lims)
    fig.tight_layout()
    if savename is not None:
        fig.savefig(f'figures/{savename}.pdf', format='pdf')
    plt.show()

def plot_times_boxplots(rts, trajs_, savename=None, scale=2):

    fig, ax = plt.subplots(1,1,figsize=(16,12))
    x = np.arange(len(trajs_))  # the label locations
    width = 0.35  # the width of the bars
    multiplier = 0

    imp = rts.iloc[:10,:].values.T
    imp_init = rts.iloc[10:,:].values.T
    
    imps = {'SIMBa_i': imp_init, 'SIMBa': imp}
    colors = {'SIMBa_i': 'black', 'SIMBa': 'tab:green'}

    bps = [4,4]
    print('Nx\t&\tQ1\t&\tMedian\t&\tQ3') 
    for method, imp in imps.items():
        i = multiplier % 2
        if i == 0:
            offset = width * multiplier - 0.025
        else:
            offset = width * multiplier + 0.025
        print(method)
        for i in range(imp.shape[0]):
            print(f'{trajs_[i]}\t&\t${np.quantile(imp[i,:], q=0.25):.2f}$\t&\t${np.median(imp[i,:]):.2f}$\t&\t${np.quantile(imp[i,:], q=0.75):.2f}$\t\\\\')
            ticks = list(np.array([i+1]*imp.shape[1]) + (np.random.rand(imp.shape[1])*0.3 - 0.15) + offset)
            c = colors[method]
            ax.scatter(ticks, imp[i,:], s=200*scale, alpha=0.25, c=c)#, label=method if i==0 else None)
            bps[multiplier] = ax.boxplot(imp[i,:], positions=[i+offset+1], sym='', widths=0.3, labels=[method],
                        boxprops=dict(color=c, lw=2.5*scale), medianprops=dict(color=c, lw=3.5*scale),
                        whiskerprops=dict(color=c, lw=2*scale), capprops=dict(color=c, lw=2*scale))
        multiplier += 1

    lims = ax.get_ylim()
    ax.vlines(np.arange(len(imp)+1)+width/2+1-0.5, ax.get_ylim()[0]-10000, ax.get_ylim()[1]+10000, colors='grey',alpha=0.5)
    ax.set_ylim(lims)
    ax.set_xticks(np.arange(len(imp))+width/2+1)
    ax.tick_params(axis='both', which='major', width=1*scale, length=5*scale)
    ax.tick_params(axis='both', which='minor', width=1*scale, length=2.5*scale)
    ax.set_xticklabels(trajs_, rotation = 45, size=20*scale)
    ax.set_yticklabels([int(x) for x in ax.get_yticks()], size=20*scale)
    ax.set_ylabel('Time [s]', size=20*scale)
    ax.set_xlabel('Training trajectories', size=20*scale)
    ax.grid(axis='y')
    ax.legend([x["boxes"][0] for x in bps], ['SIMBa_i', 'SIMBa'], prop={'size':20*scale})
    #ax.legend(prop={'size':20*scale}) #, ncols=2, bbox_to_anchor=(0.5,-.35), loc='upper center')
    fig.tight_layout()
    if savename is not None:
        fig.savefig(f'figures/{savename}.pdf', format='pdf')
    plt.show()