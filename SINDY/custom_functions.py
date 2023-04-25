import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def plot_pp(x, xdot, sim_x, sim_xdot, xlabel, ylabel, figure_name, directory=None, grid=False):

    plt.plot(x, xdot, 'b', alpha=0.8, linewidth=1, label='True')
    plt.plot(sim_x, sim_xdot, 'r--', linewidth=1, label='Identified')
    plt.xlabel(r"$" + xlabel + "$", fontsize=16)
    plt.ylabel(r"$" + ylabel + "$", fontsize=16)
    plt.legend(loc='upper right')
    if directory is not None:
        plt.savefig(directory + '/' + figure_name + '.pdf', format='pdf', dpi=300)
    if grid is True:
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.show()



def plot_comparison(t, true, identified,  xlabel, ylabel, figure_name, directory=None, grid=False):

    plt.plot(t, true, 'b', alpha=0.4, linewidth=1.5, label='True')
    plt.plot(t, identified, 'r--', linewidth=1.5, label='Identified')
    plt.xlabel(r"$" + xlabel + "$", fontsize=16)
    plt.ylabel(r"$" + ylabel + "$", fontsize=16)
    plt.legend(loc='upper right')
    plt.ylim(min(true)-(max(true)-min(true))/3.5, max(true)+(max(true)-min(true))/3.5)
    if directory is not None:
        plt.savefig(directory + '/' + figure_name + '.pdf', format='pdf', dpi=300)
    if grid is True:
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.show()



def plot_lissajous(x, xdot, xlabel, ylabel, figure_name, directory=None, grid=False):
    plt.plot(x, xdot, 'b', linewidth=2, label='True')
    plt.xlabel(r"$" + xlabel + "$", fontsize=16)
    plt.ylabel(r"$" + ylabel + "$", fontsize=16)
    if directory is not None:
        plt.savefig(directory + '/' + figure_name + '.pdf', format='pdf', dpi=300)
    if grid is True:
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.show()

def get_metrics(x, sim, feature_names):
    metrics = {}
    for i, feature in enumerate(feature_names):
        metrics[feature] = {}
        metrics[feature]["R2"] = r2_score(x[:,i], sim[:,i])
        metrics[feature]["MSE"] = mean_squared_error(x[:,i], sim[:,i])
    return metrics


