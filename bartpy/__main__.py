import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from bartpy.diagnostics.features import *
from bartpy.diagnostics.residuals import plot_qq, plot_homoskedasity_diagnostics
from bartpy.diagnostics.sampling import plot_tree_mutation_acceptance_rate
from bartpy.diagnostics.sigma import plot_sigma_convergence
from bartpy.diagnostics.trees import plot_tree_depth
from bartpy.sklearnmodel import SklearnModel
from sklearn.ensemble import RandomForestRegressor


def plot_and_save(func, model):
    func(model)
    plt.savefig(func.__name__)
    plt.clf()


def plot_and_save_diagnostics(model: SklearnModel):
    plot_and_save(plot_qq, model)
    plot_and_save(plot_homoskedasity_diagnostics, model)
    plot_and_save(plot_tree_mutation_acceptance_rate, model)
    plot_and_save(plot_sigma_convergence, model)
    plot_and_save(plot_tree_depth, model)


def test_function(X, eta):
    return 5*np.sin(np.pi*X[:, 0]*X[:, 1])+8*((X[:, 2]-0.5)**2)+5*X[:, 0]*X[:, 1]*X[:, 2]+6*np.exp(X[:, 3]*X[:, 4])+np.power(X[:,4],eta)


def generate_data(size = 200, eta=0):
    X = np.random.uniform(size=(size,10))
    X = pd.DataFrame(X).sample(frac=1.0).values
    y = test_function(X, eta).reshape((size,))+np.random.normal(0, 1, size=(size,))
    return X, y


def rf_rmse(X, y, eta):
    regr = RandomForestRegressor(random_state=0)
    regr.fit(X, y)
    return np.sqrt(np.sum((regr.predict(X)-test_function(X, eta))**2))


def plot_performance(predictions):
    eta_vec = np.linspace(0.1,1,10)
    rf_mse = np.load("rf_rmse_array.npy")
    fig, ax = plt.subplots()
    ax.scatter(eta_vec, predictions[0,:,1], label="beta=1")
    ax.scatter(eta_vec, predictions[1,:,1], label="beta=2")
    ax.scatter(eta_vec, predictions[2,:,1], label="beta=3")
    ax.scatter(eta_vec, rf_mse, label="random forest")
    ax.legend()
    plt.savefig("scatter_plot_fig.png")

X = np.load("X_array.npy")
X_test = np.load("X_array.test.npy")
noise = np.load("noise.npy")
eta = 0.5
model = SklearnModel(n_samples=1000, prior_name="exponential_splits", Gamma=3, n_burn=250, n_trees=50,
                                 store_in_sample_predictions=False, n_jobs=1)
model.fit(X,test_function(X,eta)+noise)
rmse = model.rmse(X, y)
print(rmse)

