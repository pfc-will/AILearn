import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sko.PSO import PSO
from tensorflow.keras.initializers import Initializer


class InitCentersPSO(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
        y: matrix
    """

    def __init__(self, X, y, max_iter=100):
        self.X = X
        y = (y * 1000).astype(int)
        self.y = [[a] for a in y]
        self.y = (y * 1000).astype(int)
        self.max_iter = max_iter
        self.index = 0
        super().__init__()

    

    def __call__(self, shape, dtype=None):
        assert shape[1:] == self.X.shape[1:]

        def loss_func(x):
            rbf_svm = SVC(kernel = 'rbf', C = max(np.mean(x), 0.01), gamma = max(np.var(x), 0.01))
            # rbf_svm = SVC(kernel='rbf', degree=2, gamma=np.var(x))
            cv_scores = cross_val_score(rbf_svm, self.X, self.y.ravel(), cv =3, scoring = 'accuracy')
            self.index += 1
            return cv_scores.mean()

        n_dim = shape[1]
        pop = self.X.shape[0]

        pso = PSO(func=loss_func, n_dim=n_dim, pop=pop, max_iter=self.max_iter, lb=[0, 0, 0, 0, 0], ub=[1, 1, 1, 1, 1], w=0.8, c1=0.5, c2=0.5)
        pso.run()
        return pso.pbest_x[:shape[0]]
