from abc import ABCMeta, abstractmethod


class Defense(metaclass=ABCMeta):
    """
    Abstract class for defense.
    """
    @abstractmethod
    def defense(self, model, adj, features, **kwargs):
        r"""

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        adj : scipy.sparse.csr.csr_matrix
            Adjacency matrix in form of ``N * N`` sparse matrix.
        features : torch.FloatTensor
            Features in form of ``N * D`` torch float tensor.
        kwargs :
            Keyword-only arguments.

        """
