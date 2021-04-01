from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    # @abstractmethod
    # def config(self, **kwargs):
    #     raise NotImplementedError

    @abstractmethod
    def forward(self, x, adj, dropout):
        raise NotImplementedError
