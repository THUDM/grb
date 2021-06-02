from abc import ABCMeta, abstractmethod


class Defense(metaclass=ABCMeta):

    @abstractmethod
    def defense(self, model, adj, features, **kwargs):
        """

        :param model:
        :param features:
        :param adj:
        :param kwargs:
        :return:
        """
