from abc import ABCMeta, abstractmethod


class Attack(metaclass=ABCMeta):
    @abstractmethod
    def set_config(self, **kwargs):
        """

        :param kwargs:
        :return:
        """

    @abstractmethod
    def attack(self, model, features, adj, **kwargs):
        """

        :param model:
        :param features:
        :param adj:
        :param kwargs:
        :return:
        """


class ModificationAttack(Attack):
    @abstractmethod
    def attack(self, **kwargs):
        """

        :param kwargs:
        :return:
        """

    @abstractmethod
    def modification(self):
        """

        :return:
        """


class InjectionAttack(Attack):
    @abstractmethod
    def attack(self, **kwargs):
        """

        :param kwargs:
        :return:
        """

    @abstractmethod
    def injection(self, **kwargs):
        """

        :return:
        """

    @abstractmethod
    def update_features(self, **kwargs):
        """

        :return:
        """
