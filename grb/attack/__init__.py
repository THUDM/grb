"""Attack Module for implementation of graph adversarial attacks"""
from .base import Attack, InjectionAttack, ModificationAttack
from .fgsm import FGSM
from .pgd import PGD
from .rnd import RND
from .speit import SPEIT
from .tdgia import TDGIA
