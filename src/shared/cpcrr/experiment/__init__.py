__all__ = [
    'OneWay',
    'Twokr',
    'studentt',
    'Fnm',
]

from .oneway import OneWay
from .twokr import Twokr
from .experiment_impl.stats_tables import (
    studentt,
    Fnm,
)




