from .EMKD import EMKD
from .AT import AT
from .ReviewKD import ReviewKD
from .ReviewEhcdAttKD import ReviewEhcdAttKD
from .EhcdAttKD import EhcdAttKD
from .MGD import MGD
from .CrossEhcdAttKD import CrossEhcdAttKD
from .LSR import LSR
from .OFD import OFD
from .TfKD import TfKD
from .SP import SP
from .GID import GID
from .CWKD import CWKD
from .KD import KD
from .RKD import RKD
from .CRD import CRD
from .VID import VID

distiller_dict = {
    "EMKD": EMKD,
    "AT": AT,
    "ReviewKD": ReviewKD,
    "ReviewEhcdAttKD": ReviewEhcdAttKD,
    "EhcdAttKD": EhcdAttKD,
    "MGD": MGD,
    "CrossEhcdAttKD": CrossEhcdAttKD,
    "LSR": LSR,
    "OFD": OFD,
    "TfKD": TfKD,
    "SP": SP,
    "GID": GID,
    "CWKD": CWKD,
    "KD": KD,
    "RKD": RKD,
    "CRD": CRD,
    "VID": VID,
}