from .clever import CLEVER
from .fast_lip import FastLip 
from .lip_lp import LipLP
from .naive_methods import NaiveUB, RandomLB
from .other_methods import OtherResult

OTHER_METHODS = [CLEVER, FastLip, LipLP, NaiveUB, RandomLB]
LOCAL_METHODS = [CLEVER, FastLip, LipLP, RandomLB]
GLOBAL_METHODS = [ NaiveUB]