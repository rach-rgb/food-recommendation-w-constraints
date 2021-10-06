from abc import *

class foodRecBase(metaclass=ABCMeta):
    top_N = 10
    
    # return list of top-N recommended food for uid s.t. includes iid
    @abstractmethod
    def top_N_const_1(uid, iid):
        pass

    # return list of top-N recommended food for uid s.t. excludes iid
    @abstractmethod
    def top_N_const_2(uid, iid):
        pass

    # return list of top-N recommended food for uid
    # s.t satisfied target nutrient given iid as history
    @abstractmethod
    def top_N_const_3(uid, iid, target):
        pass
        
