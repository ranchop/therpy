import collections


################################################################################
##############################    sdict    #####################################
################################################################################
'''
Ordered dictionary that can also be indexed like lists.
'''
class sdict(collections.OrderedDict):
    def __init__(self, *args, **kwargs):
        super(sdict, self).__init__(*args, **kwargs)
    def __getitem__(self, k):
        if type(k) == slice: return list(self.values())[k]
        if type(k) in [int, float]: k = list(self.keys())[int(k)]
        return super(sdict, self).__getitem__(k)
    def __setitem__(self, k, v):
        if type(k) == slice:
            for ki,vi in zip(list(self.keys())[k], v): super(sdict, self).__setitem__(ki, vi)
        elif type(k) in [int, float]: super(sdict, self).__setitem__(list(self.keys())[int(k)], v)
        else: super(sdict, self).__setitem__(k, v)
    def __iter__(self):
        for v in list(self.values()): yield v
    @property
    def k(self): return list(self.keys())
    @property
    def v(self): return list(self.values())



################################################################################
##############################    NAME     #####################################
################################################################################
