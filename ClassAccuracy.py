class Node(object):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # The class "constructor" - It's actually an initializer
    def __init__(self, tp, tn, fp, fn):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn


