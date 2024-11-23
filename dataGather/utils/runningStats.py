class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self):
        return self.M2 / self.n if self.n > 1 else 0

    @property
    def standard_deviation(self):
        return self.variance ** 0.5