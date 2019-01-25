class History(object):
    def __init__(self, limit):
        self.values = []
        self.limit = limit
        assert self.limit > 0

    def enqueue(self, value):
        self.values.append(value)
        if len(self.values) > self.limit:
            self.values = self.values[-self.limit:]

    def latest(self):
        if self.values: return self.values[-1]
        return None

    def __getitem__(self, index):
        return self.values[index]
