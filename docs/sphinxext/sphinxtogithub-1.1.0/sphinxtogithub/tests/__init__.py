
class MockExists(object):

    def __call__(self, name):
        self.name = name
        return True

class MockRemove(MockExists):

    pass


class MockStream(object):

    def __init__(self):
        self.msgs = []

    def write(self, msg):
        self.msgs.append(msg)




