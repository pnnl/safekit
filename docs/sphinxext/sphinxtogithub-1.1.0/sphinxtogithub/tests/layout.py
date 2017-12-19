
from sphinxtogithub.tests import MockExists, MockRemove

import sphinxtogithub
import unittest

class MockHandler(object):

    def __init__(self):

        self.processed = False

    def process(self):

        self.processed = True



class TestLayout(unittest.TestCase):

    def testProcess(self):

        directory_handlers = []
        file_handlers = []

        for i in range(0, 10):
            directory_handlers.append(MockHandler())
        for i in range(0, 5):
            file_handlers.append(MockHandler())

        layout = sphinxtogithub.Layout(directory_handlers, file_handlers)

        layout.process()

        # Check all handlers are processed by reducing them with "and"
        self.assert_(reduce(lambda x, y: x and y.processed, directory_handlers, True))
        self.assert_(reduce(lambda x, y: x and y.processed, file_handlers, True))


def testSuite():
    suite = unittest.TestSuite()

    suite.addTest(TestLayout("testProcess"))

    return suite

