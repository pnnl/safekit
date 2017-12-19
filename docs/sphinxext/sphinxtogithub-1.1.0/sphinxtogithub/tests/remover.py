
from sphinxtogithub.tests import MockExists, MockRemove

import sphinxtogithub
import unittest

class TestRemover(unittest.TestCase):

    def testCall(self):

        exists = MockExists()
        remove = MockRemove()
        remover = sphinxtogithub.Remover(exists, remove)

        filepath = "filepath"
        remover(filepath)

        self.assertEqual(filepath, exists.name)
        self.assertEqual(filepath, remove.name)


def testSuite():
    suite = unittest.TestSuite()

    suite.addTest(TestRemover("testCall"))

    return suite

