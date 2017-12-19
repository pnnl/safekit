
from sphinxtogithub.tests import MockExists, MockRemove, MockStream

import sphinxtogithub
import unittest
import os



class MockRename(object):

    def __call__(self, from_, to):
        self.from_ = from_
        self.to = to

class TestForceRename(unittest.TestCase):

    def testCall(self):

        rename = MockRename()
        remove = MockRemove()
        renamer = sphinxtogithub.ForceRename(rename, remove)

        from_ = "from"
        to = "to"
        renamer(from_, to)

        self.assertEqual(rename.from_, from_)
        self.assertEqual(rename.to, to)
        self.assertEqual(remove.name, to)


class TestVerboseRename(unittest.TestCase):

    def testCall(self):

        rename = MockRename()
        stream = MockStream()
        renamer = sphinxtogithub.VerboseRename(rename, stream)

        from_ = os.path.join("path", "to", "from")
        to = os.path.join("path", "to", "to")
        renamer(from_, to)

        self.assertEqual(rename.from_, from_)
        self.assertEqual(rename.to, to)
        self.assertEqual(
                stream.msgs[0],
                "Renaming directory '%s' -> '%s'\n" % (os.path.basename(from_), os.path.basename(to))
                )



def testSuite():
    suite = unittest.TestSuite()

    suite.addTest(TestForceRename("testCall"))
    suite.addTest(TestVerboseRename("testCall"))

    return suite

