
import sphinxtogithub
import unittest

class MockApp(object):

    def __init__(self):
        self.config_values = {}
        self.connections = {}

    def add_config_value(self, name, default, rebuild):

        self.config_values[name] = (default, rebuild)

    def connect(self, stage, function):

        self.connections[stage] = function


class TestSetup(unittest.TestCase):

    def testSetup(self):

        # Sadly not flexible enough to test it independently
        # so the tests rely on and test the values pass in the 
        # production code
        app = MockApp()
        sphinxtogithub.setup(app)

        self.assertEqual(app.connections["build-finished"], sphinxtogithub.sphinx_extension)
        self.assertEqual(len(app.connections), 1)

        self.assertEqual(app.config_values["sphinx_to_github"],(True, ''))
        self.assertEqual(app.config_values["sphinx_to_github_verbose"],(True, ''))
        self.assertEqual(app.config_values["sphinx_to_github_encoding"],('utf-8', ''))
        self.assertEqual(len(app.config_values),3)


def testSuite():
    suite = unittest.TestSuite()

    suite.addTest(TestSetup("testSetup"))

    return suite

