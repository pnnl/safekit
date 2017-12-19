
from sphinxtogithub.tests import MockStream

import sphinxtogithub
import unittest
import os
import shutil

root = "test_path"
dirs = ["dir1", "dir2", "dir_", "d_ir", "_static", "_source"]
files = ["file1.html", "nothtml.txt", "file2.html", "javascript.js"]

def mock_is_dir(path):

    directories = [ os.path.join(root, dir_) for dir_ in dirs ]

    return path in directories

def mock_list_dir(path):

    contents = []
    contents.extend(dirs)
    contents.extend(files)
    return contents

def mock_walk(path):

    yield path, dirs, files

class MockHandlerFactory(object):

    def create_file_handler(self, name, replacers, opener):

        return sphinxtogithub.FileHandler(name, replacers, opener)

    def create_dir_handler(self, name, root, renamer):

        return sphinxtogithub.DirectoryHandler(name, root, renamer)


class TestLayoutFactory(unittest.TestCase):

    def setUp(self):

        verbose = True
        force = False
        stream = MockStream()
        dir_helper = sphinxtogithub.DirHelper(
            mock_is_dir,
            mock_list_dir,
            mock_walk,
            shutil.rmtree
            )

        file_helper = sphinxtogithub.FileSystemHelper(
            open,
            os.path.join,
            shutil.move,
            os.path.exists
            )

        operations_factory = sphinxtogithub.OperationsFactory()
        handler_factory = MockHandlerFactory()

        self.layoutfactory = sphinxtogithub.LayoutFactory(
                operations_factory,
                handler_factory,
                file_helper,
                dir_helper,
                verbose,
                stream,
                force
                )

    def tearDown(self):
        
        self.layoutfactory = None

    def testUnderscoreCheck(self):

        func = self.layoutfactory.is_underscore_dir
        self.assert_(func(root, "_static"))
        self.assert_(not func(root, "dir_"))
        self.assert_(not func(root, "d_ir"))
        self.assert_(not func(root, "dir1"))


    def testCreateLayout(self):

        layout = self.layoutfactory.create_layout(root)

        dh = layout.directory_handlers
        self.assertEqual(dh[0].name, "_static")
        self.assertEqual(dh[1].name, "_source")
        self.assertEqual(len(dh), 2)
        
        fh = layout.file_handlers
        self.assertEqual(fh[0].name, os.path.join(root,"file1.html"))
        self.assertEqual(fh[1].name, os.path.join(root,"file2.html"))
        self.assertEqual(fh[2].name, os.path.join(root,"javascript.js"))
        self.assertEqual(len(fh), 3)




def testSuite():
    suite = unittest.TestSuite()

    suite.addTest(TestLayoutFactory("testUnderscoreCheck"))
    suite.addTest(TestLayoutFactory("testCreateLayout"))

    return suite

