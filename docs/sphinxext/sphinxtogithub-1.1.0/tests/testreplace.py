# -*- coding: utf-8 -*-

from unittest import TestCase

import sphinxtogithub

class TestReplace(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_unicode_replace(self):

        print u"this is a test ✓".replace( "this", "that" )


