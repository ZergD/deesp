from unittest import TestCase
from deesp import text


class TestJoke(TestCase):
    def test_is_string(self):
        s = text.joke()
        self.assertTrue(isinstance(s, str))
