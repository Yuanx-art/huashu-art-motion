import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('mouth', Path(__file__).parents[1] / 'scripts/align_mandarin_mouth.py')
mouth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mouth)


class MouthAlignmentTests(unittest.TestCase):
    def test_repeated_character_requires_intervening_blank(self):
        # Two identical letters must occupy distinct emissions, with a blank.
        probabilities = np.array([[.01,.99],[.99,.01],[.01,.99]])
        spans = mouth.force_align(np.log(probabilities), [1,1], 0)
        self.assertEqual(spans, [(0,1),(2,3)])

    def test_long_transcript_backtrace_uses_python_integer(self):
        # More than 128 states exposed NumPy int8 subtraction overflow.
        tokens=[1,2]*100
        probabilities=np.full((200,3),.01)
        for i,token in enumerate(tokens): probabilities[i,token]=.98
        spans=mouth.force_align(np.log(probabilities),tokens,0)
        self.assertEqual(len(spans),200)
        self.assertEqual(spans[-1],(199,200))

    def test_impossible_alignment_is_rejected(self):
        with self.assertRaises(ValueError):
            mouth.force_align(np.log(np.array([[.01,.99],[.01,.99]])), [1,1], 0)

    def test_labial_and_labiodental_are_distinct(self):
        self.assertEqual(mouth.ipa_for('百'), ['p','a','i'])
        self.assertEqual(mouth.ipa_for('副'), ['f','u'])
        self.assertEqual(mouth.phone_viseme('p'), 'MBP')
        self.assertEqual(mouth.phone_viseme('f'), 'FV')
        self.assertEqual(mouth.phone_viseme('u'), 'U')
        self.assertIsNone(mouth.phone_viseme('t'))


if __name__ == '__main__':
    unittest.main()
