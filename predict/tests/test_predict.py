import unittest
from unittest.mock import MagicMock
import tempfile
import json
import argparse
import os
import time
from collections import OrderedDict

from tensorflow.keras.models import load_model
from numpy import argsort

from predict.predict.run import TextPredictionModel

class TestTrain(unittest.TestCase):
    def test_from_artefacts(self):
        artefacts_path = './resources/testingModel'
        model = load_model(artefacts_path)
        params = json.load(open(artefacts_path + '/params.json'))
        labels_to_index = json.load(open(artefacts_path + '/labels_index.json'))

        modelFromScratch = TextPredictionModel(model,params, labels_to_index )
        modelFromArtifact = TextPredictionModel.from_artefacts(artefacts_path)

        self.assertEqual(modelFromScratch, modelFromArtifact)

    def test_predict(self):
        textToPredict = [
            "Is it possible to execute the procedure of a function in the scope of the caller?",
            "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
            "Is it possible to execute the procedure of a function in the scope of the caller?",
            "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
            "Is it possible to execute the procedure of a function in the scope of the caller?",
            "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
            "Is it possible to execute the procedure of a function in the scope of the caller?",
            "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
            "Is it possible to execute the procedure of a function in the scope of the caller?",
            "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        ]
        tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]
        artefacts_path = './resources/testingModel'
        model = TextPredictionModel.from_artefacts(artefacts_path)
        prediction = model.predict(textToPredict, top_k=10)
        self.assertEqual(prediction, tags)