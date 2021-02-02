### import packages ###
import os
import re

import pandas as pd
from .model_hf import ModelHF
from .model_sumy import ModelSumy

class Model():
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

        self._get_model()

    def _get_model(self):
        """
        load model based on model to use
        
        Args:
            None
        Returns:
            None
        """
        if self.model_name == 'ModelHF':
            self.model = ModelHF()
        elif self.model_name == 'ModelSumy':
            self.model = ModelSumy()

    def get_data(self, data_path, samples, seed):
        """
        get dataset
        
        Args:
            None
        Returns:
            None
        """
        data = pd.read_csv(data_path).sample(n=samples, random_state=seed)
        return data

    def preprocess(self, text):
        """
        preprocess text

        Args:
            text: input text to preprocess
        Returns:
            output_text: preprocessed text
        """
        ## regex method - remove '\n'
        cleantext = re.sub(r"\\n", " ", text)
        ## remove '\BA'
        cleantext = re.sub(r"\\BA", " ", cleantext)
        ## remove '\'
        cleantext = re.sub(r"\\", " ", cleantext)
        ## remove parenthesis
        cleantext = re.sub(r'\([^)]*\)', '', cleantext)
        ## removing double spaces with single space
        cleantext = re.sub('\s\s+', " ", cleantext)
        # remove 'b''
        cleantext = re.sub(r"(b')","",cleantext)
        ## substitute % with ' percent'
        cleantext = re.sub(r"[%]", " percent",cleantext)
        ## substitute $ with 'USD'
        cleantext = re.sub(r"[$]", "USD",cleantext)
        ## remove '\'
        cleantext = re.sub(r"[\\]", "", cleantext)
        ## remove '-'
        cleantext = re.sub(r"(-')", " ", cleantext)
        ## remove '""'
        cleantext = re.sub(r"[\"]", "", cleantext)
        ## remove all b"
        cleantext = cleantext.strip('b"')

        ## remove xc2
        cleantext = re.sub(r"xc2", "", cleantext)
        # remove xa359m
        cleantext = re.sub(r"xa359m", "", cleantext)
        # remove xa35.7n 
        cleantext = re.sub(r"xa35.7n ", "", cleantext)
        # remove xa3160m
        cleantext = re.sub(r"xa3160m", "", cleantext)
        # remove xa35.7bn
        cleantext = re.sub(r"xa35.7bn ", "", cleantext)
        # remove xa3125m
        cleantext = re.sub(r"xa3125m", "", cleantext)
        # remove xa375m
        cleantext = re.sub(r"xa375m", "", cleantext)
        # remove xa3106m
        cleantext = re.sub(r"xa3106m", "", cleantext)
        # remove xa31.97bn
        cleantext = re.sub(r"xa31.97bn", "", cleantext)
        # remove xa3250m
        cleantext = re.sub(r"xa3250m", "", cleantext)
        # remove xa36
        cleantext = re.sub(r"xa36", "", cleantext)
        # remove xa310
        cleantext = re.sub(r"xa310", "", cleantext)
        # remove xa34
        cleantext = re.sub(r"xa34", "", cleantext)
        # remove xa32.50
        cleantext = re.sub(r"xa32.50", "", cleantext)
        
        return cleantext
        