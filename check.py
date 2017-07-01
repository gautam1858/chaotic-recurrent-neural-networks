import re
import numpy as np
import six
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile
from tensorflow.contrib import learn  

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)

def tokenizer(iterator):
  for value in iterator:
    yield list(value)

class MyVocabularyProcessor(learn.preprocessing.VocabularyProcessor):
    def __init__(self,
               max_document_length,
               min_frequency=0,
               vocabulary=None,
               tokenizer_fn=tokenizer):
        super().__init__(max_document_length,min_frequency,vocabulary,tokenizer_fn)


    def transform(self, raw_documents):
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids
