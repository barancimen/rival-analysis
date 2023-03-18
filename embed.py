import os
import openai
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from dataclasses import dataclass
from tenacity import retry, wait_random_exponential, stop_after_attempt

load_dotenv()
openai.api_key = os.environ['OPENAI_KEY']

@dataclass
class Embeddings:
    username: str
    raw: pd.DataFrame
    
    def load_embeddings(self, refresh = False):
        try:
            assert not refresh
            saved = np.load(f'{self.username}.npz')
            self.embeddings = pd.DataFrame(saved['emb'], index = saved['idx'])
        except:
            self.embeddings = pd.DataFrame()
            self.save_embeddings()
    
    def save_embeddings(self):
        self.embeddings.dropna(inplace = True)
        np.savez(self.username,
            idx = self.embeddings.index.values,
            emb = self.embeddings.values
        )
    
    def read(self, tweet):
        idx = self.raw.reset_index().set_index('text').id.loc[tweet]
        return self.embeddings.loc[idx].values
    
    @retry(
        wait = wait_random_exponential(min = 1, max = 20), 
        stop = stop_after_attempt(6)
    )
    def embed(self, tweets):
        return np.stack(pd.DataFrame(openai.Embedding.create(
            input = tweets,
            model = 'text-embedding-ada-002'
        ).data).set_index('index').sort_index().embedding.values)
    
    def run(self, batch_size = 200, checkpoint = 5):
        """
        For larger jobs, see:
        https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
        """
        self.load_embeddings()
        
        new = self.raw.loc[self.raw.index.difference(self.embeddings.index)]
        batches = np.array_split(new, new.shape[0] // batch_size + 1)
        
        for j, batch in enumerate(batches):
            self.embeddings = pd.concat([
                self.embeddings,
                pd.DataFrame(self.embed(list(batch.text)), index = batch.index)
            ])
            if (j + 1) % checkpoint == 0:
                self.save_embeddings()
        
        self.embeddings = self.embeddings.loc[self.raw.index]
        self.save_embeddings()