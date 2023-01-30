import subprocess
import pandas as pd

from datetime import timedelta
from dataclasses import dataclass

@dataclass
class Tweets:
    username: str
    
    def __post_init__(self):
        self.load_data()
        
    def load_data(self, refresh = False):
        try:
            assert not refresh
            self.data = pd.read_pickle(self.username)
        except:
            self.data = pd.DataFrame()
            self.save_data()
    
    def save_data(self):
        self.data.to_pickle(self.username)
    
    def query(self, params = {}, seconds = 60):
        subprocess.call(
            ' '.join([
                f'cd twint-zero && timeout {seconds}', 
                f'go run main.go -Query "from:{self.username}'
            ] + [
                f'{k}:{v}' for k,v in params.items()
            ]) + '" > ../temp', shell = True
        )
    
    def load_temp(self):
        temp = pd.read_csv('temp', sep = '\t', names = ['id', 'date', 'handle', 'text'])
        temp = temp[temp['handle'] == f'@{self.username}'].drop('handle', axis = 1)
        temp['date'] = pd.to_datetime(temp['date'], format = '%b %d, %Y · %I:%M %p %Z')

        return temp.set_index('id')
                                      
    def fill(self, forward = True, date = None):        
        while True:
            size = self.data.shape[0]
            try:
                dates = self.data.date
                self.query({
                    'since': date if date else dates.max().strftime('%Y-%m-%d')
                } if forward else {
                    'until': date if date else (dates.min() + timedelta(days = 1)).strftime('%Y-%m-%d')
                })
            except:
                self.query()
            
            temp = self.load_temp()
            self.data = pd.concat([self.data, temp]).groupby('id').first() if size > 0 else temp
            self.save_data()
            
            if self.data.shape[0] == size:
                break
    
    def run(self, backward = False):
        if backward:
            self.fill(forward = False)
        self.fill()