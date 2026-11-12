import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class Config:
    def __init__(self, task=None):
        self.data_root = str(ROOT / 'data')
        self.model_name = os.environ.get('TOPPPI_ESM_PATH', str(ROOT / 'pretrained' / 'esm2_t33_650M_UR50D'))
        self.select_task(task or os.environ.get('TOPPPI_TASK', 'yeast'))

    def select_task(self, task):
        if task not in {'yeast', 'multispecies', 'multiclass', 'human_virus'}:
            raise ValueError(f'Unknown task: {task}')
        values = json.loads((ROOT / 'configs' / f'{task}.json').read_text(encoding='utf8'))
        self.__dict__.update(values)
        self.task = task

config = Config()
