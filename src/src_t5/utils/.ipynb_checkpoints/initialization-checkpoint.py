import torch
import random
import numpy as np
import os
import sys
from datetime import date, datetime
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import OrderedDict
import torch.nn as nn
import json
import gzip
import time


def random_initialization(model, tokenizer):
    ids = []
    for x in range(30000):
        tokenized_ids = tokenizer.encode(str(x))
        if 3 in tokenized_ids:
            tokenized_ids.remove(3)
        if 1 in tokenized_ids:
            tokenized_ids.remove(1)
        ids += tokenized_ids
    ids = list(set(ids))
    for index in ids:
        model.shared.weight.data[index] = nn.init.normal_(
            model.shared.weight.data[index], 0, 1.0
        )

    return model
