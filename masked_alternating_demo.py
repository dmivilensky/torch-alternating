#!/usr/bin/env python3
# Copyright 2021 (c) Dmitry Pasechnyuk--Vilensky

import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from itertools import chain
from enum import Enum, auto
from alternating import masked
from alternating.masked import AlternatingEnvelopeAccelerated, AlternatingEnvelopeStandard, SwitchingStrategy

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

HIDDEN_SIZE = 50
EPOCHS = 5000
BATCHES_PER_EPOCH = 1


class AlternatingAlgorithm(Enum):
    STANDARD = auto()
    ACCELERATED = auto()


ALTERNATING_ENABLED = True
ALTERNATING_PERIOD = 200
ALTERNATING_ALGORITHM = AlternatingAlgorithm.ACCELERATED
ALTERNATING_SWITCHING = SwitchingStrategy.BEST_CONVEX

LOG_PERIOD = ALTERNATING_PERIOD * 5


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3 = nn.Linear(HIDDEN_SIZE, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

loss_fn = nn.CrossEntropyLoss()


def closure(backward=True):
    model.zero_grad()
    value = 0.0

    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    value += loss.item()
    if backward:
        loss.backward()

    return value


model = Model(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if ALTERNATING_ALGORITHM == AlternatingAlgorithm.STANDARD:
    alternation = AlternatingEnvelopeStandard(
        model.parameters(),
        period=ALTERNATING_PERIOD * BATCHES_PER_EPOCH
    )
elif ALTERNATING_ALGORITHM == AlternatingAlgorithm.ACCELERATED:
    alternation = AlternatingEnvelopeAccelerated(
        model.parameters(),
        closure=closure,
        switching_strategy=ALTERNATING_SWITCHING,
        period=ALTERNATING_PERIOD * BATCHES_PER_EPOCH
    )

mask_size = alternation.mask_size()
masks_number = 3
indices = np.random.randint(masks_number, size=mask_size)
masks = []
for i in range(masks_number):
    masks.append((indices == i).astype(int))
alternation.set_masks(masks)

for epoch in range(EPOCHS):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    if (epoch + 1) % LOG_PERIOD == 0:
        print("epoch:", epoch + 1)
        print("loss:", loss.item())

    model.zero_grad()
    loss.backward()
    alternation.step(closure=closure)
    optimizer.step()

    if (epoch + 1) % LOG_PERIOD == 0:
        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) ==
                       y_test).type(torch.FloatTensor)

            print("accuracy:", correct.mean().item())
            print()
