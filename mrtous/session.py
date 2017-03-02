import numpy as np

from torch.nn import MSELoss
from torch.optim import Adam
from torch.autograd import Variable

def is_empty(images):
    return np.any(images[1].numpy())

def train(model, loader, epochs, train_fn):
    criterion = MSELoss(size_average=False)
    optimizer = Adam(model.parameters())

    for epoch in range(1, epochs+1):
        total_loss = 0

        for step, (mr, us) in enumerate(filter(is_empty, loader)):
            inputs = Variable(mr).unsqueeze(1)
            targets = Variable(us).unsqueeze(1)
            results = model(inputs)

            optimizer.zero_grad()
            loss = criterion(results, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]

        train_fn(inputs, targets, results, epoch, total_loss)

def evaluate(model, loader, eval_fn):
    criterion = MSELoss(size_average=False)

    total_loss = 0

    for _, (mr, us) in enumerate(filter(is_empty, loader)):
        inputs = Variable(mr).unsqueeze(1)
        targets = Variable(us).unsqueeze(1)
        results = model(inputs)

        total_loss += criterion(results, targets).data[0]

    eval_fn(inputs, targets, results, total_loss)