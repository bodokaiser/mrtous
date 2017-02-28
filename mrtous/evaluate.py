import utils
import numpy as np

from network import Simple
from dataset import MNIBITE
from transform import RegionCrop

from torch import FloatTensor
from torch.nn import MSELoss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

model = Simple()

loader = DataLoader(MNIBITE('mnibite', 13, transform=RegionCrop()))

criterion = MSELoss(size_average=False)
optimizer = Adam(model.parameters())

def is_empty(images):
    # applied as filter to DataLoader instance
    # this will ensure no empty us images
    return np.any(images[1].numpy())

def show_result(loader, model):
    for _, (mr, us) in enumerate(filter(is_empty, loader)):
        inputs = Variable(mr).unsqueeze(1)
        target = Variable(us).unsqueeze(1)

        results = model(inputs)
        differs = target-results

        utils.show_images([
            inputs.data[0][0].numpy(), target.data[0][0].numpy(),
            results.data[0][0].numpy(), differs.data[0][0].numpy(),
        ], 2, 2)

for epoch in range(1, 31):
    epoch_loss = 0

    for step, (mr, us) in enumerate(filter(is_empty, loader)):
        inputs = Variable(mr).unsqueeze(1)
        target = Variable(us).unsqueeze(1)

        optimizer.zero_grad()
        loss = criterion(model(inputs), target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data[0]

    print(f'epoch: {epoch}, loss: {epoch_loss}')

show_result(loader, model)