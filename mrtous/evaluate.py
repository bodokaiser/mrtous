from network import Simple
from dataset import MNIBITE
from transform import RegionCrop

from torch.nn import MSELoss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

model = Simple()

loader = DataLoader(MNIBITE('mnibite', 13))

criterion = MSELoss(size_average=False)
optimizer = Adam(model.parameters())

for epoch in range(1, 10):
    epoch_loss = 0

    for step, (mr, us) in enumerate(loader):
        inputs = Variable(mr).unsqueeze(1)
        target = Variable(us).unsqueeze(1)

        optimizer.zero_grad()
        loss = criterion(model(inputs), target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data[0]

    print(f'epoch: {epoch}, loss: {epoch_loss}')