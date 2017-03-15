import json
import os

class SummaryWriter(object):

    def __init__(self, filename):
        dirname = os.path.dirname(filename)

        if os.path.exists(dirname):
            if os.path.exists(filename):
                os.remove(filename)
        else:
            os.makedirs(dirname)

        self.file = open(filename, 'w')
        self.format = os.path.splitext(filename)[1]

    def write(self, **event):
        if self.format != '.json':
            raise ValueError('SummaryWriter only support JSON')
        self.file.write(json.dumps(event))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()