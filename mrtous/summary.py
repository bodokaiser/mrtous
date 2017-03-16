import json
import os

class SummaryWriter(object):

    def __init__(self, dirname, filename):
        self.format = os.path.splitext(filename)[1]
        assert self.format == '.json', 'SummaryWriter only supports JSON'

        self.file = None
        self.dirname = dirname
        self.filename = filename

    def open(self):
        self.path = os.path.join(self.dirname, self.filename)

        if os.path.exists(self.path):
            os.remove(self.path)

        self.file = open(self.path, 'w')

    def write(self, **event):
        if self.file is None:
            self.open()
        self.file.write(json.dumps(event))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()