import os
import sys


class Logger(object):
    def __init__(self, file_dir="./logs", file_name="default.out", stream=sys.stdout):
        self.terminal = stream
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        self.log = open(os.path.join(file_dir, file_name), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
