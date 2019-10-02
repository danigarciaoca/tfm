import numpy as np


class BufferCircular:
    index = 0

    def __init__(self, numRow, numCol):
        self.maxIndex = numRow
        self.data = np.zeros((numRow, numCol))

    def add(self, train_set):

        self.data[self.index, :] = np.asarray(train_set)
        if self.index == self.maxIndex - 1:
            self.index = 0
        else:
            self.index += 1

    def restore_buffer(self):
        self.data = np.zeros(self.data.shape)

# Example:
# a = BufferCircular(6,3)
# a.add((1,2,3))
