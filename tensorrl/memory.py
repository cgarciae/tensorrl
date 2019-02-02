import numpy as np

class ReplayMemory:

    def __init__(self, max_size):

        self.max_size = max_size
        self.memory = None
        self.size = 0
        self.i = -1

    def append(self, **row):

        if self.memory is not None:
            assert set(row) == set(self.memory), "append keys not identical to memory"

        if self.memory is None:
            self.memory = {}

            for name, column in row.items():
                column = np.array(column).squeeze()
                self.memory[name] = np.tile(
                    column,
                    (self.max_size,) + (1,) * column.ndim,
                )
        else:
            for name in row:
                elem = row[name]
                elem = np.array(elem).squeeze()
                i = (self.i + 1) % self.max_size
                self.memory[name][i] = elem


        if self.size < self.max_size:
            self.size += 1

        self.i = (self.i + 1) % self.max_size

    
    def sample(self, batch_size):
        assert self.size >= batch_size, "Not enough elements to sample from."

        indexes = np.random.choice(self.size, batch_size)

        return {
            name: column[indexes]
            for name, column in self.memory.items()
        }

    def reset(self):
        self.size = 0
        self.i = -1

    def __repr__(self):

        if self.memory is None:
            content = "empty"
        else:
            content = ", ".join([
                f"{name}: {column.shape} [{column.dtype}]"
                for name, column in self.memory.items()
            ])

        return f"ReplayMemory({content})"
        
        
