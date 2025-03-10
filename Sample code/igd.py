import numpy as np
from pymoo.indicators.gd import GD

ind = GD(base_line)             # Line used as a standard for comparision
print(ind(compared_line))       # Line that we are doing