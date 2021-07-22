import numpy as np
import pandas as pd
from itertools import combinations

# Generate sales
sales = pd.DataFrame({
    'date':np.repeat(['2021-01-01', '2021-01-02', '2021-01-03'], 4),
    'variant':np.tile(['small', 'medium', 'large', 'xl'], 3),
    'sales':[15,20,23,11, 73,99,114,105, 2,21,20,15],
    'depleted':[False, False, False, False,  False, False, False, False,  True, False, False, False]
})

# Save
sales.to_csv('sales.csv', index=False)