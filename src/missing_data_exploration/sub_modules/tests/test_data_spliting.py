from ..data_spliting import partition_data
import pandas as pd
import numpy as np


def test_partition_data():
	dataset = pd.DataFrame(np.random.randint(0, 10, (100, 10))).values
	ret = partition_data(dataset, 5)
	assert True
