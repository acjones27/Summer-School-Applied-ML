# <codecell> importing packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# <codecell> printing
print("Hi, just testing")
print("Goodbye")


# <codecell> plotting
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
t = np.linspace(0, 20, 500)

plt.plot(t, np.sin(t))
plt.show()

# <codecell> table
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None


iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

df1 = pd.read_csv(iris_url)

df1
