import matplotlib.pyplot as plt
x = ["A", "B", "C", "D"]
y = [1, 2, 3, 4]
plt.barh(x, y)

for index, value in enumerate(y):
    plt.text(value, index, str(value))
plt.show()