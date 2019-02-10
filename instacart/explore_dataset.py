"""
Work in progress
"""

import plotly.plotly
import plotly.graph_objs
import pandas
import numpy

graph_out = "../charts/"

aisles_pd = pandas.read_csv("data/aisles.csv")
departments_pd = pandas.read_csv("data/departments.csv")
order_products__prior_pd = pandas.read_csv("data/order_products__prior.csv")
order_products__train_pd = pandas.read_csv("data/order_products__train.csv")
orders_pd = pandas.read_csv("data/orders.csv")
products_pd = pandas.read_csv("data/products.csv")

"""
orders_pd → order_hour_of_day
order_products__prior_pd → add_to_cart_order | reordered
"""
orders_pd.info()
orders_pd.head()
order_products__prior_pd.info()
order_products__prior_pd.head()

rows_count = orders_pd.eval_set.value_counts()
trace = plotly.graph_objs.Scatter(x = rows_count.index, y = rows_count.values)
plot_data = [trace]
figure = plotly.graph_objs.Figure(data = plot_data)
plotly.plotly.offline.plot(figure, filename = graph_out + "0" + '.html', auto_open = False)

"""How many customers are there?"""
number = lambda x : len(numpy.unique(x))
customer_no = orders_pd.groupby("eval_set")["user_id"].aggregate(number)
print(customer_no)

"""What is the distribution of orders by customer?"""


"""What is the frequency of orders by the day of week?"""


"""What is the distribution of orders by the day's hour?"""


"""How would two of above look on a heatmap?"""


"""What is the frequency distribution by days from prior order?
How many days are passing in a row between two orders?"""


"""How many products are bought in an order?"""
