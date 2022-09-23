"""
Work in progress
"""

import plotly
import plotly.graph_objs
import pandas
import numpy

graph_out = "../charts/"

aisles_pd = pandas.read_csv("data/aisles.csv")
print("aisles loaded")
departments_pd = pandas.read_csv("data/departments.csv")
print("departments loaded")
order_products__prior_pd = pandas.read_csv("data/order_products__prior.csv")
print("products__prior loaded")
order_products__train_pd = pandas.read_csv("data/order_products__train.csv")
print("products__train loaded")
orders_pd = pandas.read_csv("data/orders.csv")
print("orders loaded")
products_pd = pandas.read_csv("data/products.csv")
print("products loaded")

"""
orders_pd → order_hour_of_day
order_products__prior_pd → add_to_cart_order | reordered
"""
print("AISLES:")
aisles_pd.info()
print(aisles_pd.head())
print("DEPARTMENTS")
departments_pd.info()
print(departments_pd.head())
print("PRIOR PRODUCTS")
order_products__prior_pd.info()
print(order_products__prior_pd.head())
print("TRAINING PRODUCTS")
order_products__train_pd.info()
print(order_products__train_pd.head())
print("ORDERS")
orders_pd.info()
print(orders_pd.head())
print("PRODUCTS")
products_pd.info()
print(products_pd.head())

rows_count = orders_pd.eval_set.value_counts()
trace = plotly.graph_objs.Scatter(x = rows_count.index, y = rows_count.values)
plot_data = [trace]
figure = plotly.graph_objs.Figure(data = plot_data)
plotly.offline.plot(figure, filename = graph_out + "0" + '.html', auto_open = False)

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
