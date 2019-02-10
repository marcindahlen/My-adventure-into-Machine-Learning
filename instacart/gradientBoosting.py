"""
Work in progress
"""

import plotly.plotly
import plotly.graph_objs
import h2o                  #https://www.h2o.ai
import pandas
import numpy

graph_out = "../charts/"

"""Check using numpy types"""
aisles_pd = pandas.read_csv("data/aisles.csv")
departments_pd = pandas.read_csv("data/departments.csv")
order_products__prior_pd = pandas.read_csv("data/order_products__prior.csv")
order_products__train_pd = pandas.read_csv("data/order_products__train.csv")
orders_pd = pandas.read_csv("data/orders.csv")
products_pd = pandas.read_csv("data/products.csv")

""""""
building_products = pandas.DataFrame()
building_products['orders'] = order_products__prior_pd.groupby(order_products__prior_pd.product_id).size()
building_products['reorders'] = order_products__prior_pd['reordered'].groupby(order_products__prior_pd.product_id).sum()


