"""
Work in progress
"""

import plotly.plotly
import plotly.graph_objs
import h2o                  #https://www.h2o.ai
import pandas
import numpy

graph_out = "../charts/"

#@TODO Check using numpy types
""""""
aisles_pd = pandas.read_csv("data/aisles.csv")
departments_pd = pandas.read_csv("data/departments.csv")
order_products__prior_pd = pandas.read_csv("data/order_products__prior.csv")
order_products__train_pd = pandas.read_csv("data/order_products__train.csv")
orders_pd = pandas.read_csv("data/orders.csv")
products_pd = pandas.read_csv("data/products.csv")

#@TODO DataFrame methods' parameters need to be commented
""""""
building_products = pandas.DataFrame()
building_products['orders'] = order_products__prior_pd.groupby(order_products__prior_pd.product_id).size()
building_products['reorders'] = order_products__prior_pd['reordered'].groupby(order_products__prior_pd.product_id).sum()
products_pd = products_pd.join(building_products, on = 'product_id')
products_pd.set_index('product_id', drop = False, inplace = True)

del building_products
products_pd.head()

""""""
orders_pd.set_index('order_id', drop = False, inplace = True)
order_products__prior_pd = order_products__prior_pd.join(orders_pd, on = 'order_id', rsuffix='_')  #@TODO exactly why suffix?
order_products__prior_pd.drop('order_id_', inplace = True, axis = 1)
order_products__prior_pd.head()

""""""
building_users = pandas.DataFrame()
building_users['avg_order_delay'] = orders_pd.groupby('user_id')['days_since_prior_order'].mean()
building_users['orders_no'] = orders_pd.groupby('user_id').size()
users_pd = pandas.DataFrame()
users_pd['total_items'] = order_products__prior_pd.groupby('user_id').size()
users_pd['all_products'] = order_products__prior_pd.groupby('user_id')['product_id'].apply(set)
users_pd['total_distinct_items'] = (users_pd.all_products.map(len))

users_pd = users_pd.join(building_users)
users_pd['avg_basket'] = (users_pd.total_items / users_pd.orders_no)
del building_users
users_pd.head()

#@TODO exactly why should i harcode any number here?
""""""
order_products__prior_pd['user_product'] = order_products__prior_pd.product_id + order_products__prior_pd.user_id * 100000
order_products__prior_pd.head()

""""""
d = dict()
for row in order_products__prior_pd.itertuples():
    z = row.user_product
    if z not in d:
        d[z] = (1,
                (row.order_number, row.order_id),
                row.add_to_cart_order)
    else:
        d[z] = (d[z][0] + 1,
                max(d[z][1], (row.order_number, row.order_id)),
                d[z][2] + row.add_to_cart_order)

user_product = pandas.DataFrame.from_dict(d, orient = 'index')
del d

""""""
user_product.columns = ['orders_no', 'last_order_id', 'sum_pos_in_cart']
user_product.last_order_id = user_product.last_order_id.map(lambda x: x[1])
user_product.head()

del order_products__prior_pd


