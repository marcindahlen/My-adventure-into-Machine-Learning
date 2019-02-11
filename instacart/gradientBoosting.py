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

""""""
test_orders = orders_pd[orders_pd.eval_set == 'test']
train_orders = orders_pd[orders_pd.eval_set == 'train']

train_orders.set_index(['order_id', 'product_id'], inplace=True, drop=False)


def features(selected_orders, labels_given=False):
    #@TODO explanations to be added
    """"""
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i = 0
    for row in selected_orders.itertuples():
        i += 1
        if i % 10000 == 0: print('order row', i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users_pd.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train_orders.index for product in user_products]

    dataFrame = pandas.DataFrame({'order_id': order_list, 'product_id': product_list}, dtype=numpy.int32)
    labels = numpy.array(labels, dtype=numpy.int8)
    del order_list
    del product_list

    #@TODO dataFrame to be processed
    """"""
    dataFrame['user_id'] = dataFrame.order_id.map(orders_pd.user_id)
    dataFrame['user_total_orders'] = dataFrame.user_id.map(users_pd.orders_no)
    dataFrame['user_total_items'] = dataFrame.user_id.map(users_pd.total_items)
    dataFrame['total_distinct_items'] = dataFrame.user_id.map(users_pd.total_distinct_items)
    dataFrame['user_average_days_between_orders'] = dataFrame.user_id.map(users_pd.avg_order_delay)
    dataFrame['user_average_basket'] =  dataFrame.user_id.map(users_pd.avg_basket)

    dataFrame['order_hour_of_day'] = dataFrame.order_id.map(orders_pd.order_hour_of_day)
    dataFrame['days_since_prior_order'] = dataFrame.order_id.map(orders_pd.days_since_prior_order)
    dataFrame['days_since_ratio'] = dataFrame.days_since_prior_order / dataFrame.user_average_days_between_orders

    dataFrame['aisle_id'] = dataFrame.product_id.map(products_pd.aisle_id)
    dataFrame['department_id'] = dataFrame.product_id.map(products_pd.department_id)
    dataFrame['product_orders'] = dataFrame.product_id.map(products_pd.orders)
    dataFrame['product_reorders'] = dataFrame.product_id.map(products_pd.reorders)
    dataFrame['product_reorder_rate'] = dataFrame.product_id.map(products_pd.reorder_rate)

    dataFrame['z'] = dataFrame.user_id * 100000 + dataFrame.product_id
    dataFrame.drop(['user_id'], axis=1, inplace=True)
    dataFrame['UP_orders'] = dataFrame.z.map(user_product.orders_no)
    dataFrame['UP_orders_ratio'] = (dataFrame.UP_orders / dataFrame.user_total_orders).astype(np.float32)
    dataFrame['UP_last_order_id'] = dataFrame.z.map(user_product.last_order_id)
    dataFrame['UP_average_pos_in_cart'] = (dataFrame.z.map(user_product.sum_pos_in_cart) / dataFrame.UP_orders)
    dataFrame['UP_reorder_rate'] = (dataFrame.UP_orders / dataFrame.user_total_orders)
    dataFrame['UP_orders_since_last'] = dataFrame.user_total_orders - dataFrame.UP_last_order_id.map(orders_pd.order_number)
    dataFrame['UP_delta_hour_vs_last'] = abs(dataFrame.order_hour_of_day - dataFrame.UP_last_order_id.map(orders_pd.order_hour_of_day)).map(lambda x: min(x, 24 - x))

    dataFrame.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)

#@TODO features to be used
""""""

#@TODO set up h2o
""""""

#@TODO train the gbm model
""""""

#@TODO test predictions
""""""