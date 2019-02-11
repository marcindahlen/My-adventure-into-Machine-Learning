"""
Work in progress
"""

import plotly.plotly
import plotly.graph_objs    #plotly saves charts as interactive html pages
import h2o                  #https://www.h2o.ai
import pandas
import numpy

graph_out = "../charts/"

#@TODO Check using numpy types
"""
Since the data is already cleaned up and prepared for calculations
I simply load the csv files to pandas data frames.
Cleaned up data means there are no meaningless nulls,
and no unnecessary columns.
"""
aisles_pd = pandas.read_csv("data/aisles.csv")                                  #2.5KB
print("aisles loaded")
departments_pd = pandas.read_csv("data/departments.csv")                        #420B
print("departments loaded")
order_products__prior_pd = pandas.read_csv("data/order_products__prior.csv")    #1GB
print("products__prior loaded")
order_products__train_pd = pandas.read_csv("data/order_products__train.csv")    #42MB
print("products__train loaded")
orders_pd = pandas.read_csv("data/orders.csv")                                  #183MB
print("orders loaded")
products_pd = pandas.read_csv("data/products.csv")                              #1.5MB
print("products loaded")

"""
I need to build up the data frame which will be passed 
to H2O machine learning model.
I will start with adding two new columns to products' data frame.
Orders column will contain group size by product type occurrence,
while reorders will contain cumulative sums of reorders by given product.
"""
building_products = pandas.DataFrame()
building_products['orders'] = order_products__prior_pd.groupby(order_products__prior_pd.product_id).size() #Compute group sizes
building_products['reorders'] = order_products__prior_pd['reordered'].groupby(order_products__prior_pd.product_id).sum() #consider only reordered column
products_pd = products_pd.join(building_products, on = 'product_id')
products_pd.set_index('product_id', drop = False, inplace = True)

del building_products
products_pd.info()
print(products_pd.head())
print()

"""
Now, I will combine two data frames
by adding prepared orders_pd to order_products__prior_pd.
Updated data frame will have columns as following:
order_id; product_id; add_to_cart_order; reordered; user_id; eval_set; order_number; order_dow; order_hour_of_day; days_since_prior_order;
"""
orders_pd.set_index('order_id', drop = False, inplace = True) #from now, order_id is the new index
order_products__prior_pd = order_products__prior_pd.join(orders_pd, on = 'order_id', rsuffix='_')  #suffix added to not overwrite the new indexing
order_products__prior_pd.drop('order_id_', inplace = True, axis = 1)
order_products__prior_pd.info()
print(order_products__prior_pd.head())
print()

"""
I will create a new data frame with following colums:
total_items; all_products; total_distinct_items; avg_order_delay; orders_no; avg_basket;

"""
building_users = pandas.DataFrame()
building_users['avg_order_delay'] = orders_pd.groupby('user_id')['days_since_prior_order'].mean() #Compute mean of groups, excluding missing values
building_users['orders_no'] = orders_pd.groupby('user_id').size()
users_pd = pandas.DataFrame()
users_pd['total_items'] = order_products__prior_pd.groupby('user_id').size()
users_pd['all_products'] = order_products__prior_pd.groupby('user_id')['product_id'].apply(set)
users_pd['total_distinct_items'] = (users_pd.all_products.map(len))

users_pd = users_pd.join(building_users)
users_pd['avg_basket'] = (users_pd.total_items / users_pd.orders_no) #Mean number of products in a basket by user.
del building_users
users_pd.info()
print(users_pd.head())
print()

#@TODO exactly why should i hardcode any number here?
""""""
order_products__prior_pd['user_product'] = order_products__prior_pd.product_id + order_products__prior_pd.user_id * 100000
order_products__prior_pd.info()
print(order_products__prior_pd.head())
print()

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
user_product.info()
print(user_product.head())
print()

del order_products__prior_pd

""""""
test_orders = orders_pd[orders_pd.eval_set == 'test']
train_orders = orders_pd[orders_pd.eval_set == 'train']

train_orders.set_index(['order_id', 'product_id'], inplace=True, drop=False) #@TODO KeyError: 'product_id'


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

    return (dataFrame, labels)

#@TODO features to be used
""""""
dataFrame_training, labels = features(train_orders, labels_given = True)
dataFrame_training.head()

features_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders', 'user_average_basket',
       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
       'aisle_id', 'department_id', 'product_orders', 'product_reorders',
       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
       'UP_delta_hour_vs_last']

#@TODO set up h2o
""""""
h2o.init()
h2o_frame = h2o.H2OFrame(dataFrame_training)

#@TODO train the gbm model
""""""

#@TODO test predictions
""""""