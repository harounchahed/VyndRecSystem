'''
|**********************************************************************;
* Project           : Vynd Recommender System
*
* Program name      : recommender_system_hidden_password
*
* Author            : Haroun Chahed 
*
* Date created      : 20171211 
*
|**********************************************************************;
'''

# Packages
import pyodbc
import numpy as np
import pandas as pd
import webbrowser
import os
import matrix_factorization_utilities
import tabulate
import math
import ast
from itertools import chain
from sys import exit

# RC Weight Settings
expected_rating_weight = 0.1
shared_classification_weight = 0.25
shared_AddInfo_weight = 0.55
distance_weight = 0.1
rec_num = 5

# Functions
    # Simple distance function (assuming a flat surface)
def flat_distance(bx, by, rx, ry):
    sq1 = (rx-bx)**2
    sq2 = (ry-by)**2
    return np.sqrt(sq1 + sq2)

    # Function that counts number of common elements of two lists
def number_of_common_elements(list1, list2):
    return len([x for x in list1 if x in list2])


    # dataframe division functions that replaces inf and naan values with 0
def divide_df (x,d):
    return (x/d).replace([np.inf, -np.inf], np.nan).fillna(0)


    # Function for table weighted sorting according to 4 columns with 4 weights
    # the first three columns are sorted DESC the fourth one ASC
def df_weighted_sorter (df, col_1_name, col_2_name, col_3_name, col_4_name,
                        col_1_weight, col_2_weight, col_3_weight, col_4_weight):

        # calculating an SV (sorting value) for each row
        # by multiplying the weight of each cell relative to its column
        # by the given weight
        # then summing the resultant of the cells considered
    col1 = df[col_1_name]
    col2 = df[col_2_name]
    col3 = df[col_3_name]
    col4 = df[col_4_name]

    df = df.assign(SV=((divide_df(col1, col1.max()) * col_1_weight\
               + divide_df(col2, col2.max()) * col_2_weight\
               + divide_df(col3, col3.max()) * col_3_weight\
               + divide_df(col4.min(), col4) * col_4_weight).values))

        #sorting the table according to SV
    df = df.sort_values(by=['SV'], ascending=False)
    return df

# Connecting to base
# credentials are hidden 
username='-----------'
password='----------'
server ='-------------------'
database='--------------'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

# Creating a cursor
cursor = cnxn.cursor()

# Extracting the rates along with relevant info about user and business
cursor.execute('''SELECT Comments.UserId, Comments.BusinessId, Businesses.name, Businesses.Latitude, Businesses.Longitude,
               EnsembledClassification.SubCategory_name, Comments.rate
               FROM Comments
               /* Add Business name for each rate */
               INNER JOIN Businesses
               ON (Comments.BusinessId=Businesses.Id)
               /* Add a list of classification labels for each rate*/
               Inner JOIN (
                   SELECT B.[Id]
                         ,'[' + STUFF(
                            (   SELECT ',"' + C.[Label] + '"'
                            FROM [dbo].[BusinessClassification] AS BC, [dbo].[Classifications] AS C
                            /* Conditions on Businesses and Classifications */
                            WHERE B.[Id] = BC.[BusinessId]
                                AND C.[Id] = BC.[ClassificationId]
                                AND C.Type = 2
                                AND C.Valid=1
                                AND C.DeleteDate IS NULL
                            FOR xml path('')
                            ), 1, 1, '') + ']' as SubCategory_name
                    FROM [dbo].[Businesses] AS B
               ) EnsembledClassification
               ON Comments.BusinessId=EnsembledClassification.Id
               /* more conditions on businesses and comments*/
               WHERE (Businesses.Valid=1)
                AND (Businesses.DeleteDate IS NULL)
                AND (Comments.DeleteDate IS NULL)''')

Comments = cursor.fetchall()

# Extracting the additional Info along with businessId
cursor.execute(''' SELECT B.[Id],
                 '[' +STUFF(
                    (SELECT ',' + CONVERT(VARCHAR, [ValidInfoId])
                        FROM 
                            (Select DISTINCT IB.[ValidInfoId]
                            FROM [dbo].[InfosBusinesses] AS IB 
                            WHERE IB.[BusinessId] = B.[Id] AND [Value] = 1 AND [Validated] = 1) AS InfoIds
                        FOR xml path(''))
                        , 1, 1, '') + ']' AS valid_info_id
                FROM [dbo].[Businesses] AS B 
                WHERE B.[DeleteDate] IS NULL AND B.[Valid] = 1''')

AdditionalInfo = cursor.fetchall()

# creating pandas dataframes of ratings and businesses
UserId=[]
BusinessId=[]
BusinessName=[]
Classifications=[]
rate=[]
Latitude=[]
Longitude=[]

for row in Comments:
    UserId+=[int(row.UserId)]
    BusinessId+=[int(row.BusinessId)]
    BusinessName+=[row.name]
    Classifications+=[row.SubCategory_name]
    rate+=[int(row.rate)]
    Latitude+=[float(row.Latitude)]
    Longitude+=[float(row.Longitude)]

ratings=pd.DataFrame(
    {'UserId':UserId,
     'BusinessId': BusinessId,
     'rate':rate
    })

businesses_df=pd.DataFrame(
    {'BusinessId': BusinessId,
     'BusinessName': BusinessName,
     'Classification':Classifications,
     'Latitude': Latitude,
     'Longitude': Longitude
     })

# creating dataframe of additional info

BusinessId=[]
AddInfo=[]

for row in AdditionalInfo:
    BusinessId+=[int(row.Id)]
    AddInfo+=[row.valid_info_id]

AddInfo_df = pd.DataFrame(
    {'BusinessId': BusinessId,
     'AddInfo': AddInfo})

# Drop duplicated from businesses_df
businesses_df=businesses_df.drop_duplicates(subset=['BusinessId'], keep='first')

# Add columns for Addinfo
businesses_df = businesses_df.merge(AddInfo_df, on='BusinessId')

# Convert the running list of user ratings into a matrix
ratings_df = pd.pivot_table(ratings, index='UserId', columns='BusinessId', aggfunc=np.max)

##creating the preliminary recommendation system###################################
# Apply matrix factorization to find the latent features (the empty cells)
# U is the matrix userId * user attribute
# B is the matrix user attribute * businessID
# U*B = predicted_ratings is the matrix UserId * business ID (it is the same as ratings_df, except with expected ratings)
# each of these ratings will be on a scale from -5 to 5 (this will help later when calculating the predicted review)
# num_features is the number of latent attributes we are considering (they are latent because we don't know what they could be)
# (which is also the number of columns of U and the number of rows of B)
# (imagine that in U each user is rated for num_features attributes i.e., how much he/she cares about each attribute
# and that in B each business is rated for the same num_features attributes i.e., how good it is considering one feature at a time)
# we will take num_feature to be 5 initially, because that is the actualy number of features the users have rated
U, B = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df.as_matrix(),
                                                                    num_features=5,
                                                                    regularization_amount=0.1)

# Find all predicted ratings by multiplying the U by M
# In This matrix, each value is a row of U multiplied by a column of B
# i.e., for each attribute, the user's attribute value is multiplied by the business's atribute value, then all the products are summed up.
# If both attributes are positive (the user cares about this attribute and the business has a good rate for this attirbute)
# or both attibutes are negative (the user does not care about this attibure and the business scores badly on this attribute)
# then the product is positive (considering this attribute, this business is a good fit for this user)
# If the attributes have difference signes ((the user cares about this attribute and the business scores badly on it) or (the user does not care about this attribute and the business scores good on it))
# then the product is negative (considering this attribute this business is a bad fit for this user)
# Overall, the sum of the products should give us an overall estimation of how much would a user like a business (the numbers are scaled down to normal)
predicted_ratings = np.matmul(U, B)

# Creating a sorted list of user Ids of Users that have done at least one review
UniqueUserId = sorted(list(set(UserId)))

for user_id_to_search in UniqueUserId:
    # Sorting recommended_df #1: Add user ratings as a column in businesses_df to sort the table according to this column
    user_ratings = predicted_ratings[UniqueUserId.index(user_id_to_search)]
    businesses_df['rating'] = user_ratings

    # Sorting recommended_df #2: Add number of shared classifications as a column in recommended_df
    reviewed_businesses_df = ratings[ratings['UserId'] == user_id_to_search]
    reviewed_businesses_df = reviewed_businesses_df.merge(businesses_df, on='BusinessId')
    Classifications_as_string = reviewed_businesses_df['Classification'].str.cat(sep=',')
    classification_as_tuple_of_lists = ast.literal_eval(Classifications_as_string)
    Classification_as_list = []
    for classif in classification_as_tuple_of_lists:
        if isinstance(classif, list):
            Classification_as_list += classif
        else:
            Classification_as_list += [classif]

    shared_classifications = []
    for k in range(len(businesses_df)):
        Classification_as_list_k = ast.literal_eval(businesses_df['Classification'].iloc[k])
        shared_classifications += [
            number_of_common_elements(Classification_as_list,
                                      Classification_as_list_k)]

    businesses_df['shared_classifications'] = shared_classifications

    # Sorting Recommended_df #3: number of shared additional information labels
    AddInfo = []
    for k in reviewed_businesses_df['AddInfo']:
        if k is not None:
            AddInfo += ast.literal_eval(k)
    AddInfo = list(set(AddInfo))
    shared_AddInfo = []
    for k in range(len(businesses_df)):
        if businesses_df['AddInfo'].iloc[k] is None:
            shared_AddInfo += [0]
        else:
            AddInfo_k = ast.literal_eval(businesses_df['AddInfo'].iloc[k])
            shared_AddInfo += [number_of_common_elements(AddInfo, AddInfo_k)]
    businesses_df['shared_AddInfo'] = shared_AddInfo

    # sorting recommended_df #4: minimal distance from reviewed businesses
    distance = []
    for k in range(len(businesses_df)):
        distance += [flat_distance(businesses_df['Latitude'].iloc[k],
                              businesses_df['Longitude'].iloc[k],
                              reviewed_businesses_df['Latitude'],
                              reviewed_businesses_df['Longitude']).min()]

    businesses_df['distance'] = distance

    # Removing businesses that have already been reviewed
    already_reviewed = reviewed_businesses_df['BusinessId']
    recommended_df = businesses_df[businesses_df.BusinessId.isin(already_reviewed) == False]

    # Removing businesses that have an invalid location (0,0)
    invalid_location = businesses_df.loc[(businesses_df['Latitude'] == 0) & (businesses_df['Longitude'] == 0)]
    recommended_df = recommended_df[recommended_df.BusinessId.isin(invalid_location.BusinessId) == False]

    # Sort recommended_df according to different columns with different weights
    # (ratings, shared_classifications, shared_AddIndo, distance)
    recommended_df = df_weighted_sorter (recommended_df,
                                         'rating',
                                         'shared_classifications',
                                         'shared_AddInfo',
                                         'distance',
                                         expected_rating_weight,
                                         shared_classification_weight,
                                         shared_AddInfo_weight,
                                         distance_weight)

    # Filling in recommendations if they did not change
    cursor.execute("SELECT TOP 1 * "
                   "FROM RECOMMENDATIONS "
                   "WHERE UserId = ? ORDER BY UserId", user_id_to_search)

    last = cursor.fetchall()
    for row in last:
        last_rec_1 = row.Rec_1_Id
        last_rec_2 = row.Rec_2_Id
        last_rec_3 = row.Rec_3_Id
        last_rec_4 = row.Rec_4_Id
        last_rec_5 = row.Rec_5_Id
        last_rec_6 = row.Rec_6_Id
        last_rec_7 = row.Rec_7_Id
        last_rec_8 = row.Rec_8_Id
        last_rec_9 = row.Rec_9_Id
        last_rec_10 = row.Rec_10_Id
        if not (int(recommended_df.BusinessId.iloc[0]) == last_rec_1 and
            int(recommended_df.BusinessId.iloc[1]) == last_rec_2 and
            int(recommended_df.BusinessId.iloc[2]) == last_rec_3 and
            int(recommended_df.BusinessId.iloc[3]) == last_rec_4 and
            int(recommended_df.BusinessId.iloc[4]) == last_rec_5 and
            int(recommended_df.BusinessId.iloc[5]) == last_rec_6 and
            int(recommended_df.BusinessId.iloc[6]) == last_rec_7 and
            int(recommended_df.BusinessId.iloc[7]) == last_rec_8 and
            int(recommended_df.BusinessId.iloc[8]) == last_rec_9 and
            int(recommended_df.BusinessId.iloc[9]) == last_rec_10):
            cursor.execute("INSERT INTO RECOMMENDATIONS (UserId, Rec_1_Id, Rec_2_Id, Rec_3_Id,"
                           " Rec_4_Id, Rec_5_Id, Rec_6_Id, Rec_7_Id, Rec_8_Id, Rec_9_Id, Rec_10_Id, Rec_TS)"
                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                           user_id_to_search,
                           int(recommended_df.BusinessId.iloc[0]),
                           int(recommended_df.BusinessId.iloc[1]),
                           int(recommended_df.BusinessId.iloc[2]),
                           int(recommended_df.BusinessId.iloc[3]),
                           int(recommended_df.BusinessId.iloc[4]),
                           int(recommended_df.BusinessId.iloc[5]),
                           int(recommended_df.BusinessId.iloc[6]),
                           int(recommended_df.BusinessId.iloc[7]),
                           int(recommended_df.BusinessId.iloc[8]),
                           int(recommended_df.BusinessId.iloc[9]))

    cnxn.commit()

cnxn.close()

# Code to be triggered once a day to update data on server


'''
end of recommender_system_hidden_password.py
'''






