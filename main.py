import mysql.connector
import pandas as pd
import numpy as np
import os
import re


def main_part():

    # -----------------------------------BOOKS CLEANING----------------------------------------------------

    df_books = pd.read_csv('BX-Books.csv', sep='";"', encoding='Latin-1',
                           engine='python')  # use of python engine in order to use more complex sep pattern
    print('The number of the initial dataset is:', len(df_books), 'lines\n')

    # ISBN and Image-URL-L cols has " symbol attached due to read_csv method and has to be removed
    df_books['"ISBN'] = df_books['"ISBN'].str.replace('"', '')
    df_books['Image-URL-L"'] = df_books['Image-URL-L"'].str.replace('"', '')
    df_books.rename(columns={'"ISBN': 'ISBN', 'Image-URL-L"': 'Image-URL-L'}, inplace=True)

    # 1.FIX INVALID ISBN(VALID ISBN ONLY WITH 10 DIGITS)
    print('The number of ISBN with equal to 10 characters is:', len(df_books[df_books['ISBN'].apply(len) == 10]),
          'lines\n')
    print('The number of ISBN with more than 10 characters is:', len(df_books[df_books['ISBN'].apply(len) > 10]),
          'lines\n')
    print('The number of ISBN with lower than 10 characters is:', len(df_books[df_books['ISBN'].apply(len) < 10]),
          'lines\n')

    # 1.1 Keep only the first 10 digits in order to fix the >10 ISBN
    df_books['ISBN'] = df_books['ISBN'].str[0:10]

    # 1.2 Check for punctuations in ISBN
    print('The number of ISBN with puntuation symbols is:', len(df_books[df_books['ISBN'].str.contains(r'[^\w]')]),
          'lines\n')

    # 2. Check for invalid symbols in Book titles and fix them
    df_books['Book-Title'] = df_books['Book-Title'].str.lower()
    df_books['Book-Title'] = df_books['Book-Title'].str.replace(r'[^\w\s]', '')

    # 3. GROUP BY ISBN
    # 3.1 Drop duplicated ISBN
    temp = df_books.groupby('ISBN').count().sort_values('Book-Title', ascending=False)
    print(temp.head(), '\n')
    df_books = df_books.drop_duplicates(subset='ISBN', keep='first')
    print('The number of the clear of ISBN dataset is:', len(df_books), 'lines\n')

    # 4.GROUP BY BOOK TITLE
    # 4.1 See dublicated book titles
    temp2 = df_books.groupby('Book-Title').count().sort_values('ISBN', ascending=False)
    print(temp2.head(), '\n')
    df_books.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], inplace=True)

    df_books.to_csv('BX-Books_clean.csv')

    # ------------------------------Ratigns Cealing----------------------------------------------------

    rating_df = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='Latin-1')
    print('1. The initial number of Total Ratings is:', len(rating_df), '\n')

    # OUTLIERS DETECTION AND DROP GROUP BY USER_ID
    rating_count_per_user_df = rating_df.groupby(by='User-ID').count().sort_values(by='Book-Rating', ascending=False)

    mean = rating_count_per_user_df['Book-Rating'].mean()
    print('The mean of ratings per user is:', mean, '\n')
    std = rating_count_per_user_df['Book-Rating'].std()
    print('The standard deviation of ratings per user is:', std, '\n')
    threshold = 4

    rating_count_per_user_df["z-score"] = (rating_count_per_user_df[
                                               'Book-Rating'] - mean) / std  # turn values to z values and add these in a z-score column

    rating_count_per_user_df = rating_count_per_user_df[
        rating_count_per_user_df['z-score'] < threshold]  ## drop lines with an exrteme number of ratings
    rating_count_per_user_df = rating_count_per_user_df[
        rating_count_per_user_df['ISBN'] > 1]  ## drop lines that isbn have received only 1 rating

    rating_df_clean = rating_df[rating_df['User-ID'].isin(rating_count_per_user_df.index)].sort_values(by='Book-Rating',
                                                                                                       ascending=False)  # make match with the initial df

    print('2. The total ratings after we drop the outliers that have a relationship with number per user is:',
          len(rating_df), '\n')

    ## OUTLIERS DETECTION AND DROP GROUP BY ISBN
    rating_count_per_isbn_df = rating_df_clean.groupby(by="ISBN").count().sort_values(by='User-ID', ascending=False)

    rating_count_per_isbn_df = rating_count_per_isbn_df[
        rating_count_per_isbn_df['User-ID'] > 1]  # drop user with only 1 rating

    rating_df_clean = rating_df_clean[rating_df_clean['ISBN'].isin(rating_count_per_isbn_df.index)].sort_values(
        by='User-ID', ascending=True)  # make match with initial
    print('3. The total ratings after we drop all outliers is', len(rating_df_clean), '\n')

    ## FIX INVALID ISBN(VALID ISBN ONLY WITH 10 DIGITS)
    print('The number of ISBN with equal to 10 characters is:',
          len(rating_df_clean[rating_df_clean['ISBN'].apply(len) == 10]), '\n')
    print('The number of ISBN with more than 10 characters is:',
          len(rating_df_clean[rating_df_clean['ISBN'].apply(len) > 10]), '\n')
    print('The number of ISBN with lower than 10 characters is:',
          len(rating_df_clean[rating_df_clean['ISBN'].apply(len) < 10]), '\n')
    # 1. Check ISBN > 10 characters
    # 1.1 Try to remove punctuations from ISBN with more than 10 chars in order to add by mistake ISBN in their valid version
    rating_df_clean['ISBN'] = rating_df_clean['ISBN'].astype(str)
    rating_df_clean['ISBN'] = rating_df_clean['ISBN'].str.replace(r'[^\w]', '')
    rating_df_clean = rating_df_clean[rating_df_clean['ISBN'].apply(len) == 10]
    print(rating_df_clean.head(20))
    print('The number of ISBN with equal to 10 characters after we add the fixed values is:', len(rating_df_clean),
          '\n')
    # 2. Check 10-digit ISBN not start with letter
    rating_df_clean = rating_df_clean[rating_df_clean['ISBN'].str.contains(r'^[a-zA-Z]') == False]
    print('The number of valid ISBN that does not start with letter is:',
          len(rating_df_clean[rating_df_clean['ISBN'].apply(len) == 10]), '\n')
    # 3. Drop all ISBN that have just repeated ivalid digits
    rating_df_clean = rating_df_clean[rating_df_clean['ISBN'].str.contains(r'\b(\d)\1+\b') == False]
    print('The number of valid ISBN that has not invalid value of repeated digits is:',
          len(rating_df_clean[rating_df_clean['ISBN'].apply(len) == 10]), '\n')

    rating_df_clean = rating_df_clean[rating_df_clean['ISBN'].apply(len) == 10]

    ## DROP ZERO RATINGS OF HATERS
    final_rating_clean = rating_df_clean[rating_df_clean['Book-Rating'] > 0]
    final_rating_clean = final_rating_clean[final_rating_clean['Book-Rating'] < 11]
    print('The cleaned ratings dataset has', len(final_rating_clean), 'lines')

    final_rating_clean.to_csv('BX-Book-Ratings_clean.csv', index=False)

    # ____________________________________USERS CLEANING________________________________________________________________

    users_df = pd.read_csv('BX-Users.csv', sep=";", encoding="Latin-1")
    print('The initial number of users is:', len(users_df), 'lines\n')
    # CHECK USER-ID
    users_df = users_df.astype(str)
    users_df = users_df[users_df['User-ID'].str.contains(r'\d*')]
    print('The number of users is:', len(users_df), '\n')
    # CHECK AGES
    users_df = users_df[(users_df['Age'] < '99') & (users_df['Age'] > '10')]
    print('The number of users after we exclude some ages as outliers is:', len(users_df))

    print(users_df)

    users_df.to_csv('BX-Users_clean.csv', index=False)

    # ----------------------------------COMMON_VALUES BETWEEN TABLES--------------------------------------------

    df_ratings = pd.read_csv('BX-Book-Ratings_clean.csv')
    df_users = pd.read_csv('BX-Users_clean.csv')
    df_books = pd.read_csv('BX-Books_clean.csv')
    print('Users =', len(df_users))
    print('Ratings=', len(df_ratings))
    print('Books=', len(df_books))
    # Checking common USER-ID
    df_ratings = df_ratings[df_ratings['User-ID'].isin(df_users['User-ID'])]
    print('We have', len(df_ratings), 'ratings with valid User-ID')
    # Checking common ISBN
    df_ratings = df_ratings[df_ratings['ISBN'].isin(df_books['ISBN'])]
    print('Finally we have', len(df_ratings), 'ratings')

    df_ratings.to_csv('BX-Book-Ratings_clean.csv', index=False)

    # -----------------------------C_SIM--------------------------------------------------------------------

    # Delete csv if it exists because we use append modes later on
    if os.path.exists("user-pairs.csv"):
        os.remove('user-pairs.csv')

    if os.path.exists("user_means.csv"):
        os.remove('user_means.csv')

    #In this function u,v must have the same length
    # u is the ratings of user_a and v ratings of user_b (Only for their common books)
    # user_one and user_two contains all the ratings of each user
    def csim(u, v, user_one, user_two):
        return np.dot(u, v) / (np.linalg.norm(user_one) * np.linalg.norm(user_two))

    user_pairs = pd.DataFrame(columns=['user1', 'user2', 'similarity'])

    # Read csv and sort values
    df = pd.read_csv('BX-Book-Ratings_clean.csv').sort_values(by='User-ID')
    df['Book-Rating'] = df['Book-Rating'].astype('int8')
    df['User-ID'] = df['User-ID'].astype('int32')
    print('Creating Dictionary...')
    # Create empty dicts and lists
    isbn_dict = {"ISBN": [], "ratings": []}
    user_dict = []
    user_list = []

    # group by isbn splits df into multiple dfs based on user-id
    df_g = df.groupby('User-ID')['ISBN', 'Book-Rating']

    for User, User_df in df_g:
        temp_df = df_g.get_group(User)  # get helps isolate subset of our df
        isbn_dict = temp_df.to_dict('list')  # append user dict
        user_dict.append(isbn_dict)
        user_list.append(User)
    #
    # # we have a list of dictionaries and each dictionary has keys: ISBN and Book-Ratings
    # And as value of each key is a list with all books the user read in the past and the ratings of them
    # by zipping into dict not we have a dictionary with
    # # key = user-ID and values = dictionary (the dictionary above)

    books_dictionary = dict(zip(user_list, user_dict))
    # Now we have a nested dictionary
    print('Done\n start looking for similarities')

    # You can uncomment the next for loop to see the results of the zipped dictionary
    # for key, value in books_dictionary.items():
    #     print(key,value)

    counter = 0

    for user_a, values_a in books_dictionary.items():  # loop for each user to compare him with all the other users
        counter += 1
        if counter % 1000 == 0:
            print(counter, len(books_dictionary))
        books_a = values_a['ISBN'] # All the books user_a has rate so far
        ratings_books_a = values_a['Book-Rating'] # All the ratings of those books
        mean_user_a = np.mean(ratings_books_a) # Mean rating of user_a
        for user_b, values_b in books_dictionary.items():  # second user to compare
            if user_b != user_a:  # check not to compare user with himself
                u = []  # vector of ratings for user a
                v = []  # vector of ratings for user b
                books_b = values_b['ISBN']  # list of books of user_b
                ratings_books_b = values_b['Book-Rating'] # Ratings of those books

                if set(books_b).intersection(books_a):
                    for book in books_b:  # check if they have common books
                        if book in books_a:
                            # the .index helps append the 2 vectors with respect at the position
                            # because we need each rating's position of vector u for a specific book to be in the same
                            # index position at vector v
                            u.append(values_a['Book-Rating'][books_a.index(book)])
                            v.append(values_b['Book-Rating'][books_b.index(book)])
                x = float(csim(u, v, ratings_books_a, ratings_books_b))
                text = f'{user_a};{user_b};{x}\n'
                print(text)
                #If the 2 vectors are not empty then we can calculate the similarity
                if len(u) > 0 and len(v) > 0:
                    x = float(csim(u, v, ratings_books_a, ratings_books_b))
                    with open('user_similarity.csv', 'a') as out:
                        text = f'{user_a};{user_b};{x}\n'
                        # out.write(text)
                    with open('user-pairs-books.data', 'a') as out2:
                        text = f'{user_a};{user_b};{x}\n'
                        # out2.write(text)
    out.close()
    out2.close()

    # Also store the mean rating of each user in a csv for later use with easier access to it
    for user_a, values_a in books_dictionary.items():  # loop for your to compare
        ratings_books_a = values_a['Book-Rating']
        mean_user_a = np.mean(ratings_books_a)
        with open('user_av_rating.csv', 'a') as out3:
            text = f'{user_a};{mean_user_a}\n'
            out3.write(text)
    out3.close()


def prediction_world():
    # _____________________________________K_nearest_______________________________________________________________
    # Delete the csv if it exists
    if os.path.exists("k_nearest.csv"):
        os.remove('k_nearest.csv')

    # Read as DataFrame the csv of user pairs similarity
    df = pd.read_csv('user_similarity.csv', sep=';', names=['user_a', 'user_b', 'c_sim'])
    k = 2 # Set the number of nearest neighbors we want

    x = df.groupby(['user_a'])[['user_b', 'c_sim']]

    print('export k_nearest...')
    # Group the DataFrame based on user_a
    for user_a, user_b in x:
        temp_df = x.get_group(user_a)
        temp_df.sort_values(by='c_sim', ascending=False, inplace=True) # Sort the data so highest similarity is up
        temp_df['user_a'] = user_a
        temp_df = temp_df[:k] # Keep only the first k neighbors

        temp_df = temp_df[['user_a', 'user_b', 'c_sim']] #name the columns
        # print(temp_df)
        temp_df.to_csv('k_nearest.csv', mode='a', header=False, index=False)
    print('Finished!')

    # temp_df.to_json('k_nearest_json')

    # -------------------------------------Predict----------------------------------------------------------------
    # Read the appropriate DataFrames and delete the csv with predictions of each user if exists
    df_mean = pd.read_csv('user_av_rating.csv', names=['user', 'mean_rating'], sep=';')
    df_mean['mean_rating'] = df_mean['mean_rating'].astype('float32')
    df_mean['user'] = df_mean['user'].astype('int32')
    df_mean.set_index('user')

    if os.path.exists("user_predictions.csv"):
        os.remove("user_predictions.csv")

    print(df_mean.head())

    df_ratings = pd.read_csv('BX-Book-Ratings_clean.csv')
    df_ratings['Book-Rating'] = df_ratings['Book-Rating'].astype('int8')
    df_ratings['User-ID'] = df_ratings['User-ID'].astype('int32')
    df_ratings.set_index('User-ID')

    print(df_ratings.head())

    df_nearest = pd.read_csv('k_nearest.csv', sep=',', names=['user_a', 'user_b', 'c_sim'])
    df_nearest['c_sim'] = df_nearest['c_sim'].astype('float32')
    df_nearest['user_a'] = df_nearest['user_a'].astype('int32')
    df_nearest['user_b'] = df_nearest['user_b'].astype('int32')
    print(df_nearest.head())
    # errors = []

    counter = 0
    # Loop for each unique user
    for user_a in df_mean['user']:
        counter += 1
        if counter % 500 == 0:
            print(round((counter / len(df_mean) * 100), 2), '%')
        user_a_books = list(df_ratings['ISBN'][df_ratings['User-ID'] == user_a]) #books of user a
        mean_a = df_mean['mean_rating'][df_mean['user'] == user_a].values[0] # mean rating of user_a
        neighbors_a = list(df_nearest[df_nearest['user_a'] == user_a]['user_b']) # list with all user_a's neighbors
        if len(neighbors_a) > 0: # If he has neighbors we are calculating the predicted rating, else next user
            # For each book of user a we will check if any of his neighbors has rate this book
            # If so we will add at the nominator or prediction formula the coresponding value
            for book in user_a_books:
                nominator = 0
                denominator = 0

                for user_b in neighbors_a: # for each neighbor of user_a
                    # find his mean rating and the similarity with user_a
                    user_b_books = list(df_ratings['ISBN'][df_ratings['User-ID'] == user_b])
                    mean_b = df_mean[df_mean['user'] == user_b]['mean_rating'].values[0]
                    # print(mean_b)
                    sim = df_nearest['c_sim'][
                        (df_nearest['user_b'] == user_b) & (df_nearest['user_a'] == user_a)].values
                    denominator += sim
                    # If they have a common book then add the value to the nominator
                    if book in user_b_books:
                        user_b_rating = df_ratings['Book-Rating'][(df_ratings['User-ID'] == user_b) & (
                                df_ratings['ISBN'] == book)].values[0]
                        nominator += sim * (user_b_rating - mean_b)
                    # Else add 0 to the nominator by replacing the missing rating of user_b for that book with his mean
                    # Which will result 0
                    else:
                        user_b_rating = mean_b
                        nominator += sim * (user_b_rating - mean_b)
                prediction = (mean_a + (nominator / denominator))[0]
                # After applying the formula we have a prediction for user_a for each book he has read already
                # And we save these results in a csv file
                with open('user_predictions.csv', 'a') as output:
                    text = f'{user_a};{book};{round(float(prediction), 2)}\n'
                    output.write(text)
                # Now we can go on and evaluate these predictions with the original ratings of each user

    output.close()

    # ------------------------------------------------EVALUATION--------------------------------------------------
    # Fix predictions more than 10 oless than 0
    df_predict = pd.read_csv('user_predictions.csv', sep=';', names=['User-ID', 'ISBN', 'Prediction'])
    df_predict[df_predict['Prediction'] > 10] = 10
    df_predict[df_predict['Prediction'] < 0] = 0
    print(df_predict.head())

    df_ratings = pd.read_csv('BX-Book-Ratings_clean.csv')
    print(df_ratings.head())
    # The evaluation will be applied only on users that :
    # Had similarity with at least 1 user by subtracting each prediction from his initial rating
    df_merged = pd.merge(df_predict, df_ratings, how='inner', on=['User-ID', 'ISBN'])
    df_merged['diff'] = df_merged['Prediction'] - df_merged['Book-Rating']
    df_merged['diff_square'] = (df_merged['Prediction'] - df_merged['Book-Rating']) ** 2
    print(df_merged.sort_values(by='diff', ascending=False))

    # And here we constract out formulas to measure those differences
    mae = sum(np.abs(df_merged['diff'])) / len(df_merged['diff'])
    nmae = mae / (df_merged['Book-Rating'].max() - df_merged['Book-Rating'].min())
    rmse = np.sqrt(sum(df_merged['diff_square']) / len(df_merged['diff_square']))

    print(len(df_merged), 'merged')
    print(len(df_ratings), 'ratings')
    print(len(df_predict), 'predictions')

    print('mae:', mae)
    print('nmae: ', nmae)
    print('rmse: ', rmse)


def prediction_america():
    # _____________________________________K_nearest_______________________________________________________________
    # This function will calculate predict and evaluate the results just like the previous one based only on
    # users from USA and Canada
    if os.path.exists("k_nearest_america.csv"):
        os.remove('k_nearest_america.csv')

    df = pd.read_csv('user_similarity.csv', sep=';', names=['user_a', 'user_b', 'c_sim'])
    users_df = pd.read_csv('BX-Users_clean.csv')
    users_df = users_df[(users_df['Location'].str.contains(', usa')) | (users_df['Location'].str.contains(', canada'))]
    df = df[df['user_a'].isin(users_df['User-ID'])]
    df = df[df['user_b'].isin(users_df['User-ID'])]
    k = 2

    x = df.groupby(['user_a'])[['user_b', 'c_sim']]

    print('export k_nearest...')

    for user_a, user_b in x:
        temp_df = x.get_group(user_a)
        temp_df.sort_values(by='c_sim', ascending=False, inplace=True)
        temp_df['user_a'] = user_a
        temp_df = temp_df[:k]
        # temp_df.set_index('user_a',inplace=True)

        temp_df = temp_df[['user_a', 'user_b', 'c_sim']]
        # print(temp_df)
        temp_df.to_csv('k_nearest_america.csv', mode='a', header=False, index=False)
    print('Finished!')

    # temp_df.to_json('k_nearest_json')

    # -------------------------------------Predict----------------------------------------------------------------

    df_mean = pd.read_csv('user_av_rating.csv', names=['user', 'mean_rating'], sep=';')
    df_mean['mean_rating'] = df_mean['mean_rating'].astype('float32')
    df_mean['user'] = df_mean['user'].astype('int32')
    df_mean = df_mean[df_mean['user'].isin(users_df['User-ID'])]
    df_mean.set_index('user')

    if os.path.exists("user_predictions_america.csv"):
        os.remove("user_predictions_america.csv")

    print(df_mean.head())

    df_ratings = pd.read_csv('BX-Book-Ratings_clean.csv')
    df_ratings['Book-Rating'] = df_ratings['Book-Rating'].astype('int8')
    df_ratings['User-ID'] = df_ratings['User-ID'].astype('int32')
    df_ratings.set_index('User-ID')

    print(df_ratings.head())

    df_nearest = pd.read_csv('k_nearest_america.csv', sep=',', names=['user_a', 'user_b', 'c_sim'])
    df_nearest['c_sim'] = df_nearest['c_sim'].astype('float32')
    df_nearest['user_a'] = df_nearest['user_a'].astype('int32')
    df_nearest['user_b'] = df_nearest['user_b'].astype('int32')
    print(df_nearest.head())
    # errors = []

    counter = 0
    for user_a in df_mean['user']:
        counter += 1
        if counter % 500 == 0:
            print(round((counter / len(df_mean) * 100), 2), '%')
        user_a_books = list(df_ratings['ISBN'][df_ratings['User-ID'] == user_a])
        mean_a = df_mean['mean_rating'][df_mean['user'] == user_a].values[0]
        # print(mean_a)
        neighbors_a = list(df_nearest[df_nearest['user_a'] == user_a]['user_b'])
        if len(neighbors_a) > 0:
            for book in user_a_books:
                nominator = 0
                denominator = 0

                for user_b in neighbors_a:
                    user_b_books = list(df_ratings['ISBN'][df_ratings['User-ID'] == user_b])
                    mean_b = df_mean[df_mean['user'] == user_b]['mean_rating'].values[0]
                    sim = df_nearest['c_sim'][
                        (df_nearest['user_b'] == user_b) & (df_nearest['user_a'] == user_a)].values
                    denominator += sim

                    if book in user_b_books:
                        user_b_rating = df_ratings['Book-Rating'][(df_ratings['User-ID'] == user_b) & (
                                df_ratings['ISBN'] == book)].values[0]
                        nominator += sim * (user_b_rating - mean_b)
                    else:
                        user_b_rating = mean_b
                        nominator += sim * (user_b_rating - mean_b)
                prediction = (mean_a + (nominator / denominator))[0]
                with open('user_predictions_america.csv', 'a') as output:
                    text = f'{user_a};{book};{round(float(prediction), 2)}\n'
                    output.write(text)

    output.close()

    # ------------------------------------------------EVALUATION--------------------------------------------------

    df_predict = pd.read_csv('user_predictions_america.csv', sep=';', names=['User-ID', 'ISBN', 'Prediction'])
    df_predict[df_predict['Prediction'] > 10] = 10
    df_predict[df_predict['Prediction'] < 0] = 0
    print(df_predict.head())

    df_ratings = pd.read_csv('BX-Book-Ratings_clean.csv')
    print(df_ratings.head())

    df_merged = pd.merge(df_predict, df_ratings, how='inner', on=['User-ID', 'ISBN'])
    df_merged['diff'] = df_merged['Prediction'] - df_merged['Book-Rating']
    df_merged['diff_square'] = (df_merged['Prediction'] - df_merged['Book-Rating']) ** 2
    print(df_merged.sort_values(by='diff', ascending=False))

    mae = sum(np.abs(df_merged['diff'])) / len(df_merged['diff'])
    nmae = mae / (df_merged['Book-Rating'].max() - df_merged['Book-Rating'].min())
    rmse = np.sqrt(sum(df_merged['diff_square']) / len(df_merged['diff_square']))

    print(len(df_merged), 'merged')
    print(len(df_ratings), 'ratings')
    print(len(df_predict), 'predictions')

    print('mae:', mae)
    print('nmae: ', nmae)
    print('rmse: ', rmse)


def non_america():
    # _____________________________________K_nearest_______________________________________________________________
    # This function will calculate predict and evaluate the results just like the previous one based only on
    # users not from USA and Canada

    if os.path.exists("k_nearest_non_america.csv"):
        os.remove('k_nearest_non_america.csv')

    df = pd.read_csv('user_similarity.csv', sep=';', names=['user_a', 'user_b', 'c_sim'])
    users_df = pd.read_csv('BX-Users_clean.csv')
    users_df = users_df[
        ((users_df['Location'].str.contains(', usa')) | (users_df['Location'].str.contains(', canada'))) == False]
    df = df[df['user_a'].isin(users_df['User-ID'])]
    df = df[df['user_b'].isin(users_df['User-ID'])]
    k = 2

    x = df.groupby(['user_a'])[['user_b', 'c_sim']]

    print('export k_nearest...')

    for user_a, user_b in x:
        temp_df = x.get_group(user_a)
        temp_df.sort_values(by='c_sim', ascending=False, inplace=True)
        temp_df['user_a'] = user_a
        temp_df = temp_df[:k]
        # temp_df.set_index('user_a',inplace=True)

        temp_df = temp_df[['user_a', 'user_b', 'c_sim']]
        # print(temp_df)
        temp_df.to_csv('k_nearest_non_america.csv', mode='a', header=False, index=False)
    print('Finished!')

    # temp_df.to_json('k_nearest_json')

    # -------------------------------------Predict----------------------------------------------------------------

    df_mean = pd.read_csv('user_av_rating.csv', names=['user', 'mean_rating'], sep=';')
    df_mean['mean_rating'] = df_mean['mean_rating'].astype('float32')
    df_mean['user'] = df_mean['user'].astype('int32')
    df_mean = df_mean[df_mean['user'].isin(users_df['User-ID'])]
    df_mean.set_index('user')

    if os.path.exists("user_predictions_non_america.csv"):
        os.remove("user_predictions_non_america.csv")

    print(df_mean.head())

    df_ratings = pd.read_csv('BX-Book-Ratings_clean.csv')
    df_ratings['Book-Rating'] = df_ratings['Book-Rating'].astype('int8')
    df_ratings['User-ID'] = df_ratings['User-ID'].astype('int32')
    df_ratings.set_index('User-ID')

    print(df_ratings.head())

    df_nearest = pd.read_csv('k_nearest_non_america.csv', sep=',', names=['user_a', 'user_b', 'c_sim'])
    df_nearest['c_sim'] = df_nearest['c_sim'].astype('float32')
    df_nearest['user_a'] = df_nearest['user_a'].astype('int32')
    df_nearest['user_b'] = df_nearest['user_b'].astype('int32')
    print(df_nearest.head())
    # errors = []

    counter = 0
    for user_a in df_mean['user']:
        counter += 1
        if counter % 500 == 0:
            print(round((counter / len(df_mean) * 100), 2), '%')
        user_a_books = list(df_ratings['ISBN'][df_ratings['User-ID'] == user_a])
        mean_a = df_mean['mean_rating'][df_mean['user'] == user_a].values[0]
        # print(mean_a)
        neighbors_a = list(df_nearest[df_nearest['user_a'] == user_a]['user_b'])
        if len(neighbors_a) > 0:
            for book in user_a_books:
                nominator = 0
                denominator = 0

                for user_b in neighbors_a:
                    user_b_books = list(df_ratings['ISBN'][df_ratings['User-ID'] == user_b])
                    mean_b = df_mean[df_mean['user'] == user_b]['mean_rating'].values[0]
                    sim = df_nearest['c_sim'][
                        (df_nearest['user_b'] == user_b) & (df_nearest['user_a'] == user_a)].values
                    denominator += sim

                    if book in user_b_books:
                        user_b_rating = df_ratings['Book-Rating'][(df_ratings['User-ID'] == user_b) & (
                                df_ratings['ISBN'] == book)].values[0]
                        nominator += sim * (user_b_rating - mean_b)
                    else:
                        user_b_rating = mean_b
                        nominator += sim * (user_b_rating - mean_b)
                prediction = (mean_a + (nominator / denominator))[0]
                with open('user_predictions_non_america.csv', 'a') as output:
                    text = f'{user_a};{book};{round(float(prediction), 2)}\n'
                    output.write(text)

    output.close()

    # ------------------------------------------------EVALUATION--------------------------------------------------

    df_predict = pd.read_csv('user_predictions_non_america.csv', sep=';', names=['User-ID', 'ISBN', 'Prediction'])
    df_predict[df_predict['Prediction'] > 10] = 10
    df_predict[df_predict['Prediction'] < 0] = 0
    print(df_predict.head())

    df_ratings = pd.read_csv('BX-Book-Ratings_clean.csv')
    print(df_ratings.head())

    df_merged = pd.merge(df_predict, df_ratings, how='inner', on=['User-ID', 'ISBN'])
    df_merged['diff'] = df_merged['Prediction'] - df_merged['Book-Rating']
    df_merged['diff_square'] = (df_merged['Prediction'] - df_merged['Book-Rating']) ** 2
    print(df_merged.sort_values(by='diff', ascending=False))

    mae = sum(np.abs(df_merged['diff'])) / len(df_merged['diff'])
    nmae = mae / (df_merged['Book-Rating'].max() - df_merged['Book-Rating'].min())
    rmse = np.sqrt(sum(df_merged['diff_square']) / len(df_merged['diff_square']))

    print(len(df_merged), 'merged')
    print(len(df_ratings), 'ratings')
    print(len(df_predict), 'predictions')

    print('mae:', mae)
    print('nmae: ', nmae)
    print('rmse: ', rmse)


def country(c):
    # _____________________________________K_nearest_______________________________________________________________
    #  This function will calculate predict and evaluate the results just like the previous one based only on
    #  The country you will ask for.
    if os.path.exists("k_nearest_" + c + ".csv"):
        os.remove("k_nearest_" + c + ".csv")

    df = pd.read_csv('user_similarity.csv', sep=';', names=['user_a', 'user_b', 'c_sim'])
    users_df = pd.read_csv('BX-Users_clean.csv')
    users_df = users_df[users_df['Location'].str.contains(', ' + c)]
    df = df[df['user_a'].isin(users_df['User-ID'])]
    df = df[df['user_b'].isin(users_df['User-ID'])]
    k = 2

    x = df.groupby(['user_a'])[['user_b', 'c_sim']]

    print('export k_nearest...')

    for user_a, user_b in x:
        temp_df = x.get_group(user_a)
        temp_df.sort_values(by='c_sim', ascending=False, inplace=True)
        temp_df['user_a'] = user_a
        temp_df = temp_df[:k]
        # temp_df.set_index('user_a',inplace=True)

        temp_df = temp_df[['user_a', 'user_b', 'c_sim']]
        # print(temp_df)
        temp_df.to_csv("k_nearest_" + c + ".csv", mode='a', header=False, index=False)
    print('Finished!')

    # temp_df.to_json('k_nearest_json')

    # -------------------------------------Predict----------------------------------------------------------------

    df_mean = pd.read_csv('user_av_rating.csv', names=['user', 'mean_rating'], sep=';')
    df_mean['mean_rating'] = df_mean['mean_rating'].astype('float32')
    df_mean['user'] = df_mean['user'].astype('int32')
    df_mean = df_mean[df_mean['user'].isin(users_df['User-ID'])]
    df_mean.set_index('user')

    if os.path.exists("user_predictions_" + c + ".csv"):
        os.remove("user_predictions_" + c + ".csv")

    print(df_mean.head())

    df_ratings = pd.read_csv('BX-Book-Ratings_clean.csv')
    df_ratings['Book-Rating'] = df_ratings['Book-Rating'].astype('int8')
    df_ratings['User-ID'] = df_ratings['User-ID'].astype('int32')
    df_ratings.set_index('User-ID')

    print(df_ratings.head())

    df_nearest = pd.read_csv("k_nearest_" + c + ".csv", sep=',', names=['user_a', 'user_b', 'c_sim'])
    df_nearest['c_sim'] = df_nearest['c_sim'].astype('float32')
    df_nearest['user_a'] = df_nearest['user_a'].astype('int32')
    df_nearest['user_b'] = df_nearest['user_b'].astype('int32')
    print(df_nearest.head())
    # errors = []

    counter = 0
    for user_a in df_mean['user']:
        counter += 1
        if counter % 500 == 0:
            print(round((counter / len(df_mean) * 100), 2), '%')
        user_a_books = list(df_ratings['ISBN'][df_ratings['User-ID'] == user_a])
        mean_a = df_mean['mean_rating'][df_mean['user'] == user_a].values[0]
        # print(mean_a)
        neighbors_a = list(df_nearest[df_nearest['user_a'] == user_a]['user_b'])
        if len(neighbors_a) > 0:
            for book in user_a_books:
                nominator = 0
                denominator = 0

                for user_b in neighbors_a:
                    user_b_books = list(df_ratings['ISBN'][df_ratings['User-ID'] == user_b])
                    mean_b = df_mean[df_mean['user'] == user_b]['mean_rating'].values[0]
                    sim = df_nearest['c_sim'][
                        (df_nearest['user_b'] == user_b) & (df_nearest['user_a'] == user_a)].values
                    denominator += sim

                    if book in user_b_books:
                        user_b_rating = df_ratings['Book-Rating'][(df_ratings['User-ID'] == user_b) & (
                                df_ratings['ISBN'] == book)].values[0]
                        nominator += sim * (user_b_rating - mean_b)
                    else:
                        user_b_rating = mean_b
                        nominator += sim * (user_b_rating - mean_b)
                prediction = (mean_a + (nominator / denominator))[0]
                # print(user_a,'--------',prediction,'------',nominator)
                with open('user_predictions_' + c + '.csv', 'a') as output:
                    text = f'{user_a};{book};{round(float(prediction), 2)}\n'
                    output.write(text)

    output.close()

    # ------------------------------------------------EVALUATION--------------------------------------------------

    df_predict = pd.read_csv('user_predictions_' + c + '.csv', sep=';', names=['User-ID', 'ISBN', 'Prediction'])
    df_predict[df_predict['Prediction'] > 10] = 10
    df_predict[df_predict['Prediction'] < 0] = 0
    print(df_predict.head())

    df_ratings = pd.read_csv('BX-Book-Ratings_clean.csv')
    print(df_ratings.head())

    df_merged = pd.merge(df_predict, df_ratings, how='inner', on=['User-ID', 'ISBN'])
    df_merged['diff'] = df_merged['Prediction'] - df_merged['Book-Rating']
    df_merged['diff_square'] = (df_merged['Prediction'] - df_merged['Book-Rating']) ** 2
    print(df_merged.sort_values(by='diff', ascending=False))

    mae = sum(np.abs(df_merged['diff'])) / len(df_merged['diff'])
    nmae = mae / (df_merged['Book-Rating'].max() - df_merged['Book-Rating'].min())
    rmse = np.sqrt(sum(df_merged['diff_square']) / len(df_merged['diff_square']))

    print(len(df_merged), 'merged')
    print(len(df_ratings), 'ratings')
    print(len(df_predict), 'predictions')

    print('mae:', mae)
    print('nmae: ', nmae)
    print('rmse: ', rmse)


def sql_load(p,u):
    # This function will create a database
    # Create the appropriete tables
    # and populate them in order to store the results by just running after you enter your MYSQL credentials
    conn = mysql.connector.connect(host='localhost', password=p, user=u, allow_local_infile=True)

    cursor = conn.cursor()

    cursor.execute("""
    DROP DATABASE IF EXISTS books;
    """)

    cursor.execute("""
    CREATE DATABASE IF NOT EXISTS books;
    """)

    cursor.execute("""
    USE books;
    """)

    cursor.execute("""
    DROP TABLE IF EXISTS user_neighbors;
    """)

    cursor.execute("""
    DROP TABLE IF EXISTS user_pairs;
    """)

    cursor.execute("""
    DROP TABLE IF EXISTS books_data;
    """)

    cursor.execute("""
    DROP TABLE IF EXISTS user_data;
    """)

    cursor.execute("""
    DROP TABLE IF EXISTS ratings_data;
    """)

    print('Creating Tables')

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS books_data(
    isbn varchar(10) NOT NULL PRIMARY KEY,
    title varchar(255) default NULL,
    author varchar(255) default NULL,
    year int(10) default NULL,
    publisher varchar(255) default NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_data(
    users_id int(14) NOT NULL PRIMARY KEY default '0',
    location varchar(250) default NULL,
    age int(3) default NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE  IF NOT EXISTS ratings_data(
    users_id int(14) NOT NULL default '0',
    FOREIGN KEY (users_id) REFERENCES user_data(users_id),
    isbn varchar(10) NOT NULL,
    rating int(11) NOT NULL default '0',
    FOREIGN KEY (isbn) REFERENCES books_data(isbn),
    PRIMARY KEY (users_id,isbn)
    );
    """)

    cursor.execute("""
    CREATE TABLE  IF NOT EXISTS user_pairs(
    user_a int NOT NULL,
    user_b int NOT NULL,
    c_sim decimal(20,20)
    );
    """)

    cursor.execute("""
    CREATE TABLE  IF NOT EXISTS user_neighbors(
    user_a int NOT NULL,
    user_b int NOT NULL,
    c_sim decimal(20,20)
    );
    """)

    print('Loading Books')

    cursor.execute("""
    load data local infile 'BX-Books_clean.csv' 
    into table books_data 
    fields terminated by ',' OPTIONALLY ENCLOSED BY '"'
    lines terminated by '\r\n' 
    ignore 1 lines;
    """)

    print('Loading Users')

    cursor.execute("""
    load data local infile 'BX-Users_clean.csv' 
    into table user_data 
    fields terminated by ',' OPTIONALLY ENCLOSED BY '"'
    lines terminated by '\r\n' 
    ignore 1 lines;
    """)

    print('Loading Ratings')

    cursor.execute("""
    load data local infile 'BX-Book-Ratings_clean.csv' 
    into table ratings_data 
    fields terminated by ',' OPTIONALLY ENCLOSED BY '"'
    lines terminated by '\r\n' 
    ignore 1 lines; 
    """)

    print('Loading User-pairs')

    cursor.execute("""
    load data local infile 'user-pairs.csv'
    into table user_pairs
    fields terminated by ';' OPTIONALLY ENCLOSED BY '"'
    lines terminated by '\n'
    ignore 1 lines;
    """)

    print('Loading K-nearest')

    cursor.execute("""
    load data local infile 'k_nearest.csv'
    into table user_neighbors
    fields terminated by ',' OPTIONALLY ENCLOSED BY '"'
    lines terminated by '\n'
    ignore 1 lines;
    """)

    conn.commit()
    conn.close()




#
# main_part()
# prediction_world()
# prediction_america()
# non_america()
#
# passwr = input('Enter password:')
# usern = input('Enter User-Name:')
# sql_load(p=passwr, u=usern)
#
# c = str(input('choose a country:'))
#
# try:
#
#     country(c)
# except:
#     print('Country not exists.')
