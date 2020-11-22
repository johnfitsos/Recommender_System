import pandas as pd
import numpy as np

rating_df = pd.read_csv ("BX-Book-Ratings.csv",sep = ";",  encoding= "Latin-1",)
print(rating_df)



users_df = pd.read_csv ("BX-Users.csv",sep = ";",  encoding= "Latin-1",)
print(users_df)



books_df = pd.read_csv ("BX-Books_clean.csv",sep = ";",  encoding= "Latin-1")
print(books_df)


user_rating_df = pd.merge(rating_df, users_df,on="User-ID", how='inner')
print(user_rating_df)


final_merge_df = pd.merge(user_rating_df,books_df,on='ISBN',how='inner')
print(final_merge_df)


#Dataset size

print('The dataset has:',final_merge_df.shape,'->(Rows,Columns)')

print('Size of the dataset is: ', final_merge_df.memory_usage().sum() / 1024**2, ' MB')



#Book Popularity
book_popularity_df = final_merge_df.groupby(by='ISBN').count().sort_values(by='User-ID',ascending = False)
print(book_popularity_df['User-ID'])



#Author Popularity
author_popularity_df=final_merge_df.groupby(by='Book-Author').count().sort_values(by='User-ID', ascending=False)
print(author_popularity_df['User-ID'])

#Age ranges by reading activity
age_groups = pd.cut(final_merge_df['Age'], bins=[0, 20, 30, 50, 80, np.inf])

age_ranges_df= final_merge_df.groupby(age_groups).count().sort_values(by='User-ID', ascending=False)
print(age_ranges_df['ISBN'])
