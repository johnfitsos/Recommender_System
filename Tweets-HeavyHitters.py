import numpy as np
import json
import probables
import math
import pandas as pd
import os
from pympler.asizeof import asizeof

path = os.getcwd()

try:
    os.mkdir(path+'\\results')
    print('You will find the results at:',path+'\\results')


except:
    print('Folder is already exist')

new_path=path+'\\results'



def every_1000_heavy_hitters(path):



    cms_th_users = probables.CountMinSketch(confidence=0.95, error_rate=0.00001)

    cms_tot_users = probables.CountMinSketch(confidence=0.95, error_rate=0.00001)

    cms_th_tags = probables.CountMinSketch(confidence=0.95, error_rate=0.001)

    cms_tot_tags = probables.CountMinSketch(confidence=0.95, error_rate=0.001)


    csv_number_counter=0
    row_counter = 0
    tags_dict = {}
    tags_dict_thousand = {}
    users_dict = {}
    users_dict_thousand={}
    top_hitters=int(input('Give me a number of hitters:'))
    thousand_top_hitters=int(input('Give me a number of hitters for each thousand:'))

    for i in range(0, 46):
        data = f'tweets.json.{i}'
        with open(data, encoding='utf-8') as json_file:

            for row in json_file:
                json_obj = json.loads(row)                  # Make every row from json object to dictionary
                user_id = json_obj['user']['id']
                tags = json_obj['entities']['hashtags']
                row_counter += 1

                if not users_dict_thousand.get(user_id):  # Add values to the dictionary
                    users_dict_thousand[user_id] = 0
                users_dict_thousand[user_id] += 1
                cms_th_users.add(str(user_id))

                for element in tags:
                    for key, value in element.items():
                        if key == 'text':
                            tag_th = str(value)
                            if not tags_dict_thousand.get(tag_th):
                                tags_dict_thousand[tag_th] = 0
                            tags_dict_thousand[tag_th] += 1
                            cms_th_tags.add(tag_th)

                if row_counter == 1000:
                    hh_count_users=0
                    hh_count_tags=0
                    csv_number_counter+=1
                    for key,value in sorted(users_dict_thousand.items(), key = lambda item: item[1], reverse=True):
                        # print('{0}: {1:3d}/{2}'.format(key, value, cms.check(str(key))))
                        with open(new_path+'\\users'+str(csv_number_counter)+'.csv', 'a') as out:
                            text = f'{key};{value};{cms_th_users.check(str(key))}\n'
                            out.write(text)
                        hh_count_users+=1
                        if hh_count_users==thousand_top_hitters:
                            break

                    for key, value in sorted(tags_dict_thousand.items(), key=lambda item: item[1], reverse=True):

                        try:
                            with open(new_path + '\\tags' + str(csv_number_counter) + '.csv', 'a') as out:
                                text = f'{key};{value};{cms_th_tags.check(str(key))}\n'
                                out.write(text)


                        except:
                            print((key, value), 'Could not be saved. It will be saved as undefined due to encoding')
                            key = 'undefined'

                            with open(new_path + '\\tags' + str(csv_number_counter) + '.csv', 'a',
                                      encoding='utf-8') as out:
                                text = f'{key};{value};{cms_th_tags.check(str(key))}\n'
                                out.write(text)

                        hh_count_tags += 1
                        if hh_count_tags == thousand_top_hitters:
                            break



                    users_dict_thousand = {}
                    tags_dict_thousand = {}
                    cms_th_users.clear()
                    cms_th_tags.clear()
                    row_counter = 0
                    print('------------NEXT THOUSAND-------------')

                if not users_dict.get(user_id):  # Add values to the dictionary
                    users_dict[user_id] = 0
                users_dict[user_id] += 1
                cms_tot_users.add(str(user_id))


                for element in tags:
                    for key, value in element.items():
                        if key == 'text':
                            tag = str(value)
                            if not tags_dict.get(tag):
                                tags_dict[tag] = 0
                            tags_dict[tag] += 1
                            cms_tot_tags.add(tag)


        print('File', i)
        print('===============================================')
        json_file.close()

    total_hh_count_users=0
    total_hh_count_tags=0
    for key,value in sorted(users_dict.items(), key = lambda item: item[1], reverse=True):


        with open('users_comparison.csv', 'a') as out:
            text = f'{key};{value};{cms_tot_users.check(str(key))}\n'
            out.write(text)
        total_hh_count_users += 1
        if total_hh_count_users == top_hitters:
            break

    for key, value in sorted(tags_dict.items(), key=lambda item: item[1], reverse=True):
        try:
            with open('tags_comparison.csv', 'a') as out:
                text = f'{key};{value};{cms_tot_tags.check(str(key))}\n'
                out.write(text)

        except:
            print((key, value), 'Could not be saved. It will be saved as undefined due to encoding')
            key = 'undefined'

            with open('tags_comparison.csv', 'a') as out:
                text = f'{key};{value};{cms_tot_tags.check(str(key))}\n'
                out.write(text)

        total_hh_count_tags += 1
        if total_hh_count_tags == top_hitters:
            break


    print('Space of actual counting approach for users is:', asizeof(users_dict), 'bytes \n')
    print('Space of approximate counting approach for users is:', asizeof(cms_tot_users), 'bytes \n')

    print('Space of actual counting approach for tags is:', asizeof(tags_dict), 'bytes \n')
    print('Space of approximate counting approach for tags is:', asizeof(cms_tot_tags), 'bytes \n')

every_1000_heavy_hitters(new_path)

df_total_users=pd.read_csv('users_comparison.csv', names=['user_id','hits','cms_hits'], sep=';')
df_total_users['difference']=(df_total_users['hits']-df_total_users['cms_hits'])
mape_total_users=sum(np.abs(df_total_users['difference'])/df_total_users['hits'])/len(df_total_users)

print(df_total_users)




df_total_tags=pd.read_csv('tags_comparison.csv', names=['tag','hits','cms_hits'], sep=';')
df_total_tags['difference']=(df_total_tags['hits']-df_total_tags['cms_hits'])
mape_total_tags=sum(np.abs(df_total_tags['difference'])/df_total_tags['hits'])/len(df_total_tags)
print(df_total_tags)

print('The Mean Absolute Percentage Error for users counting between two methods is: ',mape_total_users*100,'%')
print('The Mean Absolute Percentage Error for tags counting between two methods is: ',mape_total_tags*100,'%')

loop= True
while loop==True:
    a=input('Do you want to see a thousand`s results?(Y/N)')
    if a == 'Y':

        j=int(input('Insert number of thousand`s results you want to see:'))
        df1=pd.read_csv(new_path+'\\users'+str(j)+'.csv', names=['user_id','hits','cms_hits'], sep=';')
        print(df1)
        df2 = pd.read_csv(new_path + '\\tags' + str(j) + '.csv', names=['tag', 'hits', 'cms_hits'], sep=';')
        print(df2)
    else:
        loop=False










