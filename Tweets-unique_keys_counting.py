import json
import math
import hyperloglog
from pympler.asizeof import asizeof
import numpy as np

users_dict = {}
tags_dict = {}
hll_users = hyperloglog.HyperLogLog(0.05)
hll_tags = hyperloglog.HyperLogLog(0.05)

for i in range(0, 46):
    data = f'tweets.json.{i}'
    with open(data, encoding='utf-8') as json_file:
        for row in json_file:

            json_obj = json.loads(row)  # Make every row from json object to dictionary
            user_id = json_obj['user']['id']
            tags = json_obj['entities']['hashtags']

            for element in tags:
                for key, value in element.items():
                    if key == 'text':
                        tag = str(value)
                        if tag not in tags_dict:  # Add values to the dictionary
                            tags_dict[tag] = 1
                        else:
                            tags_dict[tag] += 1

                        hll_tags.add(str(tag))

            if user_id not in users_dict:  # Add values to the dictionary
                users_dict[user_id] = 1
            else:
                users_dict[user_id] += 1

            hll_users.add(str(user_id))

    json_file.close()

print('Actual cardinality for unique users is:', len(users_dict))
print('Space of actual cardinality approach for unique users is:',asizeof(users_dict),'bytes')

print('Estimated Cardinality for unique users: {0}'.format(math.ceil(hll_users.card())))
print('Space of estimated cardinality approach for unique users is:',asizeof(math.ceil(hll_users.card())),'bytes')

mape_users= (np.abs(len(users_dict)-math.ceil(hll_users.card()))/len(users_dict))/2

print('Actual cardinality for tags is:', len(tags_dict))
print('Space of actual cardinality approach for tags is:',asizeof(tags_dict), 'bytes')

print('Estimated Cardinality for tags: {0}'.format(math.ceil(hll_tags.card())))
print('Space of estimated cardinality approach for tags is:',asizeof(math.ceil(hll_tags.card())),'bytes')

mape_tags= (np.abs(len(tags_dict)-math.ceil(hll_tags.card()))/len(tags_dict))/2

print('The Mean Absolute Percentage Error for users counting between two methods is:',mape_users*100,'%')
print('The Mean Absolute Percentage Error for tags counting between two methods is: ',mape_tags*100,'%')