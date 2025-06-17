import os 
import json 


# /exp/amartin/scale24/info/internvid/all_jsons/s9BbJUljUW0.info.json
with open('/exp/amartin/scale24/info/internvid/all_jsons/s9BbJUljUW0.info.json', 'r') as f:
    data = json.load(f)

print(data.keys())

print(data['title'])
print(data['resolution'])
print(data['fps'])
print(data['description'])
print(data['tags'])
print(data['was_live'])
print(data['filesize_approx'])
print(data['language'])
print(data['categories'])