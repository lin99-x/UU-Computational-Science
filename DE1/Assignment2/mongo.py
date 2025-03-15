import pymongo
import json

client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")

db = client['tweets']
collection = db.tweet

### here I try to delete all retweeted tweets but it seems like take a lot of time.
### so I didn't finish the process.
# documents = collection.find({"retweeted_status":{"$exists":True}})

# for document in documents:
#     collection.delete_one({"_id":document["_id"]})


##########
# The queries I used to search the result:
# db.getCollection('tweet').find({'text':/.*\bdenna\b.*/i},{"retweeted_status":{$exists:false}}).count()
# db.getCollection('tweet').find({'text':/.*\bden\b.*/i},{"retweeted_status":{$exists:false}}).count()
# db.getCollection('tweet').find({'text':/.*\bdet\b.*/i},{"retweeted_status":{$exists:false}}).count()
# db.getCollection('tweet').find({'text':/.*\bhan\b.*/i},{"retweeted_status":{$exists:false}}).count()
# db.getCollection('tweet').find({'text':/.*\bhen\b.*/i},{"retweeted_status":{$exists:false}}).count()
# db.getCollection('tweet').find({'text':/.*\bhon\b.*/i},{"retweeted_status":{$exists:false}}).count()
# db.getCollection('tweet').find({'text':/.*\bdenne\b.*/i},{"retweeted_status":{$exists:false}}).count()

#########
# The result I got:
# denne 6665
# hon   396532
# hen   46321
# han   834277
# det   594561
# denna 31683
# den   1623354
# unique_tweets 2341577