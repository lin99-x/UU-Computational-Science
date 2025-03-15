import requests
import pulsar
import operator

def CountFrequency(list):
    freq = {}
    for item in list:
        freq[item] = list.count(item)
    return freq

token = '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'
# should write a query
query = 'created: 2022-01-01 per_page: 100 page: 1'


t_url = 'https://api.github.com/search/repositories?q=created:2022-01-01&per_page=100&page=1'

headers = {'Authorization': 'Bearer ' + token}
test = requests.get(t_url, headers=headers)
test_dict = test.json()
print("Status_code: ", test.status_code)

pulsar_ip = "localhost"
client = pulsar.Client(f'pulsar://{pulsar_ip}:6650')
consumer = client.subscribe('Q4URLs', subscription_name='Q4-sub')
producer = client.create_producer('Q4Result')

while True:
    msg = consumer.receive()
    try:
        url = msg.data().decode('utf-8')
        # get the repo name
        repo_name = url.split("repos/")[1].split("/actions")[0]
        response = requests.get(url, headers = headers)
        # check if the repo has workflows
        if response.json()['workflows'] != []:
            print(f"{repo_name}:True")
            has_workflows = True
            producer.send(f"{repo_name}:{has_workflows}").encode('utf-8')
            consumer.acknowledge(msg)
        else:
            has_workflows = False
            print(f"{repo_name}:False")
            producer.send(f"{repo_name}:{has_workflows}").encode('utf-8')
            consumer.acknowledge(msg)
            
    except Exception as e:
        print(e)
        consumer.negative_acknowledge(msg)
        

# languages = []
# # count languages used in repos
# for repo in repo_has_workflows:
#     languages.append(repo['language'])
    
# frequency = CountFrequency(languages)
# print(frequency)
# frequency_sort = dict(sorted(frequency.items(), key = lambda ele: ele[1], reverse=True))
# print(frequency_sort)

# # print top 10 languages
# print("Top 10 programming languages that follow the test-driven development: ")
# print(list(frequency_sort.keys())[0:10])

