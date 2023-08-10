
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import PIL.Image
import urllib.request

import numpy as np
import pandas as pd


# Define the data structures
class Tweet:
    def __init__(self, id: int, authorId: int, content: int):
        self.id = id
        self.authorId = authorId
        self.content = content

class Person:
    def __init__(self, id: int, name: int, pfp: str, bio: str):
        self.id = id
        self.name = name
        self.pfp = pfp
        # Check if the pfp value is a URL or a local path
        if pfp.startswith("http://") or pfp.startswith("https://"):
            with urllib.request.urlopen(pfp) as url:
                image = PIL.Image.open(url)
                self.pfp_resolved = np.array(image)
        else:
            self.pfp_resolved = plt.imread(pfp)
        self.bio = bio
        self.tweets: list[Tweet] = []


json_file_path = "./profiles_mock.json"

# Read data from the .json file
with open(json_file_path, 'r') as file:
    loaded_data = json.load(file)

# Populate the data structures
persons = {}
tweets = []

# Create Tweet objects
for idx, tweet_data in enumerate(loaded_data["tweets"], 1):
    tweet = Tweet(id=idx, authorId=tweet_data["authorId"], content=tweet_data["content"])
    tweets.append(tweet)

# Create Person objects and associate tweets with them
for profile_data in loaded_data["profiles"]:
    person = Person(id=profile_data["id"], name=profile_data["name"], pfp=profile_data["pfp"], bio=profile_data["bio"])
    person.tweets = [tweet for tweet in tweets if tweet.authorId == person.id]
    persons[person.id] = person



# Extracting tweet content from the updated Tweet objects
documents = [tweet.content for tweet in tweets]

vectorizer = CountVectorizer()
transformer = TfidfTransformer()

# Using CountVectorizer to get the term frequency matrix
X = vectorizer.fit_transform(documents)

# Convert term frequency matrix to a dense numpy array representation
dense_X = X.toarray()

# Using TfidfTransformer to convert term frequency matrix to tf-idf representation
tfidf_matrix = transformer.fit_transform(dense_X)



# Number of topics/components
num_topics = 2

# Apply LSA (i.e., Truncated SVD)
lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)
lsa_topic_matrix = lsa_model.fit_transform(tfidf_matrix)

lsa_topic_matrix.shape

# Compute aggregated LSA representation for each person
aggregated_lsa = {}
for person_id, person in persons.items():
    tweet_indices = [tweet.id - 1 for tweet in person.tweets]  # -1 since index starts from 0
    aggregated_lsa[person_id] = np.mean(lsa_topic_matrix[tweet_indices], axis=0)

# Compute cosine similarity between the aggregated representations
similarity_matrix = cosine_similarity(list(aggregated_lsa.values()))

# Constructing the graph
G = nx.Graph()
for i, person1 in enumerate(persons.values()):
    for j, person2 in enumerate(persons.values()):
        if i < j:  # To avoid duplicate edges and self-loops
            G.add_edge(person1.name, person2.name, weight=similarity_matrix[i, j])

# Plot the graph
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, 'weight')
rounded_labels = {k: round(v, 2) for k, v in labels.items()}  # Round the weights for better visualization

# Custom function to plot node (with image)
def plot_node_with_image(node, position, image_data, ax, node_size):
    im = ax.imshow(image_data, aspect='auto', zorder=0, extent=(
        position[0] - node_size,
        position[0] + node_size,
        position[1] - node_size,
        position[1] + node_size
    ))
    return im

# Use the customized node plotting
ax = plt.gca()
for node in G.nodes():
    person = [p for p in persons.values() if p.name == node][0]
    plot_node_with_image(node, pos[node], person.pfp_resolved, ax, 0.1) # Adjust node_size as needed

# Draw edges and edge labels
nx.draw_networkx_edges(G, pos, width=3, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=15)
nx.draw_networkx_edge_labels(G, pos, edge_labels=rounded_labels, font_size=12)

plt.title("Similarity Graph between Persons based on Tweets")
plt.axis('off')  # Turn off the axis
plt.show()