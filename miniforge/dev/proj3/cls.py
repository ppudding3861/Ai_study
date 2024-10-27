

# step 1 : import module
from transformers import pipeline


# step 2 : create inference object(instance)
classifier = pipeline("sentiment-analysis", model="uget/sexual_content_dection")

# step 3 : prepare data
text = "Tiffany Doll - Wine Makes Me Anal (31.03.2018)_1080p.mp4"


# step 4 : inferece
result = classifier(text)

# step 5 : post processing
print(result)