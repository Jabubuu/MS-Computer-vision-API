import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient as vision
from msrest.authentication import CognitiveServicesCredentials as credentials
import pandas as pd
import pathlib

subscription_key = "subscription_key"
endpoint = "endpoint-url"

local_image = 'file path name here.jpg'
filename = os.path.basename(local_image)
csv_file = "microsoft-results.csv"
file = pathlib.Path(csv_file)

def save_csv(df_new):
    if file.exists ():
        try:
            df_from_csv = pd.read_csv(csv_file)
            frames = [df_from_csv, df_new]
            df_edited = pd.concat(frames)
            df_edited.to_csv(csv_file, index=False)

        except pd.errors.EmptyDataError:
            df_new.to_csv(csv_file, index=False)

    else:
        print('No file, creating new')
        df_new.to_csv(csv_file, index=False)

def tags(client):
    image = open(local_image, "rb")
    response = client.tag_image_in_stream(image)
    tags = [tag.name for tag in response.tags]
    return tags

def objects(client):
    image = open(local_image, "rb")
    response = client.detect_objects_in_stream(image)
    objects = [obj.object_property for obj in response.objects]
    return objects    

def describe(client):
    image = open(local_image, "rb")
    response = client.describe_image_in_stream(image)
    text = [caption.text for caption in response.captions]
    return text

def category(client): 
    image = open(local_image, "rb")
    features = ["categories"]
    response = client.analyze_image_in_stream(image, features)
    categories = [category.name for category in response.categories]       
    return categories    

def logos(client):
    image = open(local_image, "rb")
    features = ["brands"]
    response = client.analyze_image_in_stream(image, features)
    logos = [brand.name for brand in response.brands]
    return logos

def adult(client):
    image = open(local_image, "rb")
    features = ["adult"]
    response = client.analyze_image_in_stream(image, features)
    adult = response.adult.is_adult_content
    return adult

def faces(client):
    image = open(local_image, "rb")
    features = ["faces"]
    response = client.analyze_image_in_stream(image, features)
    face_count = len(response.faces)
    return face_count

def domain(client):  
    image = open(local_image, "rb")
    response = client.analyze_image_by_domain_in_stream("landmarks", image)
    landmarks = [landmark["name"] for landmark in response.result["landmarks"]]
    return landmarks

def texts(client):
    image = open(local_image, "rb")
    response = client.recognize_printed_text_in_stream(image)
    texts = [word.text for region in response.regions for line in region.lines for word in line.words]
    return texts    

def main():
    client = vision(endpoint, credentials(subscription_key))
    df = pd.DataFrame()
    df['Filename'] = [filename]
    df['Tags'] = [tags(client)]
    df['Objects'] = [objects(client)]
    df['Describe'] = [describe(client)]
    df['Category'] = [category(client)]
    df['Logos'] = [logos(client)]
    df['Faces'] = [faces(client)]
    df['Landmarks'] = [domain(client)]
    df['Texts'] = [texts(client)]
    save_csv(df)
    
if __name__ == "__main__":
    main()
