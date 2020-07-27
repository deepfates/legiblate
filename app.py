import streamlit as st
import pandas
import requests
import json
from slugify import slugify
from fastai.vision import *
from io import BytesIO

# Suppress streamlit warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

# Cached functions
@st.cache(allow_output_mutation=True)
def load_model(path):
    learn = load_learner(path)
    return(learn)

@st.cache
def infer(img, learn):
    pred_class,pred_idx,outputs = learn.predict(img)
    probs = outputs.squeeze().tolist()
    cats = zip(probs, learn.data.classes)
    lcat = list(sorted(cats, reverse=True))
    return(lcat)

@st.cache
def search(query):
    req = f"http://openlibrary.org/search.json?q={slugify(query)}"
    response = requests.get(req)
    return(response.json())

@st.cache
def get_cover(cover_i, size="M"):
    req = f"http://covers.openlibrary.org/b/id/{cover_i}-{size}.jpg"
    response = requests.get(req)
    return(response.content)

@st.cache
def predict(img):
        lcat = infer(img, learn)
        ldict = {cat[1]: cat[0] for cat in lcat}
        df = pandas.DataFrame.from_dict(ldict, orient='index')
        df *= 100
        return df
# App logic
## Preparation
path = '.'
learn = load_model(path)


st.title("LEGIBLATE")
st.write("""
## Use AI to guess a book's genre from its cover!

Upload an image, or search through the OpenLibrary database.

The AI has seen 57,000 book covers in its training. Newer books will be harder for it to guess, but it will do its best.

Try uploading images that aren't books -- see what kind of covers they could become!

""")
st.subheader("Get an image:")

getter = st.radio("", ["Upload", "Search"])

if getter == "Upload":
    upload = st.file_uploader("")
    if upload:
        st.image(upload, width=700)
        img = open_image(upload)
        df = predict(img)


if getter == "Search":
    query = st.text_input("Keywords")
    if query:
        options = []
        answer = search(query)
        for doc in answer["docs"][:5]:
            if "cover_i" in doc:
                image = get_cover(doc["cover_i"])
                opt = {"title": doc["title"], "cover_i": doc["cover_i"], "image": image}
                options.append(opt)
        op = pandas.DataFrame(options, columns=["title", "cover_i", "image"])
        images = list(op["image"])
        labels = list(op["cover_i"])
        st.image(images, labels)

        sel =  st.radio("Which book?", (op["cover_i"]))
        cover = get_cover(sel, "L")
        st.image(cover)
        img = open_image(BytesIO(cover))

try:
    st.subheader("Prediction:")
    df = predict(img)
    st.write(f"# {df.index[0]}")
    st.table(df.style.background_gradient())
except:
    st.write("Choose an image first")

st.write("""## Credits
This demo uses a resnet34 model trained on the corpus [Judging a Book by its Cover](https://arxiv.org/abs/1610.09204) (2016).
The authors achieved 24.7% accuracy. I believe mine to be a state-of-the-art result, with a 32.8% accuracy, in 2020.

Built with [fast.ai](https://www.fast.ai) and [Streamlit](https://www.streamlit.io). Book information and cover images from the [OpenLibrary API](https://openlibrary.org/developers/api).

Contact [Robot Face](mailto://robotfaceai@gmail.com) with questions or comments.
""")
