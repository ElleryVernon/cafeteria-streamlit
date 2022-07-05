import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import koreanize_matplotlib

import os
from zipfile import ZipFile
import gdown

import app1
import app2


BASE_PATH = "data/cafeteria"
FILE_NAME = "cafeteria.zip"
DOWNLOAD_ROOT = "https://drive.google.com/uc?export=download&id="
GDRIVE_ID = "1WNqV8300sC_q7jl11Tu3Ed3qlHvTIm5P"
URL = DOWNLOAD_ROOT + GDRIVE_ID


def fetch_data(base_path=BASE_PATH, file_name=FILE_NAME, url=URL):
    PATH = os.path.join(base_path, file_name)
    if os.path.exists(PATH):
        print(f"file already exist(path: {os.getcwd()}/{base_path})")
    else:
        os.makedirs(base_path, exist_ok=True)
        gdown.download(url, PATH, quiet=False)
        with ZipFile(PATH, "r") as zip_data:
            zip_data.extractall(base_path)


def load_data(base_path=BASE_PATH):
    train_path = os.path.join(base_path, "train.csv")
    test_path = os.path.join(base_path, "test.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)


fetch_data()
train, test = load_data()

PAGES = {
    "EDA": app1,
    "Deep Learning": app2,
}

st.set_page_config(layout="wide")
selection = st.sidebar.radio("페이지를 선택해주세요.", list(PAGES.keys()))
page = PAGES[selection]
page.app(train, test)
