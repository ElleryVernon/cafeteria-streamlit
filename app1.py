import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def draw_missing_values_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = total / len(df) * 100
    dtypes = df.dtypes
    df_missing = pd.concat(
        [total, percent, dtypes], axis=1, keys=["Count", "Percent", "Type"]
    )
    return df_missing


def app(train, test):
    st.write("""## 구내식당 식수 인원 예측 AI 경진대회""")
    st.write("""https://dacon.io/competitions/official/235743/overview/description""")
    st.write("""---""")
    st.write("""### 1. 데이터 다운하기""")
    st.write(
        """일반적으로 회사에 속하게 된 이후에 다루게 될 데이터는 관계형 데이터베이스에 들어있고 여러 테이블이나 엑셀과 같은 파일에 나누어져 있을 것입니다. 데이터에 접근하기 위해서는 접근 권한도 있어야하고 데이터에 대한 법적 제약까지도 검토해야하죠. 하지만 지금 저희가 진행하고 있는 프로젝트는 간단합니다. 데이콘에서 진행했던 실제 대회 기반의 프로젝트이기 때문에 모든 데이터가 들어있는 **CSV(comma-seprarated-value)** 파일이 들어있는 zip 파일만 구글 드라이브에서 다운받으면 됩니다."""
    )
    st.write(
        """물론 직접 데이콘 사이트에 가서 파일을 다운로드 받아서 압축을 풀고 프로젝트 디렉토리로 옮길 수도 있지만, 우리는 프로그래머입니다. 따라서 간단한 함수를 작성해서 이 과정을 자동화 할 수 있죠! 데이터를 다운로드해서 프로젝트 디렉토리로 옮기는 과정을 자동화하면 편합니다. 데이터가 정기적으로 변경되거나 수정사항이 생겨서 데이터를 업데이트해야 할 때 특히 편리해집니다."""
    )
    st.write("""다음은 우리가 진행해볼 대회의 자료가 들어있는 데이콘 구글 드라이브에서 데이터를 받아오는 함수입니다.""")
    code = """import os
from zipfile import ZipFile
# !pip install -U gdown
import gdown
    
BASE_PATH = "data/cafeteria"
FILE_NAME = "cafeteria.zip"
DOWNLOAD_ROOT = "https://drive.google.com/uc?export=download&id="
GDRIVE_ID = "1WNqV8300sC_q7jl11Tu3Ed3qlHvTIm5P"
URL = DOWNLOAD_ROOT + GDRIVE_ID

def fetch_data(base_path=BASE_PATH, gdrive_id=GDRIVE_ID, file_name=FILE_NAME, url=URL):
    PATH = os.path.join(base_path, file_name)
    if os.path.exists(PATH):
        print(f"file already exist(path: {os.getcwd()}/{base_path})")
    else:
        os.makedirs(base_path, exist_ok=True)
        gdown.download(url, PATH, quiet=False)
        with ZipFile(PATH, 'r') as zip_data:
            zip_data.extractall(base_path)\n\nfetch_data()"""

    st.code(code, language="python")
    st.write("""""")
    st.write(
        """**fetch_data** 함수를 호출하게 되면 **data/cafeteria** 디렉터리를 만들어 **cafeteria.zip** 파일을 다운받고 압축을 해제해 **train.csv**와 **test.csv** 파일을 만듭니다."""
    )
    st.write("""---""")

    st.write("""### 2. 데이터 로드하기""")
    st.write("""데이터를 다운해왔으니 판다스를 이용하여 데이터를 로드해보겠습니다. 데이터를 로드하는 간단한 함수도 하나 만들겠습니다.""")
    st.write("""다음은 **train.csv**와 **test.csv** 를 읽어들여 2개의 데이터프레임 객체를 반환하는 함수입니다.""")

    code = """def load_data(base_path=BASE_PATH):
    train_path=os.path.join(base_path, "train.csv")
    test_path=os.path.join(base_path, "test.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)"""
    st.code(code, language="python")

    st.write("""---""")

    st.write("""### 3. 데이터 구조 살펴보기""")
    st.write("""DataFrame의 head() 메서드를 사용하여 train, test의 첫 5개 레코드를 확인해보겠습니다.""")

    code = """
    train, test = load_data()\ndisplay(train.head()), display(test.head())
    """
    st.code(code, language="python")

    st.write("""""")

    col1, col2 = st.columns(2)
    with col1:
        st.write('`"Train" Dataframe Head`')
        st.dataframe(train.head())
    with col2:
        st.write('`"Test" Dataframe Head`')
        st.dataframe(test.head())

    st.write("""""")
    st.write(
        """각 행은 하루 단위의 데이터를 나타내는 것을 알 수 있습니다. 공통으로 가지고 있는 특성은 **일자**, **요일**, **본사정원수**, **본사휴가자수**, **본사출장자수**, **시간외근무명령서승인건수**, **현본사소속재택근무자수**, **조식메뉴**, **중식메뉴**, **석식메뉴**  등 총 9개입니다. (전부 보이지 않는다면 스크롤해서 확인할 수 있습니다.)"""
    )
    st.write(
        """
    `train` 에만 있는 특성은 **중식계**, **석식계**등 총 2개로 해당 특성들은 `test` 에서 예측해내야할 타켓 변수입니다.
    """
    )
    st.write("""---""")

    st.write("""#### 3.1 결측치 및 데이터 타입""")
    st.write(
        """우리가 분석할 데이터의 결측치 및 데이터 타입을 확인하기 위해 해당하는 내용들을 데이터프레임으로 만들어서 반환해주는 `draw_missing_values_table` 함수를 작성하겠습니다."""
    )
    code = """def draw_missing_values_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = total / len(df) * 100
    dtypes = df.dtypes
    df_missing = pd.concat(
        [total, percent, dtypes], axis=1, keys=["Count", "Percent", "Type"]
    )
    return df_missing"""
    st.code(code, language="python")
    st.write("""""")
    st.write("""* **실행결과**""")
    col3, col4 = st.columns(2)
    with col3:
        st.write("`train 데이터에는 결측치가 없어보입니다.`")
        st.dataframe(draw_missing_values_table(train).astype(str))
    with col4:
        st.write("`test 데이터에는 결측치가 없어보입니다.`")
        st.dataframe(draw_missing_values_table(test).astype(str))
    st.write(
        """데이터셋에 결측치는 딱히 없어보입니다. `일자`, `요일`, `조식메뉴`, `중식메뉴`, `석식메뉴`는 데이터 타입이 `object`로 되어있습니다. 물론 파이썬의 모든 객체 중 하나가 될 수도 있지만, CSV 파일에서 읽어온 데이터이기 때문에 텍스트 데이터일 것으로 보입니다."""
    )
    st.write("""---""")

    st.write("""#### 3.2 요약 정보 확인하기""")
    st.write("""모든 필드를 살펴보도록 하겠습니다.`describe` 메서드는 숫자형 특성의 요약 정보를 출력해줍니다.""")
    st.write("""""")
    code = """train.describe(), test.describe()"""
    st.code(code, language="python")
    st.write("""""")
    col5, col6 = st.columns(2)
    with col5:
        st.write("`train 데이터 요약 정보`")
        st.dataframe(train.describe().T)
    with col6:
        st.write("`test 데이터 요약 정보`")
        st.dataframe(test.describe().T)
    st.write("""""")
    st.write(
        """`count`, `mean`, `min`, `max` 컬럼이 가지고 있는 데이터들은 이름 그대로의 의미를 가지고 있습니다 (결측치는 자동적으로 제외됩니다.). `std` 컬럼은 값이 퍼져있는 정도를 알려주는 표준편차를 의미합니다. `25%`, `50%`, `75%` 컬럼은 백분위수([Percentile](https://en.wikipedia.org/wiki/Percentile))를 의미합니다."""
    )
    st.write("""---""")

    st.write("""#### 3.3 수치형 데이터 그래프로 확인하기""")

    st.write(
        """데이터셋을 빠르게 살펴보는 또 다른 방법은 수치형 데이터들을 그래프로 그려보는 것입니다. `hist` 메서드는 수치형 데이터들을 그래프로 그려줍니다."""
    )
    st.write("""""")
    code = """train.hist(bins=50, figsize=(12, 8))\nplt.suptitle("Train 데이터 히스토그램")\nst.pyplot(plt.show())
\n\ntest.hist(bins=50, figsize=(12, 8))\nplt.suptitle("test 데이터 히스토그램")\nst.pyplot(plt.show()))"""
    st.code(code, language="python")
    st.write("""""")
    col7, col8 = st.columns(2)
    with col7:
        train.hist(bins=50, figsize=(12, 8))
        plt.suptitle("Train 데이터 히스토그램")
        st.pyplot(plt.show())
    with col8:
        test.hist(bins=50, figsize=(12, 8))
        plt.suptitle("test 데이터 히스토그램")
        st.pyplot(plt.show())

    st.write("""""")
    st.write("""우리는 방금 그려본 히스토그램에서 다음과 같은 사실들을 살펴볼 수 있습니다.""")
    st.write("""1. 우선 모든 데이터가 따로 스케일이 되어 있지 않았음을 추측해볼 수 있습니다.""")
