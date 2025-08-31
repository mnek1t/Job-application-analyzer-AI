from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from typing import Union, BinaryIO
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


def extract_data_from_pdf(pdf_path: Union[str, BinaryIO]) -> str:
    reader = PdfReader(pdf_path, True)
    text = ""
    for page in reader.pages:
        extracted_text = str(
            page.extract_text()
                .encode('ascii', 'ignore')  # encode and decode to remove non-ascii characters like /xe2 etc.
                .decode()
        )
        # extracted_text = remove_special_characters(extracted_text)
        text = "".join([text, extracted_text])
    return text


# Implement remove special characters like \n, \t, etc.
def remove_special_characters(text: str) -> str:
    return text.replace('\\n', '').strip()
# Implement parsing of contracdictions: ain't -> are not, etc.


# Implement tokenization excluding delimeters like . , ; : ? ! etc.
def tokenize_text(text: str) -> list:
    tokenized_list = word_tokenize(text)
    return [word for word in tokenized_list if word.isalnum()]


# implement stemming or+ lemmatization
def lemmatize_text(list_of_words: list) -> list:
    sb = SnowballStemmer("english")
    return [sb.stem(word) for word in list_of_words]