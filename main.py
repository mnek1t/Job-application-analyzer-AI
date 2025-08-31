from modules import (extract_data, embeddings)
from fastapi import FastAPI, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.encoders import jsonable_encoder
# from sklearn.feature_extraction.text import CountVectorizer

app = FastAPI()
model = embeddings.get_model('all-MiniLM-L6-v2')


@app.post("/match")
def match_cv_to_job(cv: UploadFile, job: UploadFile):
    print(cv.file)
    try:
        # export CV data
        cv_data = extract_data.extract_data_from_pdf(
            cv.file  # "./data/Mykyta_Medvediev_CV_SF.pdf"
        )
        cv_data = extract_data.remove_special_characters(cv_data)
        cv_data = extract_data.tokenize_text(cv_data)
        # cv_data = extract_data.lemmatize_text(cv_data)

        # export Job description data
        jb_data = extract_data.extract_data_from_pdf(
            job.file  # "./data/salesforce_developer.pdf"
        )
        jb_data = extract_data.remove_special_characters(jb_data)
        jb_data = extract_data.tokenize_text(jb_data)
        # jb_data = extract_data.lemmatize_text(jb_data)

        # embed into vector db

        cv_embed = embeddings.embed_text(model, [cv_data])[0]
        jb_embed = embeddings.embed_text(model, [jb_data])[0]
        similarity_matrix = embeddings.compare_similarity(
            model,
            cv_embed,
            jb_embed
        )

        # Calculate overall similarity score by mean
        score = similarity_matrix.mean().item() * 100
        print(
            f"Overall similarity score is: {score}"
        )
        return {
            "similarity_score": score
        }
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
