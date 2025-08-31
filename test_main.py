from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_post_cv_file_success():
    with (
        open("./data/Mykyta_Medvediev_CV_SF.pdf", "rb") as cv_file,
        open("./data/salesforce_developer.pdf", "rb") as job_file
    ):
        response = client.post(
            "/match",
            files={
                "cv": ("cv.pdf", cv_file, "application/pdf"),
                "job": ("job.pdf", job_file, "application/pdf")
            },
        )

    assert response.status_code == 200
    assert "similarity_score" in response.json()
