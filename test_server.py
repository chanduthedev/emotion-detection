import pytest
import server
import copy
import json

baseQueryOriginal = {"text": "I am good", "model": "lr"}


@pytest.fixture
def client():
    server.app.config["TESTING"] = True
    client = server.app.test_client()
    yield client


def test_hello(client):
    x = client.get("/")
    assert (
        b"This emotion detection API is REST-ful and supports CORS, so it can also be used in web browsers."
        in x.data
    ), "Looks like the Skeleton is broken"


def test_text_required(client):
    baseQuery = copy.deepcopy(baseQueryOriginal)
    baseQuery.update({"text": None})
    x = client.post("/predict?", query_string=baseQuery)
    assert 400 == x.status_code, "text is a required paramater"


def test_no_model_param(client):
    baseQuery = copy.deepcopy(baseQueryOriginal)
    baseQuery.update({"model": None})
    x = client.get("/predict?", query_string=baseQuery)
    resp_data = json.loads(x.data)
    assert resp_data["emotion"] == "joy", "should return joy"
    assert resp_data["model"] == "LR", "should return LR"
    assert resp_data["text"] == "I am good", "should return 'I am good'"
    assert 200 == x.status_code, " Should be able to predict emotion"


def test_lr_model_param(client):
    baseQuery = copy.deepcopy(baseQueryOriginal)
    baseQuery.update({"model": "lr"})
    x = client.get("/predict?", query_string=baseQuery)
    resp_data = json.loads(x.data)
    assert resp_data["emotion"] == "joy", "should return joy"
    assert resp_data["model"] == "lr", "should return lr"
    assert resp_data["text"] == "I am good", "should return 'I am good'"
    assert 200 == x.status_code, " Should be able to predict emotion"


def test_svm_model_param(client):
    baseQuery = copy.deepcopy(baseQueryOriginal)
    baseQuery.update({"model": "svm"})
    x = client.get("/predict?", query_string=baseQuery)
    resp_data = json.loads(x.data)
    assert resp_data["emotion"] == "joy", "should return joy"
    assert resp_data["model"] == "svm", "should return svm"
    assert resp_data["text"] == "I am good", "should return 'I am good'"
    assert 200 == x.status_code, " Should be able to predict emotion"


def test_nb_model_param(client):
    baseQuery = copy.deepcopy(baseQueryOriginal)
    baseQuery.update({"model": "nb"})
    x = client.get("/predict?", query_string=baseQuery)
    resp_data = json.loads(x.data)
    assert resp_data["emotion"] == "fear", "should return fear"
    assert resp_data["model"] == "nb", "should return nb"
    assert resp_data["text"] == "I am good", "should return 'I am good'"
    assert 200 == x.status_code, " Should be able to predict emotion"


def test_rf_model_param(client):
    baseQuery = copy.deepcopy(baseQueryOriginal)
    baseQuery.update({"model": "rf"})
    x = client.get("/predict?", query_string=baseQuery)
    resp_data = json.loads(x.data)
    assert resp_data["emotion"] == "fear", "should return fear"
    assert resp_data["model"] == "rf", "should return rf"
    assert resp_data["text"] == "I am good", "should return 'I am good'"
    assert 200 == x.status_code, " Should be able to predict emotion"
