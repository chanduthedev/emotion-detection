import "./App.css";
import { useState, useEffect } from "react";
import TextField from "@mui/material/TextField";
import Table from "./Table";
import Button from "@mui/material/Button";
import {
  predictService,
  getTrainedModelService,
} from "./services/serviceCalls";

function App() {
  useEffect(() => {
    getModels();
  }, []);
  const [inputText, setInputText] = useState("");
  const [modelName, setModelName] = useState("nb");
  const [emotion, setEmotion] = useState("");
  const [statusCode, setStatusCode] = useState(200);
  const [trainedData, setTrainedData] = useState("");
  const [trainedModels, setTrainedModels] = useState([
    { modelName: "Linear Regression", accuracy: 83.67 },
    { modelName: "Naive Bayes", accuracy: 74.27 },
    { modelName: "Support Vector Machines", accuracy: 73.16 },
    { modelName: "Random Forest", accuracy: 83.27 },
  ]);
  const columns = [
    {
      Header: "Model Name",
      accessor: "modelName",
    },
    {
      Header: "Accuracy % when training the model.",
      accessor: "accuracy",
    },
  ];

  async function predict() {
    const respData = await predictService(inputText, modelName);
    console.log(respData);
    setStatusCode(respData.code);
    if (respData.code === 200) {
      setEmotion(respData.emotion);
    } else {
      setEmotion(respData.message);
    }
  }
  async function getModels() {
    console.log("calling getModels");
    const respData = await getTrainedModelService();

    if (respData) {
      console.log(respData.trainedModels);
      console.log(respData.trainedData);
      setTrainedModels(respData.trainedModels);
      setTrainedData(respData.trainedData);
    } else {
      setTrainedModels([]);
      setTrainedData("");
    }
  }

  return (
    <div className="App">
      <h1>Emotion Prediction Application</h1>
      <h4>Welcome to Emotion Prediction Application.</h4>
      <h4>
        This application will predict emotion of the given text basing on the
        trained Machine Learning model you selected.
      </h4>
      <h4>Trained emotions are as below</h4>
      <div
        style={{
          textAlign: "center",
          listStyle: "inside",
          fontWeight: "bold",
        }}
      >
        <list>
          <ul>Joy</ul>
          <ul>Sadness</ul>
          <ul>Fear</ul>
          <ul>Anger</ul>
        </list>
      </div>

      <div>
        <TextField
          placeholder="Type your text here.
          e.g My blood is boiling."
          multiline
          maxRows={4}
          value={inputText}
          onChange={(e) => {
            setInputText(e.target.value);
          }}
          style={{
            color: "red",
            fontSize: "20px",
            marginBottom: "10px",
            width: "40%",
          }}
        />
      </div>
      <select
        onChange={(e) => {
          setModelName(e.target.value);
        }}
        style={{
          color: "black",
          fontSize: "25px",
          padding: "5px",
          marginBottom: "10px",
          backgroundColor: "transparent",
        }}
      >
        <option value="lr" selected>
          Linear Regression
        </option>
        <option value="nb">Naive Bayes</option>
        <option value="svm">Support Vector machines</option>
        <option value="rf">Random Forest</option>
      </select>
      <div>
        <Button
          variant="contained"
          onClick={predict}
          style={{
            fontSize: "20px",
            marginBottom: "10px",
          }}
        >
          Predict Emotion
        </Button>
      </div>
      <div
        style={{
          color: "green",
          fontSize: "30px",
        }}
      >
        <label>{emotion}</label>
      </div>
      <div className="Table">
        <label
          style={{
            color: "black",
            fontSize: "30px",
            marginBottom: "10px",
          }}
        >
          I created below models by training emotion data from
          <a href={trainedData}> here.</a>{" "}
        </label>
        <Table modelData={trainedModels} colNames={columns} />
      </div>
    </div>
  );
}

export default App;
