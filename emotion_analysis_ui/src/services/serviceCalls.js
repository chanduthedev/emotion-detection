async function predictService(inputText, modelName) {
  const apiEndPoint = `https://chemotionanalysis.herokuapp.com/predict?text=${inputText}&model=${modelName}`;
  const serverResp = await fetch(apiEndPoint, {
    method: "GET",
  })
    .then(async (response) => {
      let respData = await response.json();
      respData.code = response.status;
      return respData;
    })
    .catch((err) => {
      console.log(err);
      return err;
    });
  return serverResp;
}

async function getTrainedModelService() {
  const apiEndPoint = `https://chemotionanalysis.herokuapp.com/models`;
  // const apiEndPoint = `http://192.168.10.134:9898/models`;
  const serverResp = await fetch(apiEndPoint, {
    method: "GET",
  })
    .then(async (response) => {
      let respData = await response.json();
      respData.code = response.status;
      return respData;
    })
    .catch((err) => {
      console.log(err);
      return err;
    });
  return serverResp;
}

export { predictService, getTrainedModelService };
