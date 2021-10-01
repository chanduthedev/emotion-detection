const path = require("path");
const express = require("express");
const appConfig = require("./appConfig").configuration;

const cors = require("cors");
require("dotenv").config({
  path: path.join(__dirname, ".env"),
});

const app = express();
app.use(cors());

const PORT = process.env.PORT || appConfig.appServer.port;
// add middlewares
app.use(express.static(path.join(__dirname, "..", "build")));
app.use(express.static("public"));

// Static data will be in build folder after build generating
app.use((req, res, next) => {
  res.sendFile(path.join(__dirname, "..", "build", "index.html"));
});

// start express server on port 5055
app.listen(PORT, () => {
  console.log(`server started on port ${PORT}`);
});
