const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
require("dotenv").config();

const aiRoute = require("./routes/ai");

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(bodyParser.json());

app.use("/api", aiRoute);

app.listen(PORT, () => {
  console.log(`AI server running on http://localhost:${PORT}`);
});
