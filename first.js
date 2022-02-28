const http = require("http");
const express = require("express");
const fs = require("fs");
const app = express();
const server = http.createServer(app);
const PORT = 3000;
app.use(express.static("public"));

app.get("/", (req, res) => {
  fs.readFile(`templates/index.html`, (error, data) => {
    if (error) {
      console.log(error);
      return res.status(500).send("<h1>500 Error</h1>");
    }
    res.writeHead(200, { "Content-Type": "text/html" });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`Server running on ${PORT}`);
});
