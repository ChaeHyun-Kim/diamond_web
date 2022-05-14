data = JSON.parse(localStorage.getItem("predict_result"));
console.log(data);

var targetTag = document.getElementById("sentence_result");
var addLabel = document.createElement("p");
addLabel.innerHTML = "이거 추가했다 !!!";
targetTag.appendChild(addLabel);

function restart() {
  location.href = "/restart";
}
