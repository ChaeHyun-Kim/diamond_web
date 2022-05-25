function display_result() {
  data = JSON.parse(localStorage.getItem("predict_result"));
  console.log("결과:", data);
  if (data[1] == "허위") {
    false_result_tag(data[2]);
    for (i = 0; i < data[0]; i++) {
      let false_label = document.createElement("label");
      false_label.innerHTML = data[3][i];
      if (data[4].includes(i)) {
        console.log("허위광고 문장 중 허위", i);
        index_what = data[4].indexOf(i);
        false_sen_tag(data[3][i], data[5][index_what]);
      } else {
        console.log("허위광고 문장 중 허용", i);
        true_sen_tag(data[3][i]);
        //허용문장인 경우
      }
    }
  } else {
    console.log("허용광고");
    true_result_tag(data[2]);
    true_sen_tag(data[6]);
  }
}

function true_sen_tag(sen) {
  console.log("허용문장우악!!!!!!!!!!!!!!!!!!!!!!");
  let sen_box = document.getElementById("sentence_result");
  let senten = document.createElement("label");
  senten.innerHTML = sen;
  senten.setAttribute("class", "true_sen");
  console.log(senten);
  sen_box.appendChild(senten);
}

function false_sen_tag(sen, sen_value) {
  console.log("허위문장이다!!!!!!!!!!!!!!!!!!!!");
  console.log(sen_value);
  let sen_box = document.getElementById("sentence_result");
  let senten = document.createElement("label");
  senten.innerHTML = sen;
  senten.setAttribute("class", "false_sen");
  senten.setAttribute("rel", "tooltip");
  senten.setAttribute(
    "title",
    "해당 문장의 위험도는  " +
      String(Math.round(sen_value * 100, 2)) +
      "입니다."
  );
  console.log(senten);
  sen_box.appendChild(senten);
}
function false_result_tag(result_value) {
  let result_box = document.getElementById("final_result");
  let result = document.createElement("h4");
  result.setAttribute("class", "false_result");
  result.innerHTML =
    "해당 광고는 " + String(result_value) + "의 정확도로 허위·과대광고입니다.";
  result_box.appendChild(result);
}

function true_result_tag(result_value) {
  let result_box = document.getElementById("final_result");
  let result = document.createElement("h4");
  result.setAttribute("class", "true_result");
  result.innerHTML =
    "해당 광고는 " + String(result_value) + "의 정확도로 허용광고입니다.";
  result_box.appendChild(result);
}

display_result();

function restart() {
  location.href = "/restart";
}
