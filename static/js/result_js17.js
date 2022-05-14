window.onload = function () {
  console.log("result.js다");
};
var predict_result;

function href_test() {
  location.href = "/page_render";
  console.log(" 되는거냐1111");
  setTimeout(function () {
    console.log("대라대라아저씨");
  }, 5000);
}

//화면 이동
function append_tag() {
  console.log(" append_tag함수 실행");
  // var targetTag = document.getElementById("sentence_result");
  // var addLabel = document.createElement("p");
  // addLabel.innerHTML = "이거 추가했다 !!!";
  // targetTag.appendChild(addLabel);
  for (var i = 1; i < 10; i++) {
    console.log("되냐고괻");
  }
}
function page_open() {
  href_test(append_tag);
}
//요소 추가
function plus_element() {
  console.log("텍스트 요소 추가");

  var $a = document.createElement("sentence_result");
  $a.value = "123456";
  $a.color = "#00FFFF";
  document.body.appendChild($a);
}

//분석
function predict_please() {
  console.log("문장 분석 시작");
  var sentence =
    "바르기만 해도 살이 빠져요. 혈중 중성지방 수치를 낮추기 위해선 탄수화물의 섭취량을 제한하고 총 지방과 단백질을 적정하게 섭 취하여 표준체중을 유지하여야 합니다.";

  $.ajax({
    url: "/final",
    type: "POST",
    contentType: "application/json",
    dataType: "json",
    data: JSON.stringify({
      ad: sentence,
    }), // converts js value to JSON string
  }).done(function (result) {
    predict_result = result;
    page_open();
    console.log(result);
  });
}
