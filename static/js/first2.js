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
    console.log(result);
    localStorage.setItem("predict_result", JSON.stringify(result));
    console.log("test:", JSON.parse(localStorage.getItem("predict_result")));
    location.href = "/result_render";
  });
}
