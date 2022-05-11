function predict_please() {
  // alert("123");
  console.log("왜 안되는거야!!");
  var sentence = {
    ad: "바르기만 해도 살이 빠져요. 혈중 중성지방 수치를 낮추기 위해선 탄수화물의 섭취량을 제한하고 총 지방과 단백질을 적정하게 섭 취하여 표준체중을 유지하여야 합니다.",
  };

  $.ajax({
    url: Flask.url_for("final"),
    type: "POST",
    data: JSON.stringify(sentence), // converts js value to JSON string
  }).done(function (result) {
    // on success get the return object from server
    console.log(result); // do whatever with it. In this case see it in console
  });
}
