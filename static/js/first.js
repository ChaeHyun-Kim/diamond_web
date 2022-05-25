function open_page() {
  console.log("넘어가냐?");
  location.href = "/result_render";
}
function img_loading() {
  let img_box = document.getElementById("img_pos");
  let img_load = document.createElement("img");
  img_load.setAttribute("src", "../static/img/loading.PNG");
  img_load.setAttribute("class", "image_full");
  img_box.appendChild(img_load);
}

function scrollDisable() {
  $("html, body").addClass("hidden");
}
function scrollAble() {
  $("html, body").removeClass("hidden");
}

function btn_text_Disabled() {
  const target_btn = document.getElementById("start_btn");
  const target_text = document.getElementById("input_text");
  target_btn.disabled = true;
  target_text.disabled = true;
}

//분석
function predict_please() {
  console.log("문장 분석 시작");
  scrollDisable();
  btn_text_Disabled();
  var textarea = document.getElementById("input_text");
  valu = textarea.value;
  if (valu.replace(" ", "").length < 10) {
    alert("광고 내용이 짧아 판별할 수 없습니다. 세 문장 이상 입력해주세요.");
    return;
  } else {
    var sentence = textarea.value;
    img_loading();

    $.ajax({
      url: "/final",
      type: "POST",
      contentType: "application/json",
      dataType: "json",
      data: JSON.stringify({
        ad: sentence,
      }),
      success: function (data) {
        console.log("성공");
        console.log("분석실행 결과", data);
        localStorage.setItem("predict_result", JSON.stringify(data));
        console.log(
          "test:",
          JSON.parse(localStorage.getItem("predict_result"))
        );
        open_page();
      },
      error: function (request, status, error) {
        console.log("실패");
        location.href = "/restart";
        // 실패 시 처리
      },
      complete: function (data) {
        console.log("실패지만 완료");
        //  실패했어도 완료가 되었을 때 처리
      },
    });
  }
}
