function analysis_start() {


  fetch('http://127.0.0.1:5000/final', {
    method: "GET",
    headers: {

        "Content-Type": "application/json",
    },
  })

    .then((res) => res.json())
    .then((res) =>
        console.log('Success:',res)


    );
}
function test() {
  fetch("http://localhost:5000/final", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((res) => response.json())
    .then((res) => {
        console.log(response)
    });
}
