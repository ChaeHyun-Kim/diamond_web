function analysis_start() {
  alert("123");
}
function test() {
  fetch("http://localhost:5000/final", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((res) => res.json())
    .then((json) => {
      alert("123");
    });
}
