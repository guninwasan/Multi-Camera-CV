document.addEventListener("DOMContentLoaded", function () {
  const startStopButton = document.getElementById("start-stop-button");
  const resetButton = document.getElementById("reset-button");
  const teamForm = document.getElementById("team-form");
  const elapsedTimeElement = document.getElementById("elapsed-time");
  const humanInterventionsElement = document.getElementById(
    "human-interventions"
  );
  let timerRunning = false;

  startStopButton.addEventListener("click", function () {
    fetch("/start_stop_timer", { method: "POST" })
      .then((response) => response.json())
      .then(() => {
        timerRunning = !timerRunning;
        startStopButton.textContent = timerRunning
          ? "Stop Timer"
          : "Start Timer";
        startStopButton.classList.toggle("btn-success", !timerRunning);
        startStopButton.classList.toggle("btn-danger", timerRunning);
      });
  });

  resetButton.addEventListener("click", function () {
    fetch("/reset", { method: "POST" })
      .then((response) => response.json())
      .then(() => {
        elapsedTimeElement.textContent = "0.00 s";
        humanInterventionsElement.textContent = "0";
        timerRunning = false;
        startStopButton.textContent = "Start Timer";
        startStopButton.classList.add("btn-success");
        startStopButton.classList.remove("btn-danger");
      });
  });

  teamForm.addEventListener("submit", function (event) {
    event.preventDefault();
    const formData = new FormData(teamForm);
    fetch("/update_team", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then(() => {
        alert("Team name/number updated successfully!");
      });
  });

  function updateScorecard() {
    fetch("/scorecard")
      .then((response) => response.json())
      .then((data) => {
        elapsedTimeElement.textContent = data.elapsed_time;
        humanInterventionsElement.textContent = data.human_interventions;
      });
  }

  setInterval(updateScorecard, 1000);
});

function toggleFullscreen() {
  const elem = document.getElementById("video-feed");
  if (!document.fullscreenElement) {
    if (elem.requestFullscreen) {
      elem.requestFullscreen();
    } else if (elem.mozRequestFullScreen) {
      elem.mozRequestFullScreen();
    } else if (elem.webkitRequestFullscreen) {
      elem.webkitRequestFullscreen();
    } else if (elem.msRequestFullscreen) {
      elem.msRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.mozCancelFullScreen) {
      document.mozCancelFullScreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    } else if (document.msExitFullscreen) {
      document.msExitFullscreen();
    }
  }
}
