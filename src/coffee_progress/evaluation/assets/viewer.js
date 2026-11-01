"use strict";
const report = JSON.parse(document.getElementById("reportData").textContent);
const byId = id => document.getElementById(id);
const video = byId("video"), slider = byId("timeline"), chart = byId("chart");
const cameraSelect = byId("camera");
const allVideos = [video, byId("videoRight"), byId("videoWrist")];
let activeVideos = [video];
let episode, currentFrame = 0, chartScale, playbackRequest;
const geometry = {left:62, right:875, top:16, bottom:200};
document.title = `${report.name} | Progress evaluation`;
byId("runName").textContent = report.name;

function cameraName(key) {
  return key.replace(/^observation\.images\./, "").replace(/^robot0_/, "").replaceAll("_", " ");
}
function stem() { return `episode_${String(episode.episode_id).padStart(6, "0")}`; }
function xPosition(index) {
  return geometry.left + index / Math.max(1, episode.progress.length - 1) * (geometry.right - geometry.left);
}
function yPosition(value) {
  return geometry.bottom - (value - chartScale.low) / (chartScale.high - chartScale.low) * (geometry.bottom - geometry.top);
}
function drawChart() {
  const values = episode.progress;
  const low = Math.min(0, ...values), high = Math.max(1, ...values);
  chartScale = {low:low - (high-low)*.05, high:high + (high-low)*.05};
  let svg = "";
  for (let i=0; i<=4; i++) {
    const value = low + (high-low)*i/4, y = yPosition(value);
    svg += `<line x1="${geometry.left}" x2="${geometry.right}" y1="${y}" y2="${y}" stroke="#e5ecef"/>`;
    svg += `<text x="50" y="${y+4}" text-anchor="end" fill="#627685" font-size="11">${value.toFixed(2)}</text>`;
  }
  (episode.boundaries || []).forEach((frame,index) => {
    const x = xPosition(frame);
    svg += `<line x1="${x}" x2="${x}" y1="${geometry.top}" y2="${geometry.bottom}" stroke="#8799a5" stroke-dasharray="3 5"/>`;
    svg += `<text x="${x}" y="11" text-anchor="middle" fill="#627685" font-size="10">B${index+1}</text>`;
  });
  for (let i=0; i<=4; i++) {
    const frame = Math.round((values.length-1)*i/4);
    svg += `<text x="${xPosition(frame)}" y="222" text-anchor="middle" fill="#627685" font-size="11">${frame}</text>`;
  }
  svg += '<text x="468" y="244" text-anchor="middle" fill="#627685" font-size="11">Timestep</text>';
  const points = values.map((value,index) => `${xPosition(index)},${yPosition(value)}`).join(" ");
  svg += `<polyline points="${points}" fill="none" stroke="#147d92" stroke-width="2.5" stroke-linejoin="round"/>`;
  svg += `<line id="cursor" y1="${geometry.top}" y2="${geometry.bottom}" stroke="#d78925" stroke-width="1.5" stroke-dasharray="4 4"/>`;
  svg += '<circle id="dot" r="5" fill="#d78925" stroke="white" stroke-width="2"/>';
  chart.innerHTML = svg;
}
function updateReadout(index) {
  currentFrame = Math.max(0, Math.min(episode.progress.length-1, Math.round(index)));
  const value = episode.progress[currentFrame];
  slider.value = currentFrame;
  byId("score").textContent = value.toFixed(3);
  byId("scoreFill").style.width = `${Math.max(0,Math.min(1,value))*100}%`;
  byId("frameReadout").textContent = `${currentFrame} / ${episode.progress.length-1}`;
  byId("timeReadout").textContent = `${episode.timestamps[currentFrame].toFixed(2)} s`;
  byId("sliderReadout").textContent = `${currentFrame} / ${episode.progress.length-1}`;
  byId("cursor").setAttribute("x1", xPosition(currentFrame));
  byId("cursor").setAttribute("x2", xPosition(currentFrame));
  byId("dot").setAttribute("cx", xPosition(currentFrame));
  byId("dot").setAttribute("cy", yPosition(value));
  chart.setAttribute("aria-label", `Timestep ${currentFrame}, predicted progress ${value.toFixed(3)}`);
}
function seekFrame(index) {
  video.pause();
  updateReadout(index);
  activeVideos.forEach(view => {
    if (view.readyState >= 1) view.currentTime = episode.timestamps[currentFrame] + 0.00001;
  });
}
function loadCamera() {
  allVideos.forEach(view => view.pause());
  byId("videoError").hidden = true;
  const views = [cameraSelect.value, ...Object.keys(episode.videos).filter(c => c !== cameraSelect.value)];
  const labels = ["mainCameraLabel", "rightCameraLabel", "wristCameraLabel"];
  activeVideos = allVideos.slice(0, views.length);
  byId("cameraSide").hidden = views.length < 2;
  byId("wristPanel").hidden = views.length < 3;
  byId("cameraGrid").classList.toggle("single-camera", views.length === 1);
  activeVideos.forEach((view,index) => {
    byId(labels[index]).textContent = cameraName(views[index]);
    view.src = episode.videos[views[index]];
    view.load();
  });
}
function selectEpisode(index) {
  video.pause();
  episode = report.episodes[index];
  currentFrame = 0;
  [...byId("episodes").children].forEach((button,i) => {
    button.classList.toggle("active", i===index);
    button.setAttribute("aria-pressed", i===index ? "true" : "false");
  });
  byId("episodeTitle").textContent = `Episode ${episode.episode_id}`;
  byId("frameCount").textContent = episode.progress.length.toLocaleString();
  byId("csvLink").href = `${stem()}.csv`;
  byId("plotLink").href = `${stem()}.png`;
  slider.max = episode.progress.length-1;
  const previousCamera = cameraSelect.value;
  cameraSelect.replaceChildren(...Object.keys(episode.videos).map(key => {
    const option = document.createElement("option");
    option.value = key; option.textContent = cameraName(key); return option;
  }));
  if (episode.videos[previousCamera]) cameraSelect.value = previousCamera;
  else cameraSelect.value = Object.keys(episode.videos).find(c => c.endsWith("agentview_left")) || cameraSelect.value;
  drawChart(); updateReadout(0); loadCamera();
}
function animatePlayback() {
  if (video.paused || video.ended) return;
  updateReadout(Math.floor(video.currentTime * episode.fps + 0.0001));
  activeVideos.slice(1).forEach(view => {
    if (view.readyState >= 2 && !view.seeking && Math.abs(view.currentTime-video.currentTime) > .075)
      view.currentTime = video.currentTime;
  });
  playbackRequest = requestAnimationFrame(animatePlayback);
}
async function togglePlayback() {
  if (!video.paused) { video.pause(); return; }
  if (currentFrame >= episode.progress.length-1) seekFrame(0);
  try { await Promise.all(activeVideos.map(view => view.play())); } catch (error) {
    // Seeking, pausing, or changing episodes can interrupt an outstanding play request.
    if (error.name === "AbortError") return;
    showVideoError(error.name === "NotAllowedError"
      ? "Playback was blocked. Open this report in Chrome or Edge and press Play again."
      : "Playback could not start. Open the report with coffee-view in Chrome or Edge.");
  }
}
function showVideoError(message) {
  byId("videoError").textContent = message;
  byId("videoError").hidden = false;
}
report.episodes.forEach((item,index) => {
  const button = document.createElement("button");
  const title = document.createElement("strong"), detail = document.createElement("span");
  title.textContent = `Episode ${item.episode_id}`;
  detail.textContent = `${item.progress.length} frames · ${item.timestamps.at(-1).toFixed(1)} s`;
  button.append(title,detail); button.addEventListener("click", () => selectEpisode(index));
  byId("episodes").append(button);
});
slider.addEventListener("input", () => seekFrame(Number(slider.value)));
byId("previous").addEventListener("click", () => seekFrame(currentFrame-1));
byId("next").addEventListener("click", () => seekFrame(currentFrame+1));
byId("play").addEventListener("click", togglePlayback);
cameraSelect.addEventListener("change", loadCamera);
allVideos.forEach(view => {
  view.addEventListener("loadedmetadata", () => {
    view.currentTime = episode.timestamps[currentFrame] + 0.00001;
  });
  view.addEventListener("error", () => {
    const code = view.error?.code;
    showVideoError(code === 3 || code === 4
      ? "This browser could not decode the MP4. Run coffee-view and open its URL in Chrome or Edge."
      : "Video file could not be read. Run coffee-view with the complete report folder, including videos.");
  });
  view.addEventListener("canplay", () => {
    if (!activeVideos.some(v => v.error)) byId("videoError").hidden = true;
  });
});
video.addEventListener("play", () => {
  byId("play").textContent = "Ⅱ Pause"; byId("play").setAttribute("aria-label", "Pause video");
  cancelAnimationFrame(playbackRequest); animatePlayback();
});
video.addEventListener("pause", () => {
  activeVideos.slice(1).forEach(view => view.pause());
  cancelAnimationFrame(playbackRequest);
  byId("play").textContent = "▶ Play"; byId("play").setAttribute("aria-label", "Play video");
});
video.addEventListener("ended", () => updateReadout(episode.progress.length-1));
chart.addEventListener("pointermove", event => {
  if (!video.paused) return;
  const point = new DOMPoint(event.clientX,event.clientY).matrixTransform(chart.getScreenCTM().inverse());
  seekFrame((point.x-geometry.left)/(geometry.right-geometry.left)*(episode.progress.length-1));
});
chart.addEventListener("click", event => {
  const point = new DOMPoint(event.clientX,event.clientY).matrixTransform(chart.getScreenCTM().inverse());
  seekFrame((point.x-geometry.left)/(geometry.right-geometry.left)*(episode.progress.length-1));
});
document.addEventListener("keydown", event => {
  if (["INPUT","SELECT"].includes(event.target.tagName)) return;
  if (event.key === "ArrowLeft") { event.preventDefault(); seekFrame(currentFrame-1); }
  if (event.key === "ArrowRight") { event.preventDefault(); seekFrame(currentFrame+1); }
  if (event.code === "Space") { event.preventDefault(); togglePlayback(); }
});
selectEpisode(0);
