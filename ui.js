const EMOTIONS = ["happy", "sad"]
let happyFaceCount = 0;
let sadFaceCount = 0;
let mouseDown = false;

const controllerDataset = new ControllerDataset(2);

const happyButton = document.getElementById('happy');
const sadButton = document.getElementById('sad');
const trainButton = document.getElementById("train");
const predictButton = document.getElementById("predict");
const emotionText = document.getElementById("emotion");