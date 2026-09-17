const results = {

  baseline: {
    name: "FP32 Baseline",
    accuracy: "64.75%",
    size: "42.70 MB",
    latency: "18.28 ms",
    finding:
      "Baseline reference model used for comparison with compression techniques."
  },

  pruning: {
    name: "40% Unstructured Pruning",
    accuracy: "76.56%*",
    size: "42.70 MB",
    latency: "16.68 ms",
    finding:
      "*The corrected pruning run received three fine-tuning epochs after the shortened baseline. The accuracy difference therefore must not be attributed solely to pruning."
  },

  fp16: {
    name: "FP16",
    accuracy: "64.78%",
    size: "21.37 MB",
    latency: "1140.92 ms",
    finding:
      "FP16 approximately halved storage, but inference was much slower on this specific Intel CPU and PyTorch environment."
  },

  dynamic: {
    name: "Dynamic INT8",
    accuracy: "64.79%",
    size: "42.69 MB",
    latency: "12.33 ms",
    finding:
      "Dynamic INT8 preserved accuracy, but only a very small portion of this ResNet-18 is dynamically quantized, so whole-model compression is minimal."
  },

  static: {
    name: "Static INT8",
    accuracy: "33.28%",
    size: "10.78 MB",
    latency: "17.09 ms",
    finding:
      "Static INT8 produced strong storage compression but caused a substantial accuracy drop in the current local experiment."
  }

};


document.querySelectorAll(".tech").forEach(button => {

  button.addEventListener("click", () => {

    document
      .querySelectorAll(".tech")
      .forEach(item => item.classList.remove("active-tech"));

    button.classList.add("active-tech");

    const result = results[button.dataset.tech];

    document.getElementById("techName").textContent = result.name;
    document.getElementById("accuracy").textContent = result.accuracy;
    document.getElementById("size").textContent = result.size;
    document.getElementById("latency").textContent = result.latency;
    document.getElementById("finding").textContent = result.finding;

  });

});
