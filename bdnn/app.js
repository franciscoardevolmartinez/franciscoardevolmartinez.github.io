const statusDot = document.querySelector("#status-dot");
const statusText = document.querySelector("#status-text");
const sliderList = document.querySelector("#slider-list");
const resetButton = document.querySelector("#reset-button");
const linearFluxToggle = document.querySelector("#linear-flux-toggle");
const plotElement = document.querySelector("#spectrum-plot");

let assets;
let session;
let modelInputName;
let modelOutputName;
let values = [];
let latestRun = 0;

function setStatus(text, kind = "loading") {
  statusText.textContent = text;
  statusDot.classList.toggle("ready", kind === "ready");
  statusDot.classList.toggle("error", kind === "error");
}

function formatValue(value, step) {
  const decimals = step < 0.01 ? 4 : step < 0.1 ? 2 : step < 1 ? 1 : 0;
  return Number(value).toFixed(decimals);
}

function parameterDisplayTransform(parameter) {
  return parameter.displayTransform || parameter.modelTransform || "identity";
}

function displayValue(parameter, modelValue) {
  if (parameterDisplayTransform(parameter) === "pow10") {
    return 10 ** modelValue;
  }
  return modelValue;
}

function displayStep(parameter) {
  if (parameter.name === "Teff") {
    return 1;
  }
  if (parameterDisplayTransform(parameter) === "pow10") {
    return 0.001;
  }
  return parameter.step;
}

function formatParameterValue(parameter, modelValue) {
  const shownValue = displayValue(parameter, modelValue);
  const formatted = formatValue(shownValue, displayStep(parameter));
  return parameter.unit ? `${formatted} ${parameter.unit}` : formatted;
}

function standardizeInputs(modelValues) {
  return modelValues.map((modelValue, index) => {
    return (modelValue - assets.xScaler.mean[index]) / assets.xScaler.scale[index];
  });
}

function inverseStandardizePca(coefficients) {
  return coefficients.map((value, index) => {
    return value * assets.yScaler.scale[index] + assets.yScaler.mean[index];
  });
}

function inversePca(coefficients) {
  const components = assets.pca.components;
  const mean = assets.pca.mean;
  const featureCount = components[0].length;
  const spectrum = new Array(featureCount).fill(0);

  for (let componentIndex = 0; componentIndex < coefficients.length; componentIndex += 1) {
    const coefficient = coefficients[componentIndex];
    const component = components[componentIndex];
    for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
      spectrum[featureIndex] += coefficient * component[featureIndex];
    }
  }

  for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
    spectrum[featureIndex] += mean[featureIndex];
  }

  return spectrum;
}

function reconstructSpectrum(rawOutput) {
  const pcaCoefficients = Array.from(rawOutput);
  const scaledSpectrum = inversePca(pcaCoefficients);
  const logFlux = inverseStandardizePca(scaledSpectrum);

  if (!linearFluxToggle.checked) {
    return logFlux;
  }

  return logFlux.map((value) => 10 ** value);
}

function renderSliders() {
  sliderList.textContent = "";
  values = assets.parameters.map((parameter) => parameter.default);

  assets.parameters.forEach((parameter, index) => {
    const wrapper = document.createElement("label");
    wrapper.className = "slider-control";

    const labelRow = document.createElement("div");
    labelRow.className = "slider-label";

    const name = document.createElement("span");
    name.textContent = parameter.name;

    const value = document.createElement("span");
    value.className = "slider-value";
    value.textContent = formatParameterValue(parameter, parameter.default);

    const input = document.createElement("input");
    input.type = "range";
    input.min = parameter.min;
    input.max = parameter.max;
    input.step = parameter.step;
    input.value = parameter.default;

    const rangeRow = document.createElement("div");
    rangeRow.className = "range-row";
    rangeRow.innerHTML = `<span>${formatParameterValue(parameter, parameter.min)}</span><span>${formatParameterValue(
      parameter,
      parameter.max,
    )}</span>`;

    input.addEventListener("input", () => {
      values[index] = Number(input.value);
      value.textContent = formatParameterValue(parameter, values[index]);
      scheduleInference();
    });

    labelRow.append(name, value);
    wrapper.append(labelRow, input, rangeRow);
    sliderList.append(wrapper);
  });
}

function plotSpectrum(yValues) {
  const xValues = assets.spectrum.x;
  const yLabel = linearFluxToggle.checked ? "Predicted flux (linear)" : assets.spectrum.yLabel;

  Plotly.react(
    plotElement,
    [
      {
        x: xValues,
        y: yValues,
        type: "scatter",
        mode: "lines",
        line: { color: "#0f766e", width: 2 },
        hovertemplate: `${assets.spectrum.xLabel}: %{x}<br>${yLabel}: %{y:.5g}<extra></extra>`,
      },
    ],
    {
      margin: { t: 24, r: 28, b: 58, l: 72 },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      xaxis: {
        title: assets.spectrum.xLabel,
        zeroline: false,
        showgrid: true,
        gridcolor: "#edf0f5",
      },
      yaxis: {
        title: yLabel,
        zeroline: false,
        showgrid: true,
        gridcolor: "#edf0f5",
      },
      showlegend: false,
      autosize: true,
      font: {
        family: "Inter, ui-sans-serif, system-ui, sans-serif",
        color: "#17202a",
      },
    },
    {
      responsive: true,
      displaylogo: false,
    },
  );
}

async function runInference() {
  const runId = ++latestRun;
  const inputValues = Float32Array.from(standardizeInputs(values));
  const inputTensor = new ort.Tensor("float32", inputValues, [1, inputValues.length]);
  const outputs = await session.run({ [modelInputName]: inputTensor });

  if (runId !== latestRun) {
    return;
  }

  const outputTensor = outputs[modelOutputName];
  const spectrum = reconstructSpectrum(outputTensor.data);
  plotSpectrum(spectrum);
  setStatus("Ready", "ready");
}

let inferenceTimer;
function scheduleInference() {
  window.clearTimeout(inferenceTimer);
  inferenceTimer = window.setTimeout(() => {
    runInference().catch((error) => {
      console.error(error);
      setStatus("Inference failed", "error");
    });
  }, 40);
}

function resetValues() {
  document.querySelectorAll(".slider-control input").forEach((input, index) => {
    const parameter = assets.parameters[index];
    input.value = parameter.default;
    values[index] = parameter.default;
    input.closest(".slider-control").querySelector(".slider-value").textContent = formatParameterValue(
      parameter,
      parameter.default,
    );
  });
  scheduleInference();
}

async function init() {
  try {
    setStatus("Loading assets...");
    const assetsResponse = await fetch(`./data/model-assets.json?v=${Date.now()}`);
    assets = await assetsResponse.json();

    renderSliders();
    setStatus("Loading ONNX...");

    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
    session = await ort.InferenceSession.create(`./${assets.modelPath}`, {
      executionProviders: ["wasm"],
    });
    modelInputName = session.inputNames[0];
    modelOutputName = session.outputNames[0];

    resetButton.addEventListener("click", resetValues);
    linearFluxToggle.addEventListener("change", scheduleInference);

    await runInference();
  } catch (error) {
    console.error(error);
    setStatus("Could not load model", "error");
  }
}

init();
