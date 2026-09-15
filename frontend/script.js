/* =========================================================================
   Ledger — AI House Price Estimator
   Handles: API base detection, nav rail behaviour, model info loading,
   prediction form submission, and result rendering.
   ========================================================================= */

const API_BASE = (() => {
  // When the frontend is served from the same origin as the API (e.g. via
  // FastAPI static files) this resolves automatically. Override by setting
  // window.LEDGER_API_BASE before this script loads if hosting separately.
  if (window.LEDGER_API_BASE) return window.LEDGER_API_BASE;
  return "http://127.0.0.1:8000";
})();

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const numberFormatter = new Intl.NumberFormat("en-US");

// -------------------------------------------------------------------------
// Navigation rail: active link highlighting + mobile toggle
// -------------------------------------------------------------------------
function initNav() {
  const rail = document.getElementById("rail");
  const toggle = document.getElementById("railToggle");
  toggle.addEventListener("click", () => rail.classList.toggle("is-open"));

  const links = document.querySelectorAll(".rail-link");
  const sections = document.querySelectorAll("main .section");

  links.forEach((link) => {
    link.addEventListener("click", () => {
      rail.classList.remove("is-open");
    });
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          links.forEach((l) => l.classList.remove("is-active"));
          const active = document.querySelector(
            `.rail-link[data-section="${entry.target.id}"]`
          );
          if (active) active.classList.add("is-active");
        }
      });
    },
    { rootMargin: "-40% 0px -50% 0px" }
  );
  sections.forEach((s) => observer.observe(s));
}

// -------------------------------------------------------------------------
// API health check
// -------------------------------------------------------------------------
async function checkHealth() {
  const dot = document.getElementById("apiDot");
  const text = document.getElementById("apiStatusText");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("unhealthy");
    dot.classList.add("online");
    text.textContent = "API connected";
  } catch (err) {
    dot.classList.add("offline");
    text.textContent = "API unreachable";
  }
}

// -------------------------------------------------------------------------
// Model info: populates hero stats, metrics panel, feature importance,
// and the neighborhood dropdown.
// -------------------------------------------------------------------------
async function loadModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/model-info`);
    if (!res.ok) throw new Error("Failed to load model info");
    const info = await res.json();

    document.getElementById("statModel").textContent = info.model_name || "—";
    if (info.test_metrics) {
      const { R2, MAE, RMSE } = info.test_metrics;
      document.getElementById("statR2").textContent = R2 !== undefined ? R2.toFixed(3) : "—";
      document.getElementById("statMae").textContent =
        MAE !== undefined ? currencyFormatter.format(MAE) : "—";
      document.getElementById("mMae").textContent =
        MAE !== undefined ? currencyFormatter.format(MAE) : "—";
      document.getElementById("mRmse").textContent =
        RMSE !== undefined ? currencyFormatter.format(RMSE) : "—";
      document.getElementById("mR2").textContent = R2 !== undefined ? R2.toFixed(4) : "—";
    }

    const select = document.getElementById("neighborhoodSelect");
    select.innerHTML = "";
    (info.valid_neighborhoods || []).forEach((n) => {
      const opt = document.createElement("option");
      opt.value = n;
      opt.textContent = n;
      select.appendChild(opt);
    });

    return info;
  } catch (err) {
    console.error(err);
    const errBox = document.getElementById("formError");
    errBox.textContent =
      "Could not load model information from the API. Is the backend running?";
    return null;
  }
}

async function loadFeatureImportance() {
  const container = document.getElementById("importanceList");
  try {
    const res = await fetch(`${API_BASE}/feature-importance`);
    if (!res.ok) throw new Error("no feature importance available yet");
    const rows = await res.json();
    renderImportance(container, rows.slice(0, 8));
  } catch (err) {
    container.innerHTML =
      '<p class="prose small">Feature importance is generated after training — run <code>python -m src.evaluate</code> to populate this section.</p>';
  }
}

function renderImportance(container, rows) {
  if (!rows.length) {
    container.innerHTML = '<p class="prose small">No feature importance data available.</p>';
    return;
  }
  const max = Math.max(...rows.map((r) => r.importance));
  container.innerHTML = rows
    .map((row) => {
      const pct = max > 0 ? (row.importance / max) * 100 : 0;
      const label = humanizeFeatureName(row.feature);
      return `
        <div class="importance-row">
          <span class="importance-name">${label}</span>
          <span class="importance-bar-track"><span class="importance-bar-fill" style="width:${pct.toFixed(1)}%"></span></span>
          <span class="importance-pct">${(row.importance * 100).toFixed(1)}%</span>
        </div>`;
    })
    .join("");
}

function humanizeFeatureName(raw) {
  const map = {
    "Overall Qual": "Overall quality",
    "Garage Cars": "Garage capacity",
    has_fireplace: "Has fireplace",
    total_bathrooms: "Total bathrooms",
    "Gr Liv Area": "Living area",
    "Lot Area": "Lot area",
    property_age: "Property age",
    years_since_renovation: "Years since renovation",
    living_area_per_bedroom: "Living area per bedroom",
    lot_area_per_living_area: "Lot area per living area",
    garage_indicator: "Has a garage",
    has_deck_or_porch: "Has deck / porch",
    total_rooms: "Total rooms",
    floors: "Floors",
    "Overall Cond": "Overall condition",
  };
  if (map[raw]) return map[raw];
  return raw.replace(/^(Neighborhood|Bldg Type|Central Air)_/, "").replace(/_/g, " ");
}

// -------------------------------------------------------------------------
// Prediction form
// -------------------------------------------------------------------------
function buildPayload(form) {
  const data = new FormData(form);
  const get = (name) => data.get(name);

  return {
    bedrooms: parseInt(get("bedrooms"), 10),
    bathrooms: parseFloat(get("bathrooms")),
    living_area: parseFloat(get("living_area")),
    lot_area: parseFloat(get("lot_area")),
    floors: parseFloat(get("floors")),
    garage_cars: parseInt(get("garage_cars"), 10),
    year_built: parseInt(get("year_built"), 10),
    year_remodeled: get("year_remodeled") ? parseInt(get("year_remodeled"), 10) : null,
    neighborhood: get("neighborhood"),
    bldg_type: get("bldg_type"),
    overall_qual: parseInt(get("overall_qual"), 10),
    overall_cond: parseInt(get("overall_cond"), 10),
    total_rooms: parseInt(get("total_rooms"), 10),
    fireplaces: parseInt(get("fireplaces") || "0", 10),
    central_air: form.elements["central_air"].checked ? "Y" : "N",
    has_deck_or_porch: form.elements["has_deck_or_porch"].checked,
  };
}

function validatePayload(payload) {
  if (payload.year_remodeled && payload.year_remodeled < payload.year_built) {
    return "Year remodeled cannot be earlier than year built.";
  }
  if (!payload.neighborhood) {
    return "Please select a neighborhood.";
  }
  return null;
}

async function submitPrediction(event) {
  event.preventDefault();
  const form = event.target;
  const errorBox = document.getElementById("formError");
  const submitBtn = document.getElementById("submitBtn");
  errorBox.textContent = "";

  const payload = buildPayload(form);
  const clientError = validatePayload(payload);
  if (clientError) {
    errorBox.textContent = clientError;
    return;
  }

  const resultCard = document.getElementById("resultCard");
  const loading = document.getElementById("resultLoading");
  const body = document.getElementById("resultBody");
  resultCard.hidden = false;
  loading.hidden = false;
  body.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting…";

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const json = await res.json();

    if (!res.ok) {
      const detail = Array.isArray(json.detail)
        ? json.detail.map((d) => d.msg).join("; ")
        : json.detail || "The prediction could not be completed.";
      throw new Error(detail);
    }

    renderResult(json, payload);
  } catch (err) {
    loading.hidden = true;
    resultCard.hidden = true;
    errorBox.textContent = err.message || "Something went wrong reaching the API.";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict house price";
  }
}

function renderResult(result, payload) {
  document.getElementById("resultLoading").hidden = true;
  const body = document.getElementById("resultBody");
  body.hidden = false;

  document.getElementById("resultPrice").textContent = currencyFormatter.format(
    result.predicted_price
  );

  if (result.estimated_range) {
    document.getElementById("resultRange").textContent =
      `Likely between ${currencyFormatter.format(result.estimated_range.low)} and ` +
      `${currencyFormatter.format(result.estimated_range.high)} (${result.estimated_range.basis})`;
  } else {
    document.getElementById("resultRange").textContent = "";
  }

  document.getElementById("resultModel").textContent = result.model || "—";
  document.getElementById("resultLiving").textContent =
    `${numberFormatter.format(payload.living_area)} sq ft`;
  document.getElementById("resultNeighborhood").textContent = payload.neighborhood;
  document.getElementById("resultQuality").textContent = `${payload.overall_qual} / 10`;
  document.getElementById("resultDisclaimer").textContent =
    result.disclaimer ||
    "This is an estimate, not a guaranteed market price or a professional appraisal.";
}

// -------------------------------------------------------------------------
// Init
// -------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initNav();
  checkHealth();
  loadModelInfo();
  loadFeatureImportance();
  document.getElementById("predictForm").addEventListener("submit", submitPrediction);
});
