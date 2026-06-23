// Global Application State
let metricsData = null;
let currentDisease = 'heart_disease';
let activeCharts = {};

// UI Config definitions for input fields (Ranges, Labels, steps, defaults)
const featureSchema = {
    heart_disease: {
        age: { label: "Age", type: "range", min: 20, max: 100, step: 1, default: 54, suffix: " years" },
        sex: { label: "Sex", type: "select", options: [[1, "Male"], [0, "Female"]], default: 1 },
        cp: { label: "Chest Pain Type", type: "select", options: [[1, "Typical Angina"], [2, "Atypical Angina"], [3, "Non-Anginal Pain"], [4, "Asymptomatic"]], default: 4 },
        trestbps: { label: "Resting Blood Pressure", type: "range", min: 80, max: 200, step: 1, default: 130, suffix: " mmHg" },
        chol: { label: "Serum Cholesterol", type: "range", min: 100, max: 600, step: 1, default: 240, suffix: " mg/dL" },
        fbs: { label: "Fasting Blood Sugar > 120 mg/dL", type: "select", options: [[0, "No / False"], [1, "Yes / True"]], default: 0 },
        restecg: { label: "Resting ECG Results", type: "select", options: [[0, "Normal"], [1, "ST-T Wave Abnormality"], [2, "Left Ventricular Hypertrophy"]], default: 0 },
        thalach: { label: "Max Heart Rate Achieved", type: "range", min: 60, max: 220, step: 1, default: 150, suffix: " bpm" },
        exang: { label: "Exercise Induced Angina", type: "select", options: [[0, "No / False"], [1, "Yes / True"]], default: 0 },
        oldpeak: { label: "ST Depression (oldpeak)", type: "range", min: 0.0, max: 6.2, step: 0.1, default: 1.0, suffix: " mm" },
        slope: { label: "Slope of peak exercise ST", type: "select", options: [[1, "Upsloping"], [2, "Flat"], [3, "Downsloping"]], default: 1 },
        ca: { label: "Major Vessels Colored (0-3)", type: "range", min: 0, max: 3, step: 1, default: 0, suffix: "" },
        thal: { label: "Thalassemia State", type: "select", options: [[3, "Normal"], [6, "Fixed Defect"], [7, "Reversible Defect"]], default: 3 }
    },
    diabetes: {
        Pregnancies: { label: "Pregnancies", type: "range", min: 0, max: 20, step: 1, default: 3, suffix: "" },
        Glucose: { label: "Glucose level", type: "range", min: 40, max: 200, step: 1, default: 115, suffix: " mg/dL" },
        BloodPressure: { label: "Diastolic Blood Pressure", type: "range", min: 30, max: 130, step: 1, default: 72, suffix: " mmHg" },
        SkinThickness: { label: "Triceps Skinfold Thickness", type: "range", min: 5, max: 100, step: 1, default: 20, suffix: " mm" },
        Insulin: { label: "Serum Insulin (2-hour)", type: "range", min: 10, max: 900, step: 1, default: 80, suffix: " μU/mL" },
        BMI: { label: "Body Mass Index", type: "range", min: 15.0, max: 70.0, step: 0.1, default: 32.0, suffix: " kg/m²" },
        DiabetesPedigreeFunction: { label: "Diabetes Pedigree Score", type: "range", min: 0.05, max: 2.50, step: 0.01, default: 0.47, suffix: "" },
        Age: { label: "Age", type: "range", min: 21, max: 90, step: 1, default: 33, suffix: " years" }
    },
    breast_cancer: {
        "worst area": { label: "Worst Area", type: "range", min: 150.0, max: 4250.0, step: 10.0, default: 880.0, suffix: "" },
        "worst concave points": { label: "Worst Concave Points", type: "range", min: 0.0, max: 0.3, step: 0.01, default: 0.12, suffix: "" },
        "mean concave points": { label: "Mean Concave Points", type: "range", min: 0.0, max: 0.2, step: 0.01, default: 0.05, suffix: "" },
        "worst radius": { label: "Worst Radius", type: "range", min: 7.0, max: 36.0, step: 0.1, default: 16.3, suffix: " mm" },
        "worst perimeter": { label: "Worst Perimeter", type: "range", min: 50.0, max: 250.0, step: 1.0, default: 107.0, suffix: " mm" },
        "mean perimeter": { label: "Mean Perimeter", type: "range", min: 40.0, max: 190.0, step: 1.0, default: 92.0, suffix: " mm" },
        "mean concavity": { label: "Mean Concavity", type: "range", min: 0.0, max: 0.5, step: 0.01, default: 0.09, suffix: "" },
        "mean area": { label: "Mean Area", type: "range", min: 140.0, max: 2500.0, step: 10.0, default: 650.0, suffix: "" },
        "worst concavity": { label: "Worst Concavity", type: "range", min: 0.0, max: 1.3, step: 0.01, default: 0.27, suffix: "" },
        "mean radius": { label: "Mean Radius", type: "range", min: 6.0, max: 28.0, step: 0.1, default: 14.1, suffix: " mm" }
    }
};

// Patient Presets for Quick Testing
const patientPresets = {
    heart_disease: {
        healthy: { age: 45, sex: 0, cp: 1, trestbps: 118, chol: 195, fbs: 0, restecg: 0, thalach: 172, exang: 0, oldpeak: 0.0, slope: 1, ca: 0, thal: 3 },
        sick: { age: 64, sex: 1, cp: 4, trestbps: 152, chol: 294, fbs: 1, restecg: 2, thalach: 108, exang: 1, oldpeak: 2.8, slope: 2, ca: 3, thal: 7 }
    },
    diabetes: {
        healthy: { Pregnancies: 1, Glucose: 92, BloodPressure: 64, SkinThickness: 18, Insulin: 72, BMI: 21.8, DiabetesPedigreeFunction: 0.22, Age: 24 },
        sick: { Pregnancies: 7, Glucose: 174, BloodPressure: 88, SkinThickness: 39, Insulin: 240, BMI: 37.6, DiabetesPedigreeFunction: 0.94, Age: 52 }
    },
    breast_cancer: {
        healthy: { "worst area": 450.0, "worst concave points": 0.03, "mean concave points": 0.02, "worst radius": 10.8, "worst perimeter": 70.2, "mean perimeter": 62.1, "mean concavity": 0.03, "mean area": 320.0, "worst concavity": 0.07, "mean radius": 9.5 },
        sick: { "worst area": 2050.0, "worst concave points": 0.24, "mean concave points": 0.14, "worst radius": 25.2, "worst perimeter": 172.0, "mean perimeter": 142.0, "mean concavity": 0.26, "mean area": 1490.0, "worst concavity": 0.58, "mean radius": 21.5 }
    }
};

// Document Loaded Init
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

function initApp() {
    // Tab switching
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const tabId = item.getAttribute("data-tab");
            switchTab(tabId);
        });
    });

    // Theme toggle
    const themeBtn = document.getElementById("theme-toggle");
    themeBtn.addEventListener("click", () => {
        document.body.classList.toggle("light-theme");
    });

    // Disease Selector
    const diseaseCards = document.querySelectorAll(".disease-card");
    diseaseCards.forEach(card => {
        card.addEventListener("click", () => {
            diseaseCards.forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            currentDisease = card.getAttribute("data-disease");
            generateFormFields(currentDisease);
            resetResultPanel();
        });
    });

    // Preset Buttons
    document.getElementById("btn-fill-healthy").addEventListener("click", (e) => {
        e.preventDefault();
        autofillPreset('healthy');
    });
    document.getElementById("btn-fill-sick").addEventListener("click", (e) => {
        e.preventDefault();
        autofillPreset('sick');
    });

    // Predict Form Submit
    const form = document.getElementById("prediction-form");
    form.addEventListener("submit", handlePredictionSubmit);

    // Accordion handler
    const accordions = document.querySelectorAll(".accordion-header");
    accordions.forEach(header => {
        header.addEventListener("click", () => {
            const item = header.parentElement;
            const isOpen = item.classList.contains("open");
            document.querySelectorAll(".accordion-item").forEach(acc => acc.classList.remove("open"));
            if (!isOpen) {
                item.classList.add("open");
            }
        });
    });
    // Open first accordion by default
    if (accordions.length > 0) {
        accordions[0].parentElement.classList.add("open");
    }

    // Comparison select change
    document.getElementById("comparison-disease-select").addEventListener("change", (e) => {
        updateComparisonTab(e.target.value);
    });

    // Confusion Matrix algorithm change
    document.getElementById("cm-algorithm-select").addEventListener("change", () => {
        updateConfusionMatrix();
    });

    // Fetch initial model metrics
    fetchMetrics();
    
    // Generate initial form fields
    generateFormFields(currentDisease);
}

// Switching Tab Logic
function switchTab(tabId) {
    // Update nav items
    document.querySelectorAll(".nav-item").forEach(item => {
        item.classList.remove("active");
        if (item.getAttribute("data-tab") === tabId) {
            item.classList.add("active");
        }
    });

    // Update Panes
    document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
    });
    const targetPane = document.getElementById(`tab-${tabId}`);
    if (targetPane) targetPane.classList.add("active");

    // Update Titles
    const title = document.getElementById("page-title");
    const subtitle = document.getElementById("page-subtitle");
    
    if (tabId === 'dashboard') {
        title.innerText = "Analytics Dashboard";
        subtitle.innerText = "Overview of disease predictive capabilities and active learning algorithms.";
    } else if (tabId === 'predict') {
        title.innerText = "Diagnostic Simulator";
        subtitle.innerText = "Input patient clinical indicators to simulate clinical diagnostic risk.";
    } else if (tabId === 'comparison') {
        title.innerText = "Model & Algorithm Analysis";
        subtitle.innerText = "Explore comparative classification metrics, ROC curves, and feature importance mappings.";
        // Trigger resize on charts to fix render sizes when switching tabs
        setTimeout(() => {
            Object.values(activeCharts).forEach(chart => chart.resize());
        }, 100);
    } else if (tabId === 'datasets') {
        title.innerText = "Dataset Insights";
        subtitle.innerText = "Deep dive into features, metrics, and parameters of the underlying medical training datasets.";
    }
}

// Fetch Metrics from Server
function fetchMetrics() {
    fetch('/api/metrics')
        .then(response => response.json())
        .then(res => {
            if (res.success) {
                metricsData = res.data;
                // Init comparison views
                updateComparisonTab('heart_disease');
            } else {
                console.error("Failed to fetch metrics:", res.error);
            }
        })
        .catch(err => {
            console.error("Error connecting to server api:", err);
        });
}

// Generate Input Form fields dynamically based on schema
function generateFormFields(disease) {
    const container = document.getElementById("form-fields-container");
    container.innerHTML = "";
    const schema = featureSchema[disease];
    
    for (const [key, field] of Object.entries(schema)) {
        const formGroup = document.createElement("div");
        formGroup.className = "form-group";
        
        if (field.type === "range") {
            formGroup.innerHTML = `
                <label for="f-${key}">
                    <span>${field.label}</span>
                    <span class="val-bubble" id="bubble-${key}">${field.default}${field.suffix}</span>
                </label>
                <input type="range" 
                       id="f-${key}" 
                       name="${key}" 
                       class="form-control form-range" 
                       min="${field.min}" 
                       max="${field.max}" 
                       step="${field.step}" 
                       value="${field.default}">
            `;
            
            container.appendChild(formGroup);
            
            // Slider bubble updates
            const slider = formGroup.querySelector("input[type='range']");
            slider.addEventListener("input", (e) => {
                document.getElementById(`bubble-${key}`).innerText = `${e.target.value}${field.suffix}`;
            });
            
        } else if (field.type === "select") {
            let optionsHtml = "";
            field.options.forEach(opt => {
                optionsHtml += `<option value="${opt[0]}">${opt[1]}</option>`;
            });
            
            formGroup.innerHTML = `
                <label for="f-${key}">${field.label}</label>
                <select id="f-${key}" name="${key}" class="form-control">
                    ${optionsHtml}
                </select>
            `;
            container.appendChild(formGroup);
            // set default
            formGroup.querySelector("select").value = field.default;
        }
    }
}

// Autofill with predefined presets
function autofillPreset(type) {
    const presets = patientPresets[currentDisease][type];
    if (!presets) return;
    
    for (const [key, val] of Object.entries(presets)) {
        const input = document.getElementById(`f-${key}`);
        if (input) {
            input.value = val;
            
            // Update bubble text if range slider
            const bubble = document.getElementById(`bubble-${key}`);
            if (bubble) {
                const schema = featureSchema[currentDisease][key];
                bubble.innerText = `${val}${schema.suffix || ''}`;
            }
        }
    }
}

// Reset results pane to empty
function resetResultPanel() {
    document.getElementById("result-placeholder").classList.remove("hidden");
    document.getElementById("result-active").classList.add("hidden");
}

// Handle submit predict
function handlePredictionSubmit(e) {
    e.preventDefault();
    
    const form = document.getElementById("prediction-form");
    const formData = new FormData(form);
    
    const selectedAlgo = formData.get("algorithm");
    const features = {};
    
    // Build feature dictionary
    const schemaKeys = Object.keys(featureSchema[currentDisease]);
    schemaKeys.forEach(key => {
        features[key] = parseFloat(formData.get(key));
    });
    
    // Update button text to loading state
    const btn = document.getElementById("btn-predict");
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner">Computing Analysis...</span>`;

    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            disease: currentDisease,
            algorithm: selectedAlgo,
            features: features
        })
    })
    .then(response => response.json())
    .then(res => {
        btn.disabled = false;
        btn.innerHTML = originalText;
        
        if (res.success) {
            displayPredictionResult(res);
        } else {
            alert("Error computing prediction: " + res.error);
        }
    })
    .catch(err => {
        btn.disabled = false;
        btn.innerHTML = originalText;
        console.error("Predict submit error:", err);
        alert("Server connection failed. Make sure Flask app.py is running.");
    });
}

// Display prediction results
function displayPredictionResult(res) {
    // Hide placeholder, show active result
    document.getElementById("result-placeholder").classList.add("hidden");
    document.getElementById("result-active").classList.remove("hidden");
    
    const pct = Math.round(res.probability * 100);
    document.getElementById("result-pct").innerText = `${pct}%`;
    document.getElementById("result-level").innerText = `${res.risk_level} Risk`;
    
    // Update gauge progress stroke
    const gaugeFill = document.querySelector(".gauge-fill");
    // Dasharray is 188.5, compute offset: 188.5 * (1 - fraction)
    const strokeOffset = 188.5 * (1 - res.probability);
    gaugeFill.style.strokeDashoffset = strokeOffset;
    
    // Update gauge color based on risk level
    if (res.risk_level === 'Low') {
        gaugeFill.style.stroke = "var(--success)";
        document.getElementById("result-level").style.color = "var(--success)";
        document.getElementById("result-title").innerText = "Normal Vitals / Healthy Findings";
        document.getElementById("result-badge").className = "badge badge-success";
        document.getElementById("result-badge").innerText = "Low Risk";
        document.getElementById("result-text").innerText = `The model assesses a low likelihood (${pct}%) of disease based on patient vitals. The patient appears to show normal clinical benchmarks.`;
        populateRecommendations(['Continue regular exercise and healthy nutrition.', 'Schedule routine checkups annually.', 'Monitor vitals (blood pressure, blood glucose) occasionally.']);
    } else if (res.risk_level === 'Moderate') {
        gaugeFill.style.stroke = "var(--warning)";
        document.getElementById("result-level").style.color = "var(--warning)";
        document.getElementById("result-title").innerText = "Borderline Risk Indicators";
        document.getElementById("result-badge").className = "badge badge-warning";
        document.getElementById("result-badge").innerText = "Moderate Risk";
        document.getElementById("result-text").innerText = `The model identifies a moderate risk of ${pct}% with borderline metrics. Clinical parameters fall slightly outside optimal ranges.`;
        populateRecommendations(['Consult a physician to discuss these findings.', 'Adhere to lifestyle adjustments, including reduction in sodium/sugars.', 'Re-evaluate blood work and clinical indicators in 3-6 months.']);
    } else {
        gaugeFill.style.stroke = "var(--danger)";
        document.getElementById("result-level").style.color = "var(--danger)";
        document.getElementById("result-title").innerText = "High-Risk Diagnostic Warning";
        document.getElementById("result-badge").className = "badge badge-danger";
        document.getElementById("result-badge").innerText = "High Risk";
        document.getElementById("result-text").innerText = `Critical parameters detected. The classifier assigns a high probability of ${pct}% for disease presence. Prompt evaluation is advised.`;
        populateRecommendations(['Schedule an urgent diagnostic appointment with a specialist.', 'Avoid strenuous physical activity until cleared by a physician.', 'Initiate immediate medical evaluation and diagnostic screening.']);
    }
}

function populateRecommendations(recs) {
    const list = document.getElementById("recommendations-list");
    list.innerHTML = "";
    recs.forEach(rec => {
        const li = document.createElement("li");
        li.innerText = rec;
        list.appendChild(li);
    });
}

// Update Comparison Tab layouts
function updateComparisonTab(diseaseId) {
    if (!metricsData || !metricsData[diseaseId]) return;
    
    const data = metricsData[diseaseId];
    
    // 1. Build metrics table
    const tbody = document.querySelector("#metrics-compare-table tbody");
    tbody.innerHTML = "";
    
    const algos = ['random_forest', 'xgboost', 'svm', 'logistic_regression'];
    const algoDisplayNames = {
        random_forest: "Random Forest Classifier",
        xgboost: "XGBoost Classifier",
        svm: "Support Vector Machine (SVM)",
        logistic_regression: "Regularized Logistic Regression"
    };
    
    algos.forEach(algo => {
        const m = data.models[algo];
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${algoDisplayNames[algo]}</strong></td>
            <td>${(m.accuracy * 100).toFixed(1)}%</td>
            <td>${(m.precision * 100).toFixed(1)}%</td>
            <td>${(m.recall * 100).toFixed(1)}%</td>
            <td>${(m.f1_score * 100).toFixed(1)}%</td>
            <td>${m.roc_auc.toFixed(3)}</td>
        `;
        tbody.appendChild(tr);
    });
    
    // 2. Render Charts
    renderComparisonCharts(diseaseId);
}

// Render Chart.js comparison charts
function renderComparisonCharts(diseaseId) {
    const data = metricsData[diseaseId];
    const algos = ['random_forest', 'xgboost', 'svm', 'logistic_regression'];
    const algoColors = {
        random_forest: '#6366f1',       // Indigo
        xgboost: '#06b6d4',             // Cyan
        svm: '#ec4899',                 // Pink
        logistic_regression: '#10b981'  // Emerald
    };
    
    // --- CHART 1: COMPARATIVE BAR CHART ---
    if (activeCharts.metrics) activeCharts.metrics.destroy();
    
    const ctxMetrics = document.getElementById('metricsChart').getContext('2d');
    const metricLabels = ['Accuracy', 'Precision', 'Recall', 'F1-Score'];
    
    const barDatasets = algos.map(algo => {
        const m = data.models[algo];
        return {
            label: algo.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
            data: [m.accuracy, m.precision, m.recall, m.f1_score],
            backgroundColor: algoColors[algo],
            borderRadius: 6
        };
    });
    
    activeCharts.metrics = new Chart(ctxMetrics, {
        type: 'bar',
        data: {
            labels: metricLabels,
            datasets: barDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0.5,
                    max: 1.0,
                    ticks: { color: getComputedStyle(document.body).getPropertyValue('--text-muted') },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    ticks: { color: getComputedStyle(document.body).getPropertyValue('--text-muted') },
                    grid: { display: false }
                }
            },
            plugins: {
                legend: {
                    labels: { color: getComputedStyle(document.body).getPropertyValue('--text-main') }
                }
            }
        }
    });
    
    // --- CHART 2: ROC CURVES ---
    if (activeCharts.roc) activeCharts.roc.destroy();
    const ctxRoc = document.getElementById('rocChart').getContext('2d');
    
    const rocDatasets = algos.map(algo => {
        const rocPoints = data.models[algo].roc_curve;
        return {
            label: algo.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
            data: rocPoints.map(p => ({ x: p.fpr, y: p.tpr })),
            borderColor: algoColors[algo],
            borderWidth: 2,
            fill: false,
            tension: 0.2,
            pointRadius: 2
        };
    });
    
    // Add reference line
    rocDatasets.push({
        label: 'Random Guess',
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: 'rgba(255, 255, 255, 0.25)',
        borderWidth: 1.5,
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0
    });
    
    activeCharts.roc = new Chart(ctxRoc, {
        type: 'line',
        data: {
            datasets: rocDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'False Positive Rate', color: getComputedStyle(document.body).getPropertyValue('--text-muted') },
                    ticks: { color: getComputedStyle(document.body).getPropertyValue('--text-muted') },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'True Positive Rate', color: getComputedStyle(document.body).getPropertyValue('--text-muted') },
                    ticks: { color: getComputedStyle(document.body).getPropertyValue('--text-muted') },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: getComputedStyle(document.body).getPropertyValue('--text-main') }
                }
            }
        }
    });

    // --- CHART 3: FEATURE IMPORTANCE ---
    renderFeatureImportanceChart(diseaseId);
    
    // Update Confusion Matrix display
    updateConfusionMatrix();
}

function renderFeatureImportanceChart(diseaseId) {
    const data = metricsData[diseaseId];
    if (activeCharts.importance) activeCharts.importance.destroy();
    
    const ctxImp = document.getElementById('importanceChart').getContext('2d');
    
    // Fetch random forest feature importances as default reference
    const rawImportances = data.models.random_forest.feature_importance;
    
    // Format feature labels for plotting
    const sortedFeatures = Object.entries(rawImportances)
        .sort((a, b) => b[1] - a[1]);
        
    const labels = sortedFeatures.map(item => data.feature_labels[item[0]] || item[0]);
    const values = sortedFeatures.map(item => item[1]);
    
    activeCharts.importance = new Chart(ctxImp, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Relative Importance (Random Forest)',
                data: values,
                backgroundColor: 'rgba(99, 102, 241, 0.75)',
                borderColor: 'var(--primary)',
                borderWidth: 1,
                borderRadius: 5
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: { color: getComputedStyle(document.body).getPropertyValue('--text-muted') },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                y: {
                    ticks: { color: getComputedStyle(document.body).getPropertyValue('--text-muted') },
                    grid: { display: false }
                }
            },
            plugins: {
                legend: {
                    labels: { color: getComputedStyle(document.body).getPropertyValue('--text-main') }
                }
            }
        }
    });
}

// Update Confusion Matrix Grid UI
function updateConfusionMatrix() {
    const selectDisease = document.getElementById("comparison-disease-select").value;
    const selectAlgo = document.getElementById("cm-algorithm-select").value;
    
    if (!metricsData || !metricsData[selectDisease]) return;
    
    const cm = metricsData[selectDisease].models[selectAlgo].confusion_matrix;
    
    // Set values with animated text replacement
    document.querySelector("#cm-tp .cm-val").innerText = cm.tp;
    document.querySelector("#cm-tn .cm-val").innerText = cm.tn;
    document.querySelector("#cm-fp .cm-val").innerText = cm.fp;
    document.querySelector("#cm-fn .cm-val").innerText = cm.fn;
}
