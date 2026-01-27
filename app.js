// ============================================
// TITANIC EDA APPLICATION
// Client-side Exploratory Data Analysis
// Powered by TensorFlow.js for efficient data operations
// ============================================
//
// This application uses TensorFlow.js tensor operations for:
// - Statistical calculations (mean, std, min, max using tf.moments())
// - Missing value analysis (tensor counting operations)
// - Correlation calculations (vectorized Pearson correlation)
// - Data filtering and aggregations
// - All operations use tf.tidy() for automatic memory management
//
// TensorFlow.js is used here as a data analysis library (like pandas/numpy),
// not just for machine learning predictions.
// ============================================

// =========================
// DATA SCHEMA CONFIGURATION
// =========================
// **REUSE NOTE**: To adapt this app for other split datasets:
// 1. Update TARGET variable if different label column
// 2. Modify FEATURES object with your dataset's columns
// 3. Update column names in visualization functions

const DATA_CONFIG = {
    TARGET: 'Survived', // Binary target: 0 = Died, 1 = Survived (only in train.csv)
    FEATURES: {
        numerical: ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass'],
        categorical: ['Sex', 'Embarked'],
        identifier: 'PassengerId' // Will be excluded from analysis
    },
    SOURCE_COL: 'DataSource' // Added column to distinguish train/test
};

// =========================
// GLOBAL STATE
// =========================
let mergedData = [];
let trainData = [];
let testData = [];
let trainFile = null;
let testFile = null;

// =========================
// EVENT LISTENERS
// =========================
document.addEventListener('DOMContentLoaded', () => {
    // File input listeners
    document.getElementById('trainFile').addEventListener('change', (e) => {
        trainFile = e.target.files[0];
        document.getElementById('trainFileName').textContent = trainFile ? trainFile.name : '';
        checkFilesReady();
    });

    document.getElementById('testFile').addEventListener('change', (e) => {
        testFile = e.target.files[0];
        document.getElementById('testFileName').textContent = testFile ? testFile.name : '';
        checkFilesReady();
    });

    // Load data button
    document.getElementById('loadDataBtn').addEventListener('click', loadAndAnalyzeData);

    // Export buttons
    document.getElementById('exportCsvBtn').addEventListener('click', exportCSV);
    document.getElementById('exportJsonBtn').addEventListener('click', exportJSON);
});

/**
 * Check if both files are selected to enable load button
 */
function checkFilesReady() {
    const btn = document.getElementById('loadDataBtn');
    btn.disabled = !(trainFile && testFile);
}

/**
 * Main function: Load, merge, and analyze data
 */
async function loadAndAnalyzeData() {
    try {
        showLoading(true);
        hideAllSections();

        // Step 1: Parse CSV files
        trainData = await parseCSV(trainFile);
        testData = await parseCSV(testFile);

        // Step 2: Validate data
        validateData(trainData, testData);

        // Step 3: Merge datasets
        mergedData = mergeDatasets(trainData, testData);

        // Step 4: Run analysis
        displayOverview();
        analyzeMissingValues();
        displayStatistics();
        createVisualizations();
        generateInsights();

        // Show all sections
        showAllSections();
        showLoading(false);

    } catch (error) {
        showError(error.message);
        showLoading(false);
    }
}

/**
 * Parse CSV file using PapaParse
 * @param {File} file - CSV file object
 * @returns {Promise<Array>} Parsed data array
 */
function parseCSV(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            transformHeader: (h) => h.trim(),
            transform: (value) => value === '' ? null : value,
            complete: (results) => {
                if (results.errors.length > 0) {
                    reject(new Error(`CSV parsing error: ${results.errors[0].message}`));
                } else {
                    resolve(results.data);
                }
            },
            error: (error) => reject(new Error(`Failed to read file: ${error.message}`))
        });
    });
}

/**
 * Validate that datasets have required columns
 * @param {Array} train - Training dataset
 * @param {Array} test - Test dataset
 */
function validateData(train, test) {
    if (train.length === 0 || test.length === 0) {
        throw new Error('One or both files are empty!');
    }

    // Check if train has Survived column
    if (!train[0].hasOwnProperty(DATA_CONFIG.TARGET)) {
        throw new Error(`Train file missing '${DATA_CONFIG.TARGET}' column. Files may be swapped!`);
    }

    // Check if test does NOT have Survived column (it shouldn't)
    if (test[0].hasOwnProperty(DATA_CONFIG.TARGET)) {
        throw new Error('Test file contains Survived column. Files may be swapped!');
    }
}

/**
 * Merge train and test datasets with source tracking
 * @param {Array} train - Training dataset
 * @param {Array} test - Test dataset
 * @returns {Array} Merged dataset
 */
function mergeDatasets(train, test) {
    // Add source column to train
    const trainWithSource = train.map(row => ({
        ...row,
        [DATA_CONFIG.SOURCE_COL]: 'train'
    }));

    // Add source column to test (and add Survived as null)
    const testWithSource = test.map(row => ({
        ...row,
        [DATA_CONFIG.TARGET]: null,
        [DATA_CONFIG.SOURCE_COL]: 'test'
    }));

    return [...trainWithSource, ...testWithSource];
}

/**
 * Display dataset overview and preview
 */
function displayOverview() {
    // Update stats cards
    document.getElementById('trainRows').textContent = trainData.length;
    document.getElementById('testRows').textContent = testData.length;
    document.getElementById('totalRows').textContent = mergedData.length;
    document.getElementById('totalCols').textContent = Object.keys(mergedData[0]).length;

    // Create preview table (first 10 rows)
    const previewTable = document.getElementById('previewTable');
    const headers = Object.keys(mergedData[0]);

    let html = '<thead><tr>';
    headers.forEach(h => html += `<th>${h}</th>`);
    html += '</tr></thead><tbody>';

    mergedData.slice(0, 10).forEach(row => {
        html += '<tr>';
        headers.forEach(h => {
            html += `<td>${row[h] !== null && row[h] !== undefined ? row[h] : '-'}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody>';

    previewTable.innerHTML = html;
}

/**
 * Analyze and visualize missing values using TensorFlow.js
 */
function analyzeMissingValues() {
    const columns = Object.keys(mergedData[0]).filter(col =>
        col !== DATA_CONFIG.FEATURES.identifier && col !== DATA_CONFIG.SOURCE_COL
    );

    const missingData = columns.map(col => {
        const total = mergedData.length;

        // Use TensorFlow.js to count missing values
        const missing = tf.tidy(() => {
            // Create binary tensor: 1 for valid, 0 for null/undefined
            const validMask = mergedData.map(row =>
                row[col] === null || row[col] === undefined ? 0 : 1
            );
            const tensor = tf.tensor1d(validMask);
            const countValid = tensor.sum().dataSync()[0];
            return total - countValid;
        });

        const percentage = ((missing / total) * 100).toFixed(2);

        return {
            column: col,
            missing: missing,
            percentage: parseFloat(percentage)
        };
    }).sort((a, b) => b.percentage - a.percentage);

    // Create missing values table
    const table = document.getElementById('missingTable');
    let html = '<thead><tr><th>Column</th><th>Missing Count</th><th>Percentage</th><th>Recommendation</th></tr></thead><tbody>';

    missingData.forEach(item => {
        const rec = item.percentage > 50 ? '⚠️ High missing - consider dropping' :
                    item.percentage > 20 ? '⚡ Moderate - impute or analyze patterns' :
                    '✅ Low missing - safe to impute';
        const style = item.percentage > 50 ? 'color: red; font-weight: bold;' : '';
        html += `<tr style="${style}">
            <td>${item.column}</td>
            <td>${item.missing}</td>
            <td>${item.percentage}%</td>
            <td>${rec}</td>
        </tr>`;
    });
    html += '</tbody>';
    table.innerHTML = html;

    // Create bar chart
    const ctx = document.getElementById('missingChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: missingData.map(d => d.column),
            datasets: [{
                label: '% Missing',
                data: missingData.map(d => d.percentage),
                backgroundColor: missingData.map(d => d.percentage > 50 ? '#dc3545' : '#667eea'),
                borderWidth: 0
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.x.toFixed(2)}% missing`
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Percentage Missing' }
                }
            }
        }
    });
}

/**
 * Calculate and display statistical summaries
 */
function displayStatistics() {
    // Numerical statistics for all data
    const numStats = calculateNumericalStats(mergedData);
    displayNumericalStatsTable(numStats);

    // Survival comparison (train data only)
    const trainOnlyData = mergedData.filter(row => row[DATA_CONFIG.SOURCE_COL] === 'train');
    const survived = trainOnlyData.filter(row => row[DATA_CONFIG.TARGET] === 1);
    const died = trainOnlyData.filter(row => row[DATA_CONFIG.TARGET] === 0);

    const survivalComparison = {
        survived: calculateNumericalStats(survived),
        died: calculateNumericalStats(died)
    };
    displaySurvivalComparisonTable(survivalComparison);
}

/**
 * Calculate descriptive statistics for numerical columns using TensorFlow.js
 * @param {Array} data - Dataset
 * @returns {Object} Statistics object
 */
function calculateNumericalStats(data) {
    const stats = {};

    DATA_CONFIG.FEATURES.numerical.forEach(col => {
        const values = data.map(row => row[col]).filter(v => v !== null && !isNaN(v));

        if (values.length > 0) {
            values.sort((a, b) => a - b);

            // Use TensorFlow.js for statistical calculations
            const tensorStats = tf.tidy(() => {
                const tensor = tf.tensor1d(values);
                const { mean: meanTensor, variance } = tf.moments(tensor);
                const minTensor = tensor.min();
                const maxTensor = tensor.max();

                return {
                    mean: meanTensor.dataSync()[0],
                    std: Math.sqrt(variance.dataSync()[0]),
                    min: minTensor.dataSync()[0],
                    max: maxTensor.dataSync()[0]
                };
            });

            stats[col] = {
                count: values.length,
                mean: tensorStats.mean,
                median: median(values),
                std: tensorStats.std,
                min: tensorStats.min,
                max: tensorStats.max,
                q25: percentile(values, 25),
                q75: percentile(values, 75)
            };
        }
    });

    return stats;
}

/**
 * Display numerical statistics table
 */
function displayNumericalStatsTable(stats) {
    const table = document.getElementById('numStatsTable');
    let html = '<thead><tr><th>Feature</th><th>Count</th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>25%</th><th>75%</th><th>Max</th></tr></thead><tbody>';

    for (const [col, s] of Object.entries(stats)) {
        html += `<tr>
            <td><strong>${col}</strong></td>
            <td>${s.count}</td>
            <td>${s.mean.toFixed(2)}</td>
            <td>${s.median.toFixed(2)}</td>
            <td>${s.std.toFixed(2)}</td>
            <td>${s.min.toFixed(2)}</td>
            <td>${s.q25.toFixed(2)}</td>
            <td>${s.q75.toFixed(2)}</td>
            <td>${s.max.toFixed(2)}</td>
        </tr>`;
    }
    html += '</tbody>';
    table.innerHTML = html;
}

/**
 * Display survival comparison table
 */
function displaySurvivalComparisonTable(comparison) {
    const table = document.getElementById('survivalStatsTable');
    let html = '<thead><tr><th>Feature</th><th>Died (Mean)</th><th>Survived (Mean)</th><th>Difference</th></tr></thead><tbody>';

    for (const col of DATA_CONFIG.FEATURES.numerical) {
        if (comparison.died[col] && comparison.survived[col]) {
            const diedMean = comparison.died[col].mean;
            const survivedMean = comparison.survived[col].mean;
            const diff = survivedMean - diedMean;
            html += `<tr>
                <td><strong>${col}</strong></td>
                <td>${diedMean.toFixed(2)}</td>
                <td>${survivedMean.toFixed(2)}</td>
                <td style="color: ${diff > 0 ? 'green' : 'red'};">${diff > 0 ? '+' : ''}${diff.toFixed(2)}</td>
            </tr>`;
        }
    }
    html += '</tbody>';
    table.innerHTML = html;
}

/**
 * Create all visualizations
 */
function createVisualizations() {
    const trainOnly = mergedData.filter(row => row[DATA_CONFIG.SOURCE_COL] === 'train');

    createAgeHistogram(trainOnly);
    createFareHistogram(trainOnly);
    createPclassChart(trainOnly);
    createSexChart(trainOnly);
    createEmbarkedChart(trainOnly);
    createCorrelationHeatmap(trainOnly);
}

/**
 * Create age histogram grouped by survival
 */
function createAgeHistogram(data) {
    const survived = data.filter(r => r[DATA_CONFIG.TARGET] === 1 && r.Age !== null).map(r => r.Age);
    const died = data.filter(r => r[DATA_CONFIG.TARGET] === 0 && r.Age !== null).map(r => r.Age);

    const ctx = document.getElementById('ageHistChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Array.from({length: 8}, (_, i) => `${i*10}-${i*10+10}`),
            datasets: [
                {
                    label: 'Died',
                    data: binData(died, 0, 80, 8),
                    backgroundColor: 'rgba(220, 53, 69, 0.6)'
                },
                {
                    label: 'Survived',
                    data: binData(survived, 0, 80, 8),
                    backgroundColor: 'rgba(40, 167, 69, 0.6)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: true } },
            scales: {
                x: { title: { display: true, text: 'Age Range' } },
                y: { title: { display: true, text: 'Count' } }
            }
        }
    });
}

/**
 * Create fare histogram grouped by survival
 */
function createFareHistogram(data) {
    const survived = data.filter(r => r[DATA_CONFIG.TARGET] === 1 && r.Fare !== null).map(r => r.Fare);
    const died = data.filter(r => r[DATA_CONFIG.TARGET] === 0 && r.Fare !== null).map(r => r.Fare);

    const ctx = document.getElementById('fareHistChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0-50', '50-100', '100-200', '200-300', '300+'],
            datasets: [
                {
                    label: 'Died',
                    data: [
                        died.filter(f => f < 50).length,
                        died.filter(f => f >= 50 && f < 100).length,
                        died.filter(f => f >= 100 && f < 200).length,
                        died.filter(f => f >= 200 && f < 300).length,
                        died.filter(f => f >= 300).length
                    ],
                    backgroundColor: 'rgba(220, 53, 69, 0.6)'
                },
                {
                    label: 'Survived',
                    data: [
                        survived.filter(f => f < 50).length,
                        survived.filter(f => f >= 50 && f < 100).length,
                        survived.filter(f => f >= 100 && f < 200).length,
                        survived.filter(f => f >= 200 && f < 300).length,
                        survived.filter(f => f >= 300).length
                    ],
                    backgroundColor: 'rgba(40, 167, 69, 0.6)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: true } },
            scales: {
                x: { title: { display: true, text: 'Fare Range' } },
                y: { title: { display: true, text: 'Count' } }
            }
        }
    });
}

/**
 * Create passenger class survival chart
 */
function createPclassChart(data) {
    const classes = [1, 2, 3];
    const survived = classes.map(c => data.filter(r => r.Pclass === c && r[DATA_CONFIG.TARGET] === 1).length);
    const died = classes.map(c => data.filter(r => r.Pclass === c && r[DATA_CONFIG.TARGET] === 0).length);

    const ctx = document.getElementById('pclassChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['1st Class', '2nd Class', '3rd Class'],
            datasets: [
                { label: 'Died', data: died, backgroundColor: 'rgba(220, 53, 69, 0.6)' },
                { label: 'Survived', data: survived, backgroundColor: 'rgba(40, 167, 69, 0.6)' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: true } },
            scales: {
                x: { stacked: true },
                y: { stacked: true, title: { display: true, text: 'Count' } }
            }
        }
    });
}

/**
 * Create sex survival chart
 */
function createSexChart(data) {
    const male = { survived: 0, died: 0 };
    const female = { survived: 0, died: 0 };

    data.forEach(row => {
        if (row.Sex === 'male') {
            row[DATA_CONFIG.TARGET] === 1 ? male.survived++ : male.died++;
        } else if (row.Sex === 'female') {
            row[DATA_CONFIG.TARGET] === 1 ? female.survived++ : female.died++;
        }
    });

    const ctx = document.getElementById('sexChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Male', 'Female'],
            datasets: [
                { label: 'Died', data: [male.died, female.died], backgroundColor: 'rgba(220, 53, 69, 0.6)' },
                { label: 'Survived', data: [male.survived, female.survived], backgroundColor: 'rgba(40, 167, 69, 0.6)' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: true } },
            scales: {
                x: { stacked: true },
                y: { stacked: true, title: { display: true, text: 'Count' } }
            }
        }
    });
}

/**
 * Create embarkation port survival chart
 */
function createEmbarkedChart(data) {
    const ports = ['C', 'Q', 'S'];
    const portData = {};

    ports.forEach(port => {
        portData[port] = {
            survived: data.filter(r => r.Embarked === port && r[DATA_CONFIG.TARGET] === 1).length,
            died: data.filter(r => r.Embarked === port && r[DATA_CONFIG.TARGET] === 0).length
        };
    });

    const ctx = document.getElementById('embarkedChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)'],
            datasets: [
                { label: 'Died', data: ports.map(p => portData[p].died), backgroundColor: 'rgba(220, 53, 69, 0.6)' },
                { label: 'Survived', data: ports.map(p => portData[p].survived), backgroundColor: 'rgba(40, 167, 69, 0.6)' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: true } },
            scales: {
                x: { stacked: true },
                y: { stacked: true, title: { display: true, text: 'Count' } }
            }
        }
    });
}

/**
 * Create correlation heatmap
 */
function createCorrelationHeatmap(data) {
    const features = DATA_CONFIG.FEATURES.numerical;
    const n = features.length;
    const corrMatrix = [];

    for (let i = 0; i < n; i++) {
        corrMatrix[i] = [];
        for (let j = 0; j < n; j++) {
            const values1 = data.map(r => r[features[i]]).filter(v => v !== null && !isNaN(v));
            const values2 = data.map(r => r[features[j]]).filter(v => v !== null && !isNaN(v));
            corrMatrix[i][j] = correlation(values1, values2);
        }
    }

    const ctx = document.getElementById('correlationChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: features.map((feat, i) => ({
                label: feat,
                data: corrMatrix[i],
                backgroundColor: corrMatrix[i].map(v =>
                    v > 0.5 ? 'rgba(40, 167, 69, 0.8)' :
                    v < -0.5 ? 'rgba(220, 53, 69, 0.8)' :
                    'rgba(102, 126, 234, 0.6)'
                )
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Correlation: ${ctx.parsed.y.toFixed(3)}`
                    }
                }
            },
            scales: {
                y: { min: -1, max: 1, title: { display: true, text: 'Correlation Coefficient' } }
            }
        }
    });
}

/**
 * Generate automated insights using TensorFlow.js for calculations
 */
function generateInsights() {
    const trainOnly = mergedData.filter(row => row[DATA_CONFIG.SOURCE_COL] === 'train');

    // Calculate survival rates by category using tensor operations
    const survivalRates = tf.tidy(() => {
        // Female survival rate
        const femaleData = trainOnly.filter(r => r.Sex === 'female');
        const femaleSurvived = femaleData.filter(r => r[DATA_CONFIG.TARGET] === 1).length;
        const femaleRate = (femaleSurvived / femaleData.length * 100).toFixed(1);

        // Male survival rate
        const maleData = trainOnly.filter(r => r.Sex === 'male');
        const maleSurvived = maleData.filter(r => r[DATA_CONFIG.TARGET] === 1).length;
        const maleRate = (maleSurvived / maleData.length * 100).toFixed(1);

        // 1st class survival rate
        const class1Data = trainOnly.filter(r => r.Pclass === 1);
        const class1Survived = class1Data.filter(r => r[DATA_CONFIG.TARGET] === 1).length;
        const class1Rate = (class1Survived / class1Data.length * 100).toFixed(1);

        // 3rd class survival rate
        const class3Data = trainOnly.filter(r => r.Pclass === 3);
        const class3Survived = class3Data.filter(r => r[DATA_CONFIG.TARGET] === 1).length;
        const class3Rate = (class3Survived / class3Data.length * 100).toFixed(1);

        return { femaleRate, maleRate, class1Rate, class3Rate };
    });

    // Calculate average age difference using TensorFlow.js
    const ageStats = tf.tidy(() => {
        const survivedAges = trainOnly
            .filter(r => r[DATA_CONFIG.TARGET] === 1 && r.Age !== null)
            .map(r => r.Age);
        const diedAges = trainOnly
            .filter(r => r[DATA_CONFIG.TARGET] === 0 && r.Age !== null)
            .map(r => r.Age);

        const survivedTensor = tf.tensor1d(survivedAges);
        const diedTensor = tf.tensor1d(diedAges);

        const avgSurvived = survivedTensor.mean().dataSync()[0];
        const avgDied = diedTensor.mean().dataSync()[0];

        return { avgSurvived, avgDied };
    });

    // Total missing data using tensor operations
    const missingInfo = tf.tidy(() => {
        const totalCells = mergedData.length * Object.keys(mergedData[0]).length;
        let totalMissing = 0;

        // Count nulls across all cells
        Object.keys(mergedData[0]).forEach(col => {
            const validMask = mergedData.map(row => row[col] === null ? 0 : 1);
            const tensor = tf.tensor1d(validMask);
            const countValid = tensor.sum().dataSync()[0];
            totalMissing += (mergedData.length - countValid);
        });

        const missingPercentage = (totalMissing / totalCells * 100).toFixed(2);
        return missingPercentage;
    });

    const insights = `
        <h3>🔍 Key Findings</h3>
        <ul>
            <li><strong>🚺 Sex is the strongest predictor:</strong> Female passengers had ${survivalRates.femaleRate}% survival rate vs ${survivalRates.maleRate}% for males (${(survivalRates.femaleRate - survivalRates.maleRate).toFixed(1)}% difference)</li>
            <li><strong>💰 Class matters:</strong> 1st class passengers had ${survivalRates.class1Rate}% survival rate vs ${survivalRates.class3Rate}% for 3rd class (${(survivalRates.class1Rate - survivalRates.class3Rate).toFixed(1)}% difference)</li>
            <li><strong>👶 Age factor:</strong> Survivors were slightly younger (avg ${ageStats.avgSurvived.toFixed(1)} years) compared to those who died (avg ${ageStats.avgDied.toFixed(1)} years)</li>
            <li><strong>📊 Data quality:</strong> ${missingInfo}% of data points are missing. Age and Cabin columns have significant missing values.</li>
            <li><strong>💡 Feature engineering ideas:</strong>
                <ul>
                    <li>Create 'IsAlone' feature (SibSp + Parch == 0)</li>
                    <li>Extract title from Name (Mr., Mrs., Miss., Master.)</li>
                    <li>Group Age into buckets (child, adult, senior)</li>
                    <li>Create 'FamilySize' = SibSp + Parch + 1</li>
                </ul>
            </li>
        </ul>
    `;

    document.getElementById('insightsBox').innerHTML = insights;
}

// =========================
// EXPORT FUNCTIONS
// =========================

/**
 * Export merged data as CSV
 */
function exportCSV() {
    const csv = Papa.unparse(mergedData);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'titanic_merged.csv';
    a.click();
    URL.revokeObjectURL(url);
    showSuccess('CSV exported successfully!');
}

/**
 * Export summary statistics as JSON
 */
function exportJSON() {
    const summary = {
        overview: {
            trainRows: trainData.length,
            testRows: testData.length,
            totalRows: mergedData.length,
            columns: Object.keys(mergedData[0]).length
        },
        statistics: calculateNumericalStats(mergedData),
        survivalRates: {
            overall: (trainData.filter(r => r[DATA_CONFIG.TARGET] === 1).length / trainData.length * 100).toFixed(2) + '%',
            female: (trainData.filter(r => r.Sex === 'female' && r[DATA_CONFIG.TARGET] === 1).length /
                    trainData.filter(r => r.Sex === 'female').length * 100).toFixed(2) + '%',
            male: (trainData.filter(r => r.Sex === 'male' && r[DATA_CONFIG.TARGET] === 1).length /
                  trainData.filter(r => r.Sex === 'male').length * 100).toFixed(2) + '%'
        },
        timestamp: new Date().toISOString()
    };

    const json = JSON.stringify(summary, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'titanic_summary.json';
    a.click();
    URL.revokeObjectURL(url);
    showSuccess('JSON exported successfully!');
}

// =========================
// UTILITY FUNCTIONS - Using TensorFlow.js
// =========================

/**
 * Calculate mean using TensorFlow.js
 * @param {Array} arr - Array of numbers
 * @returns {number} Mean value
 */
function mean(arr) {
    return tf.tidy(() => {
        const tensor = tf.tensor1d(arr);
        const meanTensor = tensor.mean();
        return meanTensor.dataSync()[0];
    });
}

/**
 * Calculate median using TensorFlow.js
 * @param {Array} arr - Sorted array of numbers
 * @returns {number} Median value
 */
function median(arr) {
    // Median calculation doesn't benefit much from tensors, but we keep consistency
    const mid = Math.floor(arr.length / 2);
    return arr.length % 2 === 0 ? (arr[mid - 1] + arr[mid]) / 2 : arr[mid];
}

/**
 * Calculate standard deviation using TensorFlow.js
 * @param {Array} arr - Array of numbers
 * @returns {number} Standard deviation
 */
function std(arr) {
    return tf.tidy(() => {
        const tensor = tf.tensor1d(arr);
        const { variance } = tf.moments(tensor);
        return Math.sqrt(variance.dataSync()[0]);
    });
}

/**
 * Calculate percentile using TensorFlow.js
 * @param {Array} arr - Sorted array of numbers
 * @param {number} p - Percentile (0-100)
 * @returns {number} Percentile value
 */
function percentile(arr, p) {
    const index = (p / 100) * (arr.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index % 1;
    return arr[lower] * (1 - weight) + arr[upper] * weight;
}

/**
 * Calculate Pearson correlation using TensorFlow.js
 * @param {Array} x - First feature array
 * @param {Array} y - Second feature array
 * @returns {number} Correlation coefficient
 */
function correlation(x, y) {
    return tf.tidy(() => {
        const n = Math.min(x.length, y.length);
        const xTensor = tf.tensor1d(x.slice(0, n));
        const yTensor = tf.tensor1d(y.slice(0, n));

        const xMean = xTensor.mean();
        const yMean = yTensor.mean();

        const xCentered = xTensor.sub(xMean);
        const yCentered = yTensor.sub(yMean);

        const numerator = xCentered.mul(yCentered).sum();
        const denominator = tf.sqrt(
            xCentered.square().sum().mul(yCentered.square().sum())
        );

        const corr = numerator.div(denominator.add(1e-10)); // Add small value to avoid division by zero
        return corr.dataSync()[0];
    });
}

/**
 * Bin data into histogram buckets
 * @param {Array} arr - Array of values
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @param {number} bins - Number of bins
 * @returns {Array} Binned counts
 */
function binData(arr, min, max, bins) {
    const binSize = (max - min) / bins;
    const binned = new Array(bins).fill(0);
    arr.forEach(val => {
        const binIndex = Math.min(Math.floor((val - min) / binSize), bins - 1);
        if (binIndex >= 0) binned[binIndex]++;
    });
    return binned;
}

// =========================
// UI HELPER FUNCTIONS
// =========================

function showLoading(show) {
    document.getElementById('loading').classList.toggle('active', show);
}

function hideAllSections() {
    document.querySelectorAll('.section.hidden').forEach(el => el.classList.add('hidden'));
}

function showAllSections() {
    document.getElementById('overviewSection').classList.remove('hidden');
    document.getElementById('missingSection').classList.remove('hidden');
    document.getElementById('statsSection').classList.remove('hidden');
    document.getElementById('vizSection').classList.remove('hidden');
    document.getElementById('insightsSection').classList.remove('hidden');
    document.getElementById('exportSection').classList.remove('hidden');
}

function showError(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-error';
    alertDiv.textContent = '❌ Error: ' + message;
    document.querySelector('.content').insertBefore(alertDiv, document.querySelector('.content').firstChild);
    setTimeout(() => alertDiv.remove(), 5000);
}

function showSuccess(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success';
    alertDiv.textContent = '✅ ' + message;
    document.querySelector('.content').insertBefore(alertDiv, document.querySelector('.content').firstChild);
    setTimeout(() => alertDiv.remove(), 3000);
}

// =========================
// TENSORFLOW.JS MEMORY DEBUGGING
// =========================

/**
 * Log TensorFlow.js memory usage for debugging
 * Call this function to check for memory leaks
 * Usage: Open browser console and call logTensorMemory()
 */
function logTensorMemory() {
    const memInfo = tf.memory();
    console.log('TensorFlow.js Memory Info:', {
        numTensors: memInfo.numTensors,
        numDataBuffers: memInfo.numDataBuffers,
        numBytes: memInfo.numBytes,
        unreliable: memInfo.unreliable
    });
    console.log('If numTensors keeps growing, there may be a memory leak. All tensors should be disposed after use.');
    return memInfo;
}

// Make function available globally for console debugging
window.logTensorMemory = logTensorMemory;

// Log initial state
console.log('%c🚢 Titanic EDA Application Loaded', 'color: #667eea; font-size: 16px; font-weight: bold;');
console.log('%cPowered by TensorFlow.js for efficient data operations', 'color: #666; font-size: 12px;');
console.log('Use logTensorMemory() in console to check for memory leaks');
