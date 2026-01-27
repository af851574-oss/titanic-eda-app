# 🚢 Titanic EDA Dashboard

Professional, browser-only Exploratory Data Analysis (EDA) web application for the Kaggle Titanic dataset.

## ✨ Key Features

- **Fully Client-Side**: No server required, runs entirely in the browser
- **TensorFlow.js Powered**: Uses TensorFlow.js for efficient data analysis operations (not just ML predictions)
- **Interactive Visualizations**: 6 Chart.js visualizations with detailed insights
- **Split Dataset Support**: Merges train.csv and test.csv with source tracking
- **Statistical Rigor**: Comprehensive descriptive statistics and correlation analysis
- **Export Functionality**: Download merged CSV and JSON summary

## 🔧 Technology Stack

| Library | Purpose | Usage |
|---------|---------|-------|
| **PapaParse** (v5.4+) | CSV Parsing | Robust file parsing with quote handling |
| **Chart.js** (v4.4+) | Visualizations | Interactive charts and graphs |
| **TensorFlow.js** (v4.15+) | Data Analysis | Statistical operations, correlations, aggregations |

## 🧠 TensorFlow.js for Data Analysis

This application demonstrates using **TensorFlow.js as a data analysis library** (like pandas/numpy in Python), not just for ML predictions:

### Tensor Operations Used:
- **`tf.moments()`** - Calculate mean and variance for statistics
- **`tf.tensor1d().sum()`** - Count valid/missing values
- **`tf.tensor.min()` / `tf.tensor.max()`** - Find min/max values
- **Vectorized correlation** - Pearson correlation using tensor operations
- **`tf.tidy()`** - Automatic memory management to prevent leaks

### Example from the code:
```javascript
// Calculate statistics using TensorFlow.js
function calculateStats(values) {
    return tf.tidy(() => {
        const tensor = tf.tensor1d(values);
        const { mean, variance } = tf.moments(tensor);
        const std = Math.sqrt(variance.dataSync()[0]);
        const min = tensor.min().dataSync()[0];
        const max = tensor.max().dataSync()[0];
        return { mean: mean.dataSync()[0], std, min, max };
    });
}
```

## 📊 Analysis Features

### 1. Data Overview
- Dataset preview (first 10 rows)
- Row/column counts
- Train/test split summary

### 2. Missing Values Analysis
- Percentage missing per column
- Visual bar chart
- Recommendations (drop/impute/analyze)

### 3. Statistical Summary
- Descriptive statistics (mean, median, std, quartiles)
- Survival comparison (survived vs died)
- All calculated using TensorFlow.js tensors

### 4. Visualizations
1. **Age Distribution** by survival status
2. **Fare Distribution** by survival status
3. **Survival by Passenger Class** (stacked bars)
4. **Survival by Sex** (stacked bars)
5. **Survival by Embarkation Port** (stacked bars)
6. **Feature Correlation Heatmap** (using tensor-based Pearson correlation)

### 5. Key Insights
Automated insights generation revealing:
- Sex as strongest survival predictor (74% female vs 19% male survival)
- Class impact (63% 1st class vs 24% 3rd class survival)
- Age factor analysis

## 🚀 GitHub Pages Deployment

### Quick Setup:
1. Create a new public GitHub repository
2. Upload `index.html` and `app.js`
3. Go to **Settings** → **Pages**
4. Set Source: **Deploy from branch `main`**, folder **`/root`**
5. Save and wait 2-3 minutes
6. Access at: `https://<username>.github.io/<repo-name>/`

### Testing Locally:
Simply open `index.html` in any modern browser (Chrome, Firefox, Edge, Safari). No build step or local server required!

## 📁 Dataset

### Kaggle Titanic Competition:
- **Train**: [train.csv](https://www.kaggle.com/c/titanic/download/train.csv) (with 'Survived' label)
- **Test**: [test.csv](https://www.kaggle.com/c/titanic/download/test.csv) (without 'Survived')

### Features:
- **Numerical**: Age, Fare, SibSp, Parch, Pclass
- **Categorical**: Sex, Embarked
- **Target**: Survived (0 = Died, 1 = Survived)

## 💻 Usage

1. Open the application in your browser
2. Click "Choose train.csv" and select the training dataset
3. Click "Choose test.csv" and select the test dataset
4. Click "Load & Analyze Data"
5. Explore the interactive visualizations and insights
6. Export merged data or summary statistics as needed

## 🐛 Debugging

### Memory Leak Detection:
Open browser console and call:
```javascript
logTensorMemory()
```

This will display TensorFlow.js memory usage:
- `numTensors`: Should remain stable (not growing indefinitely)
- `numBytes`: Total memory allocated
- `numDataBuffers`: Number of data buffers

All tensor operations use `tf.tidy()` for automatic cleanup, ensuring no memory leaks.

## 🎯 Code Quality

- **Modular functions** with single responsibility
- **JSDoc comments** for all functions
- **Proper error handling** with user-friendly messages
- **Memory management** using `tf.tidy()` throughout
- **Responsive design** (mobile + desktop)

## 📝 Reusability

To adapt this app for other split datasets:
1. Update `DATA_CONFIG.TARGET` if different label column
2. Modify `DATA_CONFIG.FEATURES` with new column names
3. Update chart titles/labels in visualization functions

## 📄 License

MIT License - Feel free to use for educational or commercial purposes.

## 🙏 Credits

Built with ❤️ using:
- [PapaParse](https://www.papaparse.com/) by Matt Holt
- [Chart.js](https://www.chartjs.org/) by Chart.js Contributors
- [TensorFlow.js](https://www.tensorflow.org/js) by Google Brain Team
- Data from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
