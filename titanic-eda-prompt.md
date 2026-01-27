## Role & Context
You are a senior full-stack engineer specializing in interactive data visualization and statistical analysis. You have expertise in creating production-ready, maintainable web applications that follow best practices.

## Core Objective
Create a professional, browser-only Exploratory Data Analysis (EDA) web application for the Kaggle Titanic dataset that is:
- **Fully client-side** (no server required)
- **GitHub Pages deployable**
- **Extensible** and reusable for other split datasets
- **Statistically rigorous** with proper data science methodology
- **TensorFlow.js powered** - Use TensorFlow.js for data analysis operations (statistics, transformations, filtering) instead of manual JavaScript array manipulation
- **User-friendly** with clear visualizations and insights

**Key Requirement:** Leverage TensorFlow.js tensor operations for all data processing tasks - treat it like you would use pandas/numpy in Python for EDA, not just for ML predictions.

## Technical Requirements

### 1. Architecture & Code Organization
**Strict Output Format:**
- **File 1: `index.html`** - HTML structure, CSS styling, UI layout
- **File 2: `app.js`** - All JavaScript logic, data processing, visualization
- **File 3: `README.md`** - Deployment instructions and dataset notes

**Technology Stack:**
- **PapaParse** (v5.4+) - Robust CSV parsing with proper quote handling
- **Chart.js** (v4+) - Modern, responsive charts with tooltips
- **TensorFlow.js** (v4.15+) - **Required for data analysis operations** (not just ML prediction)
  - Use tf.data API for data pipelines and transformations
  - Use tensor operations for statistical calculations (mean, std, correlation, etc.)
  - Leverage vectorized operations instead of manual JavaScript loops
  - Use tf.data.csv() or convert parsed data to tensors for efficient processing
- All via CDN for zero-build deployment

### 2. Data Schema & Business Logic

**Dataset Context:**
- **Source:** Kaggle Titanic Competition
- **Split:** train.csv (with 'Survived' label), test.csv (without 'Survived')
- **Goal:** Merge both for comprehensive EDA, mark source for traceability

**Schema Definition:**
```javascript
// Target variable (only in train.csv)
const TARGET = 'Survived'; // 0 = Died, 1 = Survived

// Feature columns (present in both files)
const FEATURES = {
  numerical: ['Age', 'Fare', 'SibSp', 'Parch'],
  categorical: ['Pclass', 'Sex', 'Embarked'],
  identifier: 'PassengerId' // Exclude from analysis
};

// Meta column added during merge
const SOURCE_COL = 'DataSource'; // Values: 'train' | 'test'
```

**📝 Note for Reusability:** To adapt this app for other split datasets:
1. Update URLs in `DATA_CONFIG`
2. Modify `FEATURES` object with new column names
3. Update `TARGET` if different label column
4. Adjust chart titles/labels in visualization functions

### 3. Functional Requirements

#### Phase 1: Data Loading & Preparation
- **Input:** Two file upload inputs for train.csv and test.csv
- **Validation:**
  - Check for required columns (Schema validation)
  - Alert user if files are swapped (train missing 'Survived')
  - Handle malformed CSV (commas in quotes, missing values)
- **Processing:**
  ```javascript
  // PapaParse config for robust parsing
  {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    transformHeader: (h) => h.trim(), // Remove whitespace
    transform: (value) => value === '' ? null : value // Handle empty strings
  }

  // Convert to TensorFlow.js data structure for efficient processing
  // Use tf.tensor() or tf.data.array() to create tensors from parsed data
  // Example: const dataTensor = tf.data.array(parsedData);
  ```
- **Merge Logic:**
  1. Add 'DataSource' column to distinguish train/test
  2. Combine into single array for unified analysis
  3. Display merge summary: train rows, test rows, total rows
  4. **Use TensorFlow.js tensor operations for merging and transformations**

#### Phase 2: Data Quality Assessment
**Missing Values Analysis:**
- Calculate % missing per column using tensor operations
- Visualize with horizontal bar chart (sorted by % missing, descending)
- Highlight columns with >50% missing in red
- Display table: Column | Count Missing | % Missing | Recommendation
- **Use TensorFlow.js:**
  ```javascript
  // Example: Count nulls using tensor operations
  const columnTensor = tf.tensor1d(columnData.map(v => v === null ? 0 : 1));
  const countValid = columnTensor.sum().dataSync()[0];
  const countMissing = totalRows - countValid;
  columnTensor.dispose(); // Clean up memory
  ```

**Outlier Detection:**
- For numerical features: Calculate IQR using tf.moments() for mean/variance
- Use tensor operations for efficient quartile calculations
- Display box plots with outliers marked
- Show outlier count per feature

#### Phase 3: Statistical Summary
**Descriptive Statistics (Numerical Features):**
- Overall dataset: count, mean, median, std, min, 25%, 50%, 75%, max
- **Use TensorFlow.js for all calculations:**
  ```javascript
  // Example: Calculate statistics using TensorFlow.js
  const featureTensor = tf.tensor1d(validValues);
  const { mean, variance } = tf.moments(featureTensor);
  const meanValue = mean.dataSync()[0];
  const stdValue = Math.sqrt(variance.dataSync()[0]);

  // For min/max
  const minValue = featureTensor.min().dataSync()[0];
  const maxValue = featureTensor.max().dataSync()[0];

  // Don't forget to dispose tensors
  featureTensor.dispose();
  mean.dispose();
  variance.dispose();
  ```
- **Grouped by Survival Status** (train data only):
  - Use tf.gather() or tensor slicing to filter by survival status
  - Mean Age for Survived=0 vs Survived=1
  - Mean Fare for Survived=0 vs Survived=1
  - Display side-by-side comparison tables

**Categorical Analysis:**
- Frequency distribution (value counts) using tensor operations
- **Survival rate by category** (e.g., % survived in 1st class vs 3rd class)
- Use boolean masking with tensors for efficient grouping
- Display as interactive tables with click-to-filter

#### Phase 4: Advanced Visualizations
**Required Charts:**

1. **Distribution Charts:**
   - Histograms: Age, Fare (with KDE overlay if possible)
   - Separate by survival status (stacked or side-by-side)

2. **Categorical Charts:**
   - Bar charts: Sex, Pclass, Embarked
   - Show survival counts within each category (stacked bars)

3. **Correlation Analysis:**
   - Heatmap: Correlation matrix for numerical features
   - **Use TensorFlow.js for correlation calculation:**
     ```javascript
     // Example: Calculate Pearson correlation using tensors
     const x = tf.tensor1d(feature1Values);
     const y = tf.tensor1d(feature2Values);

     const xMean = x.mean();
     const yMean = y.mean();

     const xCentered = x.sub(xMean);
     const yCentered = y.sub(yMean);

     const numerator = xCentered.mul(yCentered).sum();
     const denominator = tf.sqrt(xCentered.square().sum().mul(yCentered.square().sum()));

     const correlation = numerator.div(denominator).dataSync()[0];

     // Dispose all tensors
     [x, y, xMean, yMean, xCentered, yCentered, numerator, denominator].forEach(t => t.dispose());
     ```
   - Annotate cells with correlation coefficients
   - Use diverging color scheme (red=negative, blue=positive)

4. **Survival Analysis:**
   - Survival rate by Age group (0-10, 10-20, 20-30, etc.)
   - Survival rate by Fare quantiles
   - 2D heatmap: Survival rate by (Pclass × Sex)

**Chart.js Best Practices:**
- Enable responsive: true
- Add tooltips with formatted values
- Use accessible color schemes
- Add download chart as image button

#### Phase 5: Key Insights & Export
**Automated Insights Generation:**
Display a "Key Findings" section with:
- Feature with highest correlation to survival
- Category with highest survival rate
- % of missing data overall
- Suggested feature engineering ideas

**Export Options:**
1. **Merged CSV:** Download combined dataset with 'DataSource' column
2. **JSON Summary:** Export all statistics in structured JSON
3. **PDF Report:** (Optional) Generate visual report

### 4. UI/UX Requirements

**Layout Structure:**
```
┌─────────────────────────────────────┐
│  Header: Titanic EDA Dashboard      │
├─────────────────────────────────────┤
│  1. Data Upload Section             │
│     [Choose train.csv] [Choose test.csv] [Load Data]
├─────────────────────────────────────┤
│  2. Data Overview                   │
│     - Merge Summary Table           │
│     - Dataset Preview (first 10 rows)
├─────────────────────────────────────┤
│  3. Data Quality                    │
│     - Missing Values Chart          │
│     - Outliers Summary              │
├─────────────────────────────────────┤
│  4. Statistical Summary             │
│     - Descriptive Stats Tables      │
│     - Survival Comparison           │
├─────────────────────────────────────┤
│  5. Visualizations                  │
│     [6-8 interactive charts]        │
├─────────────────────────────────────┤
│  6. Key Insights                    │
│     - Automated findings            │
├─────────────────────────────────────┤
│  7. Export Options                  │
│     [Download CSV] [Download JSON]  │
└─────────────────────────────────────┘
```

**Styling Guidelines:**
- Use CSS Grid or Flexbox for responsive layout
- Mobile-first design (breakpoint at 768px)
- Color scheme: Professional (e.g., blue/gray palette)
- Loading spinners during processing
- Error messages in toast notifications

### 5. Error Handling & Edge Cases
**Robust Error Handling:**
```javascript
try {
  // CSV parsing
} catch (error) {
  showError('Failed to parse CSV. Please check file format.');
  console.error('Parse error:', error);
}

// Handle edge cases:
// - Empty files
// - Files with wrong structure
// - All values missing in a column
// - No survivors in train data (edge case)
```

**User Feedback:**
- Progress bar during data loading
- Success messages for each step
- Clear error messages with suggestions

### 6. Code Quality Standards
**Documentation:**
- JSDoc comments for all functions
- Inline comments for complex logic
- README with deployment steps

**Maintainability:**
- Modular functions (single responsibility)
- Configuration object for easy customization
- Constants at top of file
- Consistent naming conventions (camelCase)

**Performance:**
- Debounce user interactions
- Lazy load charts (render on scroll)
- Limit preview table to 100 rows

**TensorFlow.js Best Practices:**
- **Memory Management is CRITICAL:**
  ```javascript
  // Always dispose tensors after use to prevent memory leaks
  const tensor = tf.tensor1d([1, 2, 3]);
  const result = tensor.mean();
  console.log(result.dataSync()[0]);

  // Dispose tensors when done
  tensor.dispose();
  result.dispose();

  // Or use tf.tidy() for automatic cleanup
  const mean = tf.tidy(() => {
    const tensor = tf.tensor1d([1, 2, 3]);
    return tensor.mean();
  });
  // Only 'mean' survives, 'tensor' is auto-disposed
  const meanValue = mean.dataSync()[0];
  mean.dispose();
  ```
- Use `tf.tidy()` to automatically clean up intermediate tensors
- Use `dataSync()` or `await data()` to extract values from tensors
- Prefer vectorized operations over loops for better performance
- Use `tf.memory()` to debug memory leaks during development

### 7. GitHub Pages Deployment Instructions
**Include in README:**
```markdown
## Deployment Steps
1. Create new GitHub repository (public)
2. Upload `index.html`, `app.js`, `README.md`
3. Go to Settings → Pages
4. Select Source: Deploy from branch `main`, folder `/root`
5. Save and wait 2-3 minutes
6. Access at: https://<username>.github.io/<repo-name>/

## Testing Locally
Simply open `index.html` in browser (no server needed)

## Dataset Sources
- Train: https://www.kaggle.com/c/titanic/download/train.csv
- Test: https://www.kaggle.com/c/titanic/download/test.csv
```

## Expected Deliverables
1. ✅ Fully functional single-page application
2. ✅ Clean, commented code following best practices
3. ✅ **TensorFlow.js used for all statistical operations and data transformations**
4. ✅ Responsive design (mobile + desktop)
5. ✅ Minimum 8 visualizations
6. ✅ Automated insights generation
7. ✅ Export functionality
8. ✅ Deployment-ready (no build step)
9. ✅ Proper tensor memory management (no memory leaks)

## Success Criteria
- [ ] App loads and merges data without errors
- [ ] All visualizations render correctly
- [ ] Statistics are mathematically accurate (verified against pandas/numpy results)
- [ ] **TensorFlow.js tensors are properly disposed (check tf.memory().numTensors after operations)**
- [ ] All statistical calculations use tensor operations (not manual JS loops)
- [ ] Insights section provides actionable findings
- [ ] Responsive on mobile devices
- [ ] No console errors or memory leaks
- [ ] Deployed successfully on GitHub Pages

---

**🎯 Key Improvement over Original Prompt:**
1. **TensorFlow.js integration:** Use ML library for efficient data analysis operations (not just prediction)
2. **Statistical rigor:** Added grouped analysis, correlation, outliers, all computed with tensors
3. **Better UX:** Loading states, error handling, progressive disclosure
4. **Production-ready:** Modular code, documentation, testing checklist, proper memory management
5. **Extensibility:** Clear instructions for adapting to other datasets
6. **Professional output:** Automated insights, export options, polished UI
7. **Performance:** Vectorized operations using TensorFlow.js instead of manual loops

