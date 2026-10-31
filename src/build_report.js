const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak,
  LevelFormat, Header, Footer, PageNumber
} = require("docx");
const fs = require("fs");

const ROOT = path.dirname(__dirname);
const FIG = path.join(ROOT, "outputs", "figures");
const OUT = path.join(ROOT, "outputs");
const REPORTS_DIR = path.join(ROOT, "reports");
fs.mkdirSync(REPORTS_DIR, { recursive: true });

const FONT = "Calibri";
const HEAD_FONT = "Cambria";
const BODY_SIZE = 23;
const SMALL_SIZE = 20;

const ACCENT = "1F3B57";
const ACCENT_MID = "3D6D99";
const ACCENT_LIGHT = "E8EFF6";
const GOLD = "B8902E";
const PAGE_BG = "FBFCFE";
const ROW_TINT = "EEF3F9";
const BORDER_SOFT = "C7D2DE";

const sizeOf = (p) => {
  const buf = fs.readFileSync(p);
  const width = buf.readUInt32BE(16);
  const height = buf.readUInt32BE(20);
  return { width, height };
};

function figure(p, maxWidthIn = 6.0, caption) {
  const { width, height } = sizeOf(p);
  const ratio = height / width;
  const wIn = maxWidthIn;
  const hIn = wIn * ratio;
  const children = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: caption ? 60 : 200 },
      children: [
        new ImageRun({
          type: "png",
          data: fs.readFileSync(p),
          transformation: { width: Math.round(wIn * 96), height: Math.round(hIn * 96) },
        }),
      ],
    }),
  ];
  if (caption) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [new TextRun({ text: caption, italics: true, size: SMALL_SIZE, font: FONT })],
    }));
  }
  return children;
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 380, after: 180 },
    border: { left: { style: BorderStyle.SINGLE, size: 22, color: ACCENT, space: 10 } },
    children: [new TextRun({ text, bold: true, size: 28, font: HEAD_FONT, color: ACCENT })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, italics: true, size: BODY_SIZE, font: FONT, color: ACCENT_MID })],
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 200, line: 288 },
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, size: BODY_SIZE, font: FONT, color: "222222", ...opts })],
  });
}
function note(text) {
  return new Paragraph({
    spacing: { before: 60, after: 220, line: 264 },
    indent: { left: 260 },
    border: { left: { style: BorderStyle.SINGLE, size: 8, color: GOLD, space: 8 } },
    children: [new TextRun({ text, size: SMALL_SIZE, font: FONT, italics: true, color: "5B5B5B" })],
  });
}
function bulletPara(text) {
  return new Paragraph({
    numbering: { reference: "bullet-list", level: 0 },
    spacing: { after: 130, line: 288 },
    children: [new TextRun({ text, size: BODY_SIZE, font: FONT, color: "222222" })],
  });
}

function simpleTable(headers, rows, colWidthsTwips) {
  const total = colWidthsTwips.reduce((a, b) => a + b, 0);
  const borders = {
    top: { style: BorderStyle.SINGLE, size: 3, color: BORDER_SOFT },
    bottom: { style: BorderStyle.SINGLE, size: 3, color: BORDER_SOFT },
    left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  };
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((hd, i) => new TableCell({
      width: { size: colWidthsTwips[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: ACCENT_LIGHT },
      margins: { top: 90, bottom: 90, left: 120, right: 120 },
      borders: { ...borders, bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT } },
      children: [new Paragraph({ children: [new TextRun({ text: hd, bold: true, color: ACCENT, size: SMALL_SIZE, font: FONT })] })],
    })),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((cell, i) => new TableCell({
      width: { size: colWidthsTwips[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: ri % 2 === 0 ? ROW_TINT : "FFFFFF" },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      borders,
      children: [new Paragraph({ children: [new TextRun({ text: String(cell), size: SMALL_SIZE, font: FONT, color: "222222" })] })],
    })),
  }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: colWidthsTwips, rows: [headerRow, ...bodyRows] });
}

function spacer(h = 160) {
  return new Paragraph({ text: "", spacing: { after: h } });
}

// --- Load results ------------------------------------------------------------
const testMetrics = JSON.parse(fs.readFileSync(`${OUT}/test_metrics.json`));
const robustSummary = JSON.parse(fs.readFileSync(`${OUT}/robustness_summary.json`));
const temporalSummary = JSON.parse(fs.readFileSync(`${OUT}/temporal_validation_summary.json`));
const thresholdSummary = JSON.parse(fs.readFileSync(`${OUT}/threshold_analysis_summary.json`));

const bestModel = testMetrics.best_model;
const rocAucPct = (testMetrics.roc_auc * 100).toFixed(1);
const accPct = (testMetrics.accuracy * 100).toFixed(1);
const precPct = (testMetrics.precision * 100).toFixed(2);
const recallPct = (testMetrics.recall * 100).toFixed(1);
const f1Pct = (testMetrics.f1 * 100).toFixed(2);

const ci = testMetrics.bootstrap_ci_95;
const ciStr = (m, decimals = 1) => {
  const lo = (ci[m]["ci_lower_2.5%"] * 100).toFixed(decimals);
  const hi = (ci[m]["ci_upper_97.5%"] * 100).toFixed(decimals);
  return `${lo}% - ${hi}%`;
};

const aucDropPts = (robustSummary.auc_drop_at_20pct_noise * 100).toFixed(2);
const aucDropDescriptor = robustSummary.auc_drop_at_20pct_noise >= 0 ? "lost" : "actually gained (didn't lose)";
const aucDropAbs = Math.abs(robustSummary.auc_drop_at_20pct_noise * 100).toFixed(2);
const brier = robustSummary.brier_score.toFixed(4);

const audit = JSON.parse(fs.readFileSync(`${OUT}/validation_audit_summary.json`));
const pc = (x, d = 1) => (x * 100).toFixed(d) + "%";
const A = audit.A_duplicates, AT = A.random_split_test_by_twin_status;
const B = audit.B_temporal, C = audit.C_threshold, D = audit.D_single_feature_auc_all_data_no_fitting;
const thr05 = C.test_confusion_by_threshold[0], thrLow = C.test_confusion_by_threshold[1];
const ppv = C.ppv_at_other_prevalences_using_committed_threshold;
const rangeStr = (a) => a[0].toFixed(3) + "-" + a[1].toFixed(3);
const t70 = B.single_cut_by_clock_and_fraction;
const allCutAucs = Object.values(t70).flatMap(o => Object.values(o).map(v => v.auc));
const tt = B.threshold_procedure_applied_temporally;

const thCsv = fs.readFileSync(`${OUT}/threshold_tuning.csv`, "utf8").trim().split("\n");
const thHeaders = ["Threshold", "Precision", "Recall", "F1", "False Neg.", "False Pos."];
const thRows = thCsv.slice(1).map(line => {
  const [t, prec, rec, f1v, fn, fp] = line.split(",");
  return [Number(t).toFixed(2), (Number(prec) * 100).toFixed(1) + "%", (Number(rec) * 100).toFixed(1) + "%", (Number(f1v) * 100).toFixed(1) + "%", fn, fp];
});
const recThreshold = thresholdSummary.recommended_threshold.toFixed(3);
const recTestPrecision = (thresholdSummary.recommended_threshold_test_precision * 100).toFixed(1);
const recTestRecall = (thresholdSummary.recommended_threshold_test_recall * 100).toFixed(1);
const recTestFN = thresholdSummary.recommended_threshold_test_false_negatives;
const recTestFP = thresholdSummary.recommended_threshold_test_false_positives;
const defTestPrecision = (thresholdSummary.default_threshold_test_precision * 100).toFixed(1);
const defTestRecall = (thresholdSummary.default_threshold_test_recall * 100).toFixed(1);

const cvCsv = fs.readFileSync(`${OUT}/cv_model_comparison.csv`, "utf8").trim().split("\n");
const cvHeaders = ["Model", ...cvCsv[0].split(",").slice(1)];
const cvRows = cvCsv.slice(1).map(line => line.split(","));

const bootCsv = fs.readFileSync(`${OUT}/robustness_bootstrap_ci.csv`, "utf8").trim().split("\n");
const bootHeaders = ["Model", ...bootCsv[0].split(",").slice(1).map(h => h.trim())];
const bootRows = bootCsv.slice(1).map(line => line.split(",").map((v, i) => i === 0 ? v : Number(v).toFixed(4)));

const earlyN = temporalSummary.early_n;
const lateN = temporalSummary.late_n;
const randSplit = temporalSummary.random_split_test;
const tempSplit = temporalSummary.temporal_holdout_test;
const aucGap = temporalSummary.auc_gap;
const aucGapAbs = Math.abs(aucGap * 100).toFixed(2);
const aucGapDirection = aucGap >= 0 ? "lower" : "higher";
const temporalTableHeaders = ["Metric", "Random-Split Test", "Temporal Holdout"];
const temporalTableRows = [
  ["Accuracy", (randSplit.accuracy * 100).toFixed(1) + "%", (tempSplit.accuracy * 100).toFixed(1) + "%"],
  ["Precision", (randSplit.precision * 100).toFixed(1) + "%", (tempSplit.precision * 100).toFixed(1) + "%"],
  ["Recall", (randSplit.recall * 100).toFixed(1) + "%", (tempSplit.recall * 100).toFixed(1) + "%"],
  ["F1-score", (randSplit.f1 * 100).toFixed(1) + "%", (tempSplit.f1 * 100).toFixed(1) + "%"],
  ["ROC-AUC", (randSplit.roc_auc * 100).toFixed(1) + "%", (tempSplit.roc_auc * 100).toFixed(1) + "%"],
];
const earlyRatePct = (temporalSummary.early_malignant_rate * 100).toFixed(1);
const lateRatePct = (temporalSummary.late_malignant_rate * 100).toFixed(1);

const doc = new Document({
  creator: "Prachi Agarwal",
  title: "Breast Cancer Classification Report",
  background: { color: PAGE_BG },
  numbering: {
    config: [{
      reference: "bullet-list",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "-", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 400, hanging: 260 } } } }],
    }],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [
              new TextRun({ text: "Breast Cancer Classification Project", size: 16, font: FONT, color: "888888" }),
            ],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", size: 17, font: FONT, color: "999999" }),
              new TextRun({ children: [PageNumber.CURRENT], size: 17, font: FONT, color: "999999" }),
            ],
          })],
        }),
      },
      children: [
        spacer(1300),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 200, after: 200 },
          border: {
            top: { style: BorderStyle.SINGLE, size: 6, color: ACCENT },
            bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT },
          },
          children: [new TextRun({ text: "Beyond Accuracy: An Explainable and Robust", bold: true, size: 32, font: HEAD_FONT, color: ACCENT })] }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 200 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT } },
          children: [new TextRun({ text: "Machine Learning Framework for Breast Cancer Classification", bold: true, size: 32, font: HEAD_FONT, color: ACCENT })] }),
        spacer(600),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
          children: [new TextRun({ text: "Prachi Agarwal", size: BODY_SIZE, font: FONT })] }),
        spacer(600),
        new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Dataset: UCI Breast Cancer Wisconsin (Original), Wolberg et al.", italics: true, size: SMALL_SIZE, font: FONT, color: "444444" })] }),

        new Paragraph({ children: [new PageBreak()] }),

        h1("Abstract"),
        p("This project builds a machine learning pipeline for classifying breast tumors as benign or malignant using the Breast Cancer Wisconsin (Original) dataset - 699 raw patient records, 645 after removing repeat-biopsy duplicates, based on manual cytological grading of fine needle aspirate samples with real missing values. Instead of just training a model and reporting one accuracy number, this project compares several classifiers under cross-validation, explains the winning model's predictions with SHAP, stress-tests it against noisy or missing inputs, tunes its decision threshold without letting the test set influence that choice, and checks how it does on a chronologically later, unseen slice of patients as a chronological holdout and distribution-shift check (explicitly not a substitute for external validation on independent data - see Section 9). The best model was " + bestModel + ", which reached " + rocAucPct + "% ROC-AUC and " + accPct + "% accuracy (95% CI " + ciStr("accuracy") + ") on the held-out test set, and scored similarly on a chronologically later cohort from the same institution (a shift check, not external validation). These near-ceiling scores say as much about how separable this dataset is as about the model - an unfitted sum of the nine pathologist grades already reaches " + pc(D._sum_of_all_nine_features_no_model) + " AUC - so they are not evidence of clinical readiness (Sections 4.3 and 9). SHAP showed that bare nuclei appearance and clump thickness mattered most to the model. Every reported test-set number was produced without the test set influencing any modeling decision - imputation, threshold choice, and model selection are all fit on training data only."),

        h1("Contents"),
        p("1. Introduction"),
        p("2. Dataset and Exploratory Data Analysis"),
        p("3. Methods"),
        p("4. Results"),
        p("5. Explainability (SHAP)"),
        p("6. Robustness Testing"),
        p("7. Decision Threshold Tuning"),
        p("8. Chronological Holdout & Distribution-Shift Analysis"),
        p("9. Discussion and Limitations"),
        p("10. Conclusion"),
        p("References"),

        new Paragraph({ children: [new PageBreak()] }),

        h1("1. Introduction"),
        p("Breast cancer is partly diagnosed by taking a cell sample from a tumor and grading characteristics of the cell nuclei under a microscope. Machine learning models can use these gradings to help classify a tumor as benign or malignant. Throwing a classifier at this and reporting a single accuracy number skips over things that matter in a healthcare setting: can we explain why the model made a given call, does it still hold up when input data is messy or missing, and - just as important - was the evaluation itself set up so that the test set couldn't quietly leak into any decision made along the way."),
        p("This version uses the older, less commonly used Wisconsin 'Original' dataset rather than the more common Wisconsin Diagnostic (WDBC) version, because it is messier in realistic ways - missing values, duplicate patient entries - and because its sample-accession id gives an approximate chronological ordering, which enables a chronological train/test split - a distribution-shift check, not external validation - that is still closer in spirit to real-world deployment than a random split (Section 8)."),
        p("What this project does: clean a genuinely imperfect dataset without letting cleanup leak test-set information into training, compare five classic ML algorithms, pick the best one with cross-validation, explain its predictions with SHAP, run robustness checks (noise, missing features, bootstrap confidence intervals, calibration), tune the decision threshold using only training data, and check generalization on a chronologically separate group of patients."),

        h1("2. Dataset and Exploratory Data Analysis"),
        p("This dataset was collected by Dr. William H. Wolberg at the University of Wisconsin Hospitals and donated to the UCI Machine Learning Repository. It has 699 records, each scored by a physician on nine cytological characteristics, integers from 1 to 10: clump thickness, uniformity of cell size, uniformity of cell shape, marginal adhesion, single epithelial cell size, bare nuclei, bland chromatin, normal nucleoli, and mitoses. Each one is labeled benign or malignant."),
        p("54 records shared a sample id with an earlier record - repeat biopsies from the same patient submitted at different times. The raw file's row order is only an approximate proxy for chronology: it disagrees with strict id order at 19 points. So cleaning first sorts every record by id (breaking ties among identical ids by original file position), then keeps only the first record per id, leaving 645 unique patients in a properly id-sorted order that the temporal split in Section 8 depends on. Sixteen records also had a missing 'bare nuclei' value (marked '?' in the raw file); those are left as missing at this stage and imputed later using only the training data available at model-fitting time (Section 3) - filling them in here, before any train/test split existed, would have let information about test patients leak into the value used to fill training patients, and vice versa. After cleaning, the class balance was 413 benign (64.0%) and 232 malignant (36.0%)."),
        ...figure(`${FIG}/class_balance.png`, 3.8, "Figure 1. Class distribution after removing repeat-biopsy duplicates."),
        p("Basically every one of the nine features correlates fairly strongly with malignancy, which makes sense, since pathologists designed these specific measures to distinguish benign from malignant cytology in the first place. Uniformity of cell size, bare nuclei, and uniformity of cell shape came out as the three most correlated with the malignant label. (This correlation view and the distribution plot below use a median fill purely for plotting; that filled copy is never used for modeling.)"),
        ...figure(`${FIG}/correlation_heatmap.png`, 5.8, "Figure 2. Correlation heatmap across all nine cytology features and the malignancy label."),
        ...figure(`${FIG}/feature_distributions.png`, 6.0, "Figure 3. Distributions of the six most discriminative features, split by diagnosis. Malignant cases are shifted toward higher, more abnormal-looking scores across the board."),

        h1("3. Methods"),
        p("After cleaning, the data was split 80/20 into training and test sets, stratified by class. Five classifiers were trained, each inside a single scikit-learn Pipeline with two preprocessing steps ahead of the classifier: a median imputer for the missing bare-nuclei values, then a StandardScaler. Logistic Regression, an SVM with an RBF kernel, Random Forest, Gradient Boosting, and K-Nearest Neighbors - the standard lineup for a tabular classification problem like this."),
        p("Putting the imputer inside the pipeline (rather than running it once on the whole dataset beforehand) means the median used to fill missing values is refit on whichever training data is available at each point: separately for each of the 10 cross-validation folds during model comparison, and then once on the full training set for the version evaluated on the test set. The test set's bare-nuclei values never influence that median."),
        p("Each model was evaluated with stratified 10-fold cross-validation on the training set only (accuracy, precision, recall, F1, ROC-AUC). Whichever had the best mean CV ROC-AUC was refit on the full training set and evaluated exactly once on the held-out test set. On top of that baseline: SHAP explainability, a set of robustness stress tests (noise, feature dropout, bootstrap confidence intervals, calibration, a learning curve), a decision-threshold analysis tuned entirely on training-set cross-validated predictions (never on the test set - see Section 7), and a temporal holdout that retrains the model on only the chronologically earlier ~70% of patients and tests it on the later ~30% it has never seen (Section 8)."),

        h1("4. Results"),
        h2("4.1 Cross-Validation Comparison"),
        p("Table 1 has the 10-fold CV results for all five models on the training set (imputer and scaler refit inside each fold)."),
        simpleTable(cvHeaders, cvRows, [2300, 1490, 1490, 1490, 1490, 1490]),
        spacer(200),
        ...figure(`${FIG}/cv_model_comparison.png`, 6.0, "Figure 4. Mean cross-validated metrics per model (10-fold stratified CV, training set only)."),
        p(bestModel + " came out on top by mean CV ROC-AUC. All five models scored well above 98% CV ROC-AUC - these cytology features are strong predictors of malignancy by design - so the choice between models came down more to stability and interpretability than any large performance gap."),
        h2("4.2 Held-Out Test Performance"),
        p("The chosen model was refit on the full training set and evaluated once on the untouched 20% test split. Table 2 reports both the point estimate and a 95% bootstrap confidence interval (2,000 resamples of the test set) for every headline metric, so the numbers below come with an explicit sense of how much they might move with a different sample of test patients, not just a single point estimate."),
        simpleTable(
          ["Metric", "Value", "95% CI"],
          [
            ["Accuracy", accPct + "%", ciStr("accuracy")],
            ["Precision (malignant)", precPct + "%", ciStr("precision")],
            ["Recall / Sensitivity (malignant)", recallPct + "%", ciStr("recall")],
            ["F1-score", f1Pct + "%", ciStr("f1")],
            ["ROC-AUC", rocAucPct + "%", ciStr("roc_auc")],
          ],
          [3000, 1800, 1670]
        ),
        spacer(200),
        ...figure(`${FIG}/confusion_matrix.png`, 3.4, "Figure 5. Confusion matrix on the held-out test set."),
        ...figure(`${FIG}/roc_curve.png`, 4.0, "Figure 6. ROC curve on the held-out test set."),
        p("Recall on the malignant class is the number to watch most closely here, since a missed malignant case (false negative) is a much worse mistake than a false alarm. At " + recallPct + "% recall (95% CI " + ciStr("recall") + "), the default 0.5 threshold still misses some malignant cases in the test set - Section 7 looks at what moving the threshold, using only training data, can do about that."),

        h2("4.3 Reading the Headline AUC in Context"),
        p("A " + rocAucPct + "% ROC-AUC on 129 test patients should be read as 'this dataset is easy to separate with this model family', not as a forecast of clinical performance. Three points support that reading (numbers from the audit script 07_validation_audit.py):"),
        bulletPara("Near-ceiling without any model. Scoring every patient by simply summing their nine pathologist grades - no fitting at all - gives " + pc(D._sum_of_all_nine_features_no_model) + " AUC on all 645 patients, and the single feature 'uniformity of cell size' alone gives " + pc(D.uniformity_cell_size) + ". The fitted model adds a few tenths of a point over that."),
        bulletPara("Duplicate feature vectors. The 645 patients contain only " + A.distinct_feature_vectors + " distinct grade combinations; " + A.patients_sharing_a_vector_with_another_patient + " patients share their exact grades with another patient (never with a conflicting label). " + AT.test_patients_with_exact_twin_in_train + " of the 129 test patients have an exact twin in training, and all of them are benign and all are classified correctly. On the " + AT.test_patients_novel + " test patients with no twin - which includes all 46 malignant cases - AUC is " + pc(AT.novel.auc, 2) + " (95% CI " + pc(AT.novel.auc_ci95[0], 1) + " to " + pc(AT.novel.auc_ci95[1], 1) + ") and all " + AT.novel["errors_at_0.5"] + " errors at 0.5 occur. So duplicates flatter accuracy and specificity somewhat, but barely move AUC: a duplicate-grouped 10-fold CV gives " + pc(A.oof_auc_group_10fold_all_data, 2) + " versus " + pc(A.oof_auc_random_10fold_all_data, 2) + " for ordinary CV."),
        bulletPara("Small, single-site, manually graded data. 46 malignant test cases means one extra miss moves recall by about two points; the bootstrap intervals above (AUC lower bound " + ciStr("roc_auc").split(" ")[0] + ") are the honest range, and they describe sampling noise only - not a different hospital, grader or population."),

        h1("5. Explainability (SHAP)"),
        p("SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for a given prediction, giving both a global view (what matters most overall) and a local view (why the model called a particular patient malignant or benign). SHAP values here are computed on the model's actual input space (imputed and standardized, since that's what the classifier was fit on), but every plot below displays and colors points using imputed values in their original 1-10 clinical units, not standardized z-scores, since a z-score axis isn't something a clinical reader can sanity-check against their own intuition for these features."),
        h2("5.1 Global Feature Importance"),
        ...figure(`${FIG}/shap_summary_beeswarm.png`, 6.0, "Figure 7. SHAP beeswarm plot. Each dot is one test patient; color shows the feature's value in its original clinical units (red = high/abnormal, blue = low/normal); position on the x-axis shows how much that pushed the prediction toward malignant or benign."),
        ...figure(`${FIG}/shap_summary_bar.png`, 5.8, "Figure 8. Mean |SHAP value| per feature - a simpler global importance ranking."),
        p("Bare nuclei appearance and clump thickness are clearly the two biggest drivers here, which matches how pathologists are trained to read these samples - bare, enlarged nuclei and irregular clumping are classic indicators of malignancy."),
        h2("5.2 Local (Per-Patient) Explanations"),
        p("To see this play out for individual patients, SHAP waterfall plots are shown for one test patient the model got right and one it got wrong, drawn from the same fully-explained test set used for the global plots above."),
        ...figure(`${FIG}/shap_waterfall_correct_case.png`, 5.8, "Figure 9. Local explanation for a correctly classified test patient."),
        ...figure(`${FIG}/shap_waterfall_misclassified_case.png`, 5.8, "Figure 10. Local explanation for a misclassified test patient."),

        h1("6. Robustness Testing"),
        p("A model can look great on one test split and still fall apart in practice, so four extra checks probe how reliable this model actually is."),
        h2("6.1 Gaussian Noise Injection"),
        p("Each feature in the test set was perturbed with random Gaussian noise scaled to a fraction of that feature's standard deviation (a rough stand-in for grading variability between different pathologists), and performance was re-checked at each noise level. To be explicit about what this is not: this is a robustness experiment, not a synthetic-data generation step. The noised copies of the test set exist only in memory for this one check; they are never added to the training set, never treated as additional real patients, and never used to inflate the reported dataset size."),
        ...figure(`${FIG}/robustness_noise_injection.png`, 5.8, "Figure 11. ROC-AUC vs. injected noise level, for all five models."),
        p(bestModel + " " + aucDropDescriptor + " " + aucDropAbs + " ROC-AUC points when noise equal to 20% of each feature's standard deviation was added - so the decision boundary doesn't seem overly sensitive to small grading inconsistencies."),
        h2("6.2 Feature Dropout / Occlusion"),
        p("The top-k most important features (per the SHAP ranking) were masked by replacing them with their training-set mean, to see what happens if some cytology readings are missing or unavailable."),
        ...figure(`${FIG}/robustness_feature_dropout.png`, 5.8, "Figure 12. ROC-AUC as an increasing number of top-ranked features are masked."),
        p("Performance drops off gradually instead of falling off a cliff, suggesting the model spreads its predictive signal across several correlated cytology features rather than leaning on just one or two."),
        h2("6.3 Bootstrap Stability"),
        p("500 bootstrap resamples of the test set give a 95% confidence interval for ROC-AUC per model - a sense of how much the reported number might shift with a different sample of patients."),
        simpleTable(bootHeaders, bootRows, [2500, 1450, 1450, 1830, 1830]),
        spacer(200),
        ...figure(`${FIG}/robustness_bootstrap_boxplot.png`, 5.8, "Figure 13. Bootstrap distribution of test ROC-AUC per model (500 resamples)."),
        h2("6.4 Calibration"),
        p("A well-calibrated model's predicted probabilities should roughly match how often malignancy actually shows up. " + bestModel + " got a Brier score of " + brier + " (0 is perfect, lower is better)."),
        ...figure(`${FIG}/calibration_curve.png`, 4.2, "Figure 14. Reliability diagram comparing predicted probability to observed malignancy frequency."),
        h2("6.5 Learning Curve"),
        p("A learning curve checks whether the model was starved for training data or already near its ceiling."),
        ...figure(`${FIG}/learning_curve.png`, 5.8, "Figure 15. Training vs. validation ROC-AUC as a function of training set size."),
        p("The training and validation curves converge at a high score with only a small gap, indicating the model isn't overfitting much, and more data would likely not move the needle much further for this dataset."),

        h1("7. Decision Threshold Tuning"),
        p("By default, scikit-learn calls anything with predicted probability above 0.5 malignant, which treats a missed malignancy (false negative) and an unnecessary follow-up test (false positive) as equally costly - not a good fit for a screening context. Choosing the threshold by looking at test-set predictions would be a form of leakage: it lets the test set influence a modeling decision, so the resulting 'test' performance would no longer estimate performance on unseen data."),
        p("This version tunes the threshold entirely on out-of-fold (OOF) predicted probabilities from 10-fold cross-validation on the training set - every training patient gets a probability from a model that never saw that patient during fitting, so it is still an honest unseen-data estimate, but the test set is never involved in the search. Table 3 and Figure 16 show the precision/recall tradeoff computed on these training-set OOF predictions."),
        ...figure(`${FIG}/precision_recall_curve.png`, 4.2, "Figure 16. Precision-recall tradeoff across thresholds, computed on training-set cross-validated (out-of-fold) predictions - not the test set."),
        simpleTable(thHeaders, thRows, [1250, 1450, 1250, 1150, 1250, 1250]),
        spacer(200),
        p("The highest threshold with zero false negatives on the training OOF predictions is " + recThreshold + ". That threshold - chosen before the test set was ever consulted - was then applied to the test set exactly once: recall rose from " + defTestRecall + "% at the default 0.5 cutoff to " + recTestRecall + "% (" + recTestFN + " false negatives), at a cost of precision falling to " + recTestPrecision + "% (" + recTestFP + " false positives). Choosing the threshold without the test set keeps the test numbers honest, but a result of 0 misses out of 46 malignant cases is still a small-sample result: the exact 95% interval for sensitivity is " + pc(thrLow.sensitivity_ci95_exact[0]) + " to 100%, and it is not a guarantee of 100% recall elsewhere (see 7.1)."),
        h2("7.1 What the Low Threshold Costs"),
        simpleTable(
          ["Test set (129 pts)", "Threshold 0.5", "Threshold " + recThreshold],
          [
            ["Missed malignant (FN)", thr05.fn + " of 46", thrLow.fn + " of 46"],
            ["False alarms (FP)", thr05.fp + " of 83 benign", thrLow.fp + " of 83 benign"],
            ["Sensitivity (exact 95% CI)", pc(thr05.sensitivity) + " (" + pc(thr05.sensitivity_ci95_exact[0]) + "-" + pc(thr05.sensitivity_ci95_exact[1]) + ")", pc(thrLow.sensitivity) + " (" + pc(thrLow.sensitivity_ci95_exact[0]) + "-" + pc(thrLow.sensitivity_ci95_exact[1]) + ")"],
            ["Specificity (exact 95% CI)", pc(thr05.specificity) + " (" + pc(thr05.specificity_ci95_exact[0]) + "-" + pc(thr05.specificity_ci95_exact[1]) + ")", pc(thrLow.specificity) + " (" + pc(thrLow.specificity_ci95_exact[0]) + "-" + pc(thrLow.specificity_ci95_exact[1]) + ")"],
            ["Precision in this test set", pc(thr05.ppv_in_this_test_set), pc(thrLow.ppv_in_this_test_set)],
          ],
          [3000, 2800, 2800]
        ),
        spacer(160),
        p("The trade is " + (thrLow.fp - thr05.fp) + " additional false alarms (" + thr05.fp + " to " + thrLow.fp + ") to avoid " + (thr05.fn - thrLow.fn) + " additional misses (" + thr05.fn + " to " + thrLow.fn + "). Whether that is worth it depends on what a false alarm costs (a repeat biopsy, anxiety) versus a miss, which is a clinical judgment this project cannot make."),
        bulletPara("Precision does not transfer. The " + pc(thrLow.ppv_in_this_test_set, 0) + " precision above reflects this dataset's roughly 36% malignancy rate. Holding the measured specificity fixed, the same threshold would give a precision of about " + pc(ppv.prevalence_10pct, 0) + " at 10% prevalence, " + pc(ppv.prevalence_5pct, 0) + " at 5%, and " + pc(ppv.prevalence_1pct, 0) + " at 1% - i.e. most positives would be false alarms in a screening population."),
        bulletPara("The threshold is set by one patient. 'Highest threshold with zero training false negatives' equals the lowest out-of-fold score given to any malignant training patient, so one hard case decides it. Repeating the identical procedure with 50 different CV shuffles gives thresholds from " + C.threshold_across_50_cv_shuffles.min.toFixed(3) + " to " + C.threshold_across_50_cv_shuffles.max.toFixed(3) + " (median " + C.threshold_across_50_cv_shuffles.median.toFixed(3) + "). At the committed value, " + C.threshold_is_set_by_one_training_patient.train_benign_flagged_at_threshold + " of " + C.threshold_is_set_by_one_training_patient.train_benign_total + " training benign patients are flagged."),
        bulletPara("Zero misses did not replicate on the later cohort. Running the same procedure on the early cohort alone gives a threshold of " + tt.threshold_from_early_oof.toFixed(3) + "; on the later cohort it still misses " + tt["late_at_that_threshold"].fn + " of " + (tt["late_at_that_threshold"].fn + tt["late_at_that_threshold"].tp) + " malignant cases while producing " + tt["late_at_that_threshold"].fp + " false alarms (default 0.5: " + tt["late_at_0.5"].fn + " miss, " + tt["late_at_0.5"].fp + " false alarm). The procedure targets zero misses on training data; it does not promise it on new data."),
        p("The bigger takeaway is that the threshold is a design choice to make on purpose, using only data available at tuning time - not something to leave at 0.5 by default, and not something to reverse-engineer from the evaluation set. These exact threshold values are still specific to this dataset and split, and would need to be re-tuned (via the same train-only procedure) on new data before reuse."),

        h1("8. Chronological Holdout & Distribution-Shift Analysis"),
        p("Random splits of patients from a single source say nothing about how a model behaves on genuinely later patients or under a shift in the underlying population. This section is explicitly labeled a chronological holdout and distribution-shift analysis, not external validation - there is no second dataset from a different hospital here. The sample id (a proxy for submission order) is used to split the data so that 'test' patients genuinely weren't part of the pool the model was tuned on, and so that any shift in the patient population over time shows up rather than being averaged away. Because id order and raw file order disagree at 19 points in this dataset, this split explicitly sorts by id rather than trusting file order (Section 2)."),
        p("The first ~70% of patients by id order (" + earlyN + " patients) formed a development cohort, on which " + bestModel + " was retrained from scratch (imputer refit on this cohort only). The remaining ~30% (" + lateN + " patients) were held out completely and scored exactly once, at the end, as the temporal holdout."),
        p("The malignancy rate differs meaningfully between the two cohorts - about " + earlyRatePct + "% malignant in the early group versus about " + lateRatePct + "% in the later one - indicating the patient population shifted somewhat over the course of the study, something a random train/test split would completely hide since it mixes both cohorts together evenly."),
        simpleTable(temporalTableHeaders, temporalTableRows, [2300, 2400, 2400]),
        spacer(200),
        ...figure(`${FIG}/temporal_vs_random_comparison.png`, 5.8, "Figure 17. Random-split test performance vs. temporal holdout performance, side by side."),
        ...figure(`${FIG}/temporal_confusion_matrix.png`, 3.4, "Figure 18. Confusion matrix on the temporal holdout cohort."),
        p("ROC-AUC on the temporal holdout came out " + aucGapAbs + " points " + aucGapDirection + " than on the random-split test set - no degradation was detected despite a lower malignancy rate in the later cohort (see 8.1 for why this is weaker evidence than it looks). As before: this is a better generalization check than another random split, but both cohorts still come from the same institution, the same data collection process, and the same underlying patient population, just at different points in time. A genuinely independent test - different hospital, different pathologists, different patient demographics - could behave quite differently (Section 9)."),

        h2("8.1 Audit of the Temporal Validation"),
        p("Because this check is the project's main answer to 'does it hold up over time?', it was re-examined (script 07). Findings, in order of importance:"),
        bulletPara("Performance was not worse on the later cohort, so the check has not shown robustness to shift - it simply did not detect a problem. AUC on the later cohort is higher than on the random split, but the later cohort has a lower malignancy rate (" + lateRatePct + "% vs " + pc((thr05.tp + thr05.fn) / (thr05.tp + thr05.fn + thr05.fp + thr05.tn)) + " in the random test set) and a different size (" + lateN + " vs 129 patients), so the two columns are not a controlled comparison. The later cohort's AUC, " + pc(tempSplit.roc_auc, 2) + ", comes from only about 57 malignant cases; its 95% CI is " + pc(t70.id_order.train_70pct.auc_ci95[0], 1) + " to " + pc(t70.id_order.train_70pct.auc_ci95[1], 1) + "."),
        bulletPara("The result does not depend on the cut point or the clock. Across train fractions of 50-80% and both clocks (id order and raw file order), temporal-holdout AUC ranges " + pc(Math.min(...allCutAucs), 2) + " to " + pc(Math.max(...allCutAucs), 2) + ". An expanding-window check with four consecutive 100-patient blocks gives " + B.rolling_origin_id_order.map(r => pc(r.auc, 1)).join(", ") + ". Which ordering is the right clock remains unverified, since id is an accession number rather than a timestamp."),
        bulletPara("Model family was chosen using later patients. The model name comes from CV over the random 80% training split, which contains patients from the later cohort. This is a minor selection leak (hyperparameters are defaults), and it is harmless here: selecting on the early cohort alone also picks " + B.model_selection_on_early_cohort_only.winner + "."),
        bulletPara("Exact-duplicate grade combinations cross the cut. About " + pc(t70.id_order.train_70pct.test_share_with_exact_twin_in_train, 0) + " of later patients have an identical grade vector in the early cohort. On the remaining " + t70.id_order.train_70pct.n_novel + " later patients AUC is " + pc(t70.id_order.train_70pct.auc_on_novel_vectors_only, 2) + ", so the conclusion survives, but the later cohort is not fully 'unseen'."),

        h1("9. Discussion and Limitations"),
        p("Even after these methodology fixes, every number in this report still ultimately comes from one institution's data, collected decades ago, and the corrections made here address leakage and validation-procedure problems, not the underlying representativeness of the dataset. The temporal holdout in Section 8 is a real improvement over a random split, but it is still not the same as testing on an independent dataset from a different hospital or patient population."),
        p("Limitations worth being explicit about:"),
        bulletPara("Chronology is approximate: the id used to order patients in Section 8 is a sample accession number, not a verified timestamp. It disagrees with raw file order at 19 points, and 46 ids appear more than once (repeat biopsies); this analysis sorts by id and keeps the earliest occurrence per id as the best available proxy, but it is still an approximation, not a ground-truth chronological record."),
        bulletPara("Not external validation: both the training data and the chronological holdout come from the University of Wisconsin Hospitals. This section is a distribution-shift check over time within one institution, not a substitute for validation on independently collected data from elsewhere - that distinction is intentional, not a simplification."),
        bulletPara("Manual grading: these nine features were scored by a pathologist on a 1-10 scale rather than measured automatically from an image, introducing inter-observer subjectivity this dataset cannot capture (every score came from the same practice)."),
        bulletPara("Not evidence of clinical readiness: near-ceiling AUC on a small, highly separable, single-site, manually graded dataset (Section 4.3), a threshold whose zero-miss result did not replicate on the later cohort (Section 7.1), and no external validation. The model has not been tested on other hospitals, graders, or screening-level prevalence."),
        bulletPara("Small holdout: the temporal holdout cohort is only " + lateN + " patients, so that performance estimate carries meaningful uncertainty of its own, similar to the bootstrap intervals in Section 6."),
        bulletPara("Noise model: the Gaussian noise in Section 6 is a rough approximation of grading inconsistency between observers, run once in memory as a stress test - it is not synthetic training data, does not capture systematic bias, and does not model an adversary actively trying to fool the model."),
        bulletPara("Threshold values are dataset-specific: even though the threshold in Section 7 was chosen without looking at the test set, the numbers there are still specific to this dataset and split, and the same train-only tuning procedure - not the resulting numeric threshold - is what should be reused on new data."),
        bulletPara("Modest overall scale: 645 unique patients is enough to fit and compare simple models, but confidence intervals throughout this report (Sections 4 and 6) show the honest uncertainty that comes with a dataset this size."),
        bulletPara("This is a class project, not a medical device. None of this should be used to make an actual diagnosis."),

        h1("10. Conclusion"),
        p("Going past a single accuracy number - comparing several models, explaining the winning one's predictions with SHAP, stress-testing it against noise and missing data, tuning the decision threshold using only training data, and checking generalization on a chronologically separate cohort - gives a clearer picture of how a model might behave, provided the evaluation itself doesn't leak. Here, imputation happens inside the modeling pipeline rather than before the split, the chronological split is built on an explicit id sort rather than assumed file order, and the decision threshold is chosen from training-only cross-validated predictions rather than the test set. The final " + bestModel + " model reached " + rocAucPct + "% ROC-AUC (95% CI " + ciStr("roc_auc") + ") on the random-split test set and scored similarly on a temporally later cohort with a lower malignancy rate. Because an unfitted sum of the nine grades already reaches " + pc(D._sum_of_all_nine_features_no_model) + " AUC, this mostly shows that the dataset is highly separable. Predictions trace back to clinically sensible features (bare nuclei appearance and clump thickness). Every patient here still comes from one institution, so true external validation on an independent, multi-site dataset remains the natural next step before treating any of this as more than a class project."),

        new Paragraph({ children: [new PageBreak()] }),

        h1("References"),
        p("Wolberg, W. H., & Mangasarian, O. L. (1990). Multisurface method of pattern separation for medical diagnosis applied to breast cytology. Proceedings of the National Academy of Sciences, 87, 9193-9196.", { size: SMALL_SIZE }),
        p("Mangasarian, O. L., & Wolberg, W. H. (1990). Cancer diagnosis via linear programming. SIAM News, 23(5), 1-18.", { size: SMALL_SIZE }),
        p("UCI Machine Learning Repository. Breast Cancer Wisconsin (Original) Data Set. Donated by Olvi Mangasarian, 1992.", { size: SMALL_SIZE }),
        p("Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NeurIPS).", { size: SMALL_SIZE }),
        p("Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.", { size: SMALL_SIZE }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(`${REPORTS_DIR}/Breast_Cancer_ML_Report.docx`, buf);
  console.log("Report written.");
});
