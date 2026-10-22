# Hotel cancellation project: evaluation /20 and improvement plan

## Context

This report checks the project against the course's technical task and the three lecture decks, gives a score out of 20, and lists steps to improve it.

**Second review.** It covers `main` at `4bee901` ("chore: tuned models"). That means the notebook `notebook/project.ipynb` (102 cells), its text pair `notebook/project.py`, the modules in `src/hotel_cancellation/` (now including `plot_style.py`), the README and requirements. The first review covered `e3a4895` and scored 14/20. Section 7 lists what changed since then.

Reference documents:
- The course page "M1-HoML (2026-27)": deliverables, report structure, grading grid and dataset recommendations.
- The "ML Projects Checklists" deck (7 steps, checklists 1–6).
- The "Challenges" deck (data issues, over- and underfitting, validation).
- The "Landscape" deck (model families, metrics, parameters vs hyperparameters).

The slides PDF is not in the repo, and the oral grade is individual, so neither is scored here.

---

## 1. Score (course grid: 4 criteria × /5)

| Criterion | First review | Now, as saved | After a full clean run | After all steps below |
|---|---|---|---|---|
| Preprocessing (exploration, cleaning) | 3.5 | 4.0 | 4.0 | 4.5 |
| Model | 3.5 | 3.0 | 3.5 | 4.5 |
| Code readability | 3.0 | 3.0 | 4.0 | 4.5 |
| Performance evaluation and analysis | 4.0 | 1.5 | 4.5 | 5.0 |
| **Total /20** | **14.0** | **11.5** | **16.0** | **18.5** |

**The saved notebook is only half executed.** Code cells 1–28 have outputs. Every cell from the random-forest search (notebook cell 62) to the end has none. That hides the tuned results, the whole evaluation section, the test score and the plots that the conclusion refers to. A grader opening the file today sees no results.

The "after a full clean run" column assumes the remaining cells run without errors. That has not been verified. If a run fails with `AttributeError: ... has no attribute 'booking_groups'`, the kernel holds an old copy of the module. Restart the kernel and run all cells in one session.

---

## 2. Course requirements check

| Requirement | Status | Evidence |
|---|---|---|
| Tabular task, non-trivial target, >10 features, 10k–1M rows, explicit features | ✅ | 119,390 bookings; 31 inputs, 841 after one-hot; outcome columns removed |
| One notebook with code and a Markdown report | ⚠️ | The structure is complete, but half of the code cells have no output |
| Report structure: Intro, Preprocessing, Training, Evaluation, Conclusion | ✅ | Sections 0–4, with a dedicated `## 4. Conclusion` |
| All member names in the notebook and the slides | ✅ notebook / ❓ slides | Title cell: Adam Nolan, Oleksandr Hakavyi, Mykhailo Mas |
| Data not included, with a link to it in the notebook | ✅ | Kaggle link and DOI of the dataset paper in the title cell; `data/` is gitignored |
| External modules included | ✅ | `src/hotel_cancellation/`, including the new `plot_style.py`, must ship with the notebook; the dead report links are gone |
| Notebook under 10 MB | ✅ | 0.66 MB now; the size will grow once all plots are rendered |
| State-of-the-art summary, plus arguments for the chosen solution | ⚠️ | A "Related work and our contribution" table exists, but it does not compare numbers with this project's results |
| Slides PDF in English for 10 minutes, with every member speaking | ❓ | Not in the repo |

---

## 3. Assessment per criterion

### 3.1 Preprocessing (exploration, cleaning): 4 / 5 (was 3.5)

**Strengths**
- The outcome columns are removed. One grouped 60/20/20 split keeps duplicate records together, and asserts check that the parts don't overlap.
- A new data-quality table on train shows one negative `adr`, 112 zero-guest rows, 457 zero-night rows and 34% repeated raw records. The decision to keep them is explained.
- A new availability table documents which fields may be recorded after booking creation and how each is treated. It cites the dataset paper.
- The preprocessing study now feeds the final models. Log lead time and rare-category grouping are search parameters, fitted inside each fold. Results: log lead time adds +0.019 F1 to logistic regression; grouping adds +0.140 F1 to the forest and cuts encoded features by 72.5%.

**Deductions**
- Exploration is still thin compared with Checklist 3. There are no per-feature distributions, no `describe()`, and no correlation matrix (slides 19–20).
- Outliers are flagged but never examined. The `adr` of 5,400 and `adults` up to 55 are not discussed.
- `country` is still a likely top feature (PRT bookings cancel 56.6% vs 23.6% elsewhere). The notebook neither tests removing it nor discusses fairness.

### 3.2 Model: 3.5 / 5 once executed (3 as saved; was 3.5)

**Strengths**
- The same three grouped CV folds are used everywhere (`grouped_cv` with `StratifiedGroupKFold`), so no duplicate group crosses a fold. This fixes a subtle leak in the earlier CV.
- Searches select on ROC AUC, the same criterion used to choose the model, and also record F1 and log loss. This fixes the earlier F1-at-0.5 inconsistency.
- A search-space table shows which parameters vary and which are fixed. A new plot compares the forest's train and CV AUC by depth and leaf size, which illustrates bias vs variance directly.
- Boosting now tests `max_features` "all" vs "sqrt" instead of fixing "sqrt".

**Deductions**
- The grids got smaller. Most parameters have two values: C in {0.1, 1}, tree depth in {8, 16}, forest depth in {12, 24}, leaf size in {5, 10}. Best values will often land on an edge, which slide 41 says to push past.
- The forest fixes 100 trees and `max_features=0.5`, "which performed well in the preceding exploratory search". That search is not in the notebook, so the choice can't be checked.
- Boosting still tests only 4 configurations, all with 60 stages and depth 3.
- There is still no ensemble of the shortlisted models (Checklist 6.2), and no quick naive Bayes, SVM, k-NN or neural-network baseline (slide 32).
- As saved, the forest and boosting search results are not shown.

### 3.3 Code readability: 4 / 5 once executed (3 as saved; was 3)

**Strengths**
- The executed part runs in order (counts 1–28), with no errors.
- `project.py` is in sync with the notebook (102 cells each).
- The typos are gone, and headings follow one hierarchy with numbered subsections.
- Duplicated code moved into modules: `grouped_cv`, `booking_groups`, `search_space_table`, `configuration_label`, `contact_workload`, `calibration_bins`, and shared styling in `plot_style.py`.
- Several Markdown sentences are generated from results (cell 37, 87, 89, 96), so their numbers can't drift.
- Requirements are pinned, and the saved model stores its scikit-learn version, feature columns and seed.

**Deductions**
- Half the notebook is unexecuted. This is the biggest single problem in the project.
- The README tells readers to run `scripts/execute_notebook.py` and `python -m unittest discover -s tests`. Neither file exists on disk or in git.
- The README says the report was checked with Python 3.11, but `.venv` runs 3.14. `requirements.txt` pins numpy 2.4.6 while 2.5.3 is installed.
- The toy decision-boundary cell still names its dictionary `models` and re-imports numpy and matplotlib; the next cell redefines `models`.
- Module names still start with digits, which forces `import_module(...)`.

### 3.4 Performance evaluation and analysis: 4.5 / 5 once executed (1.5 as saved; was 4)

**Strengths of the design**
- The model and the reference threshold are chosen on validation before the test set is scored, and the notebook says so.
- Confidence intervals come from a bootstrap over whole duplicate groups, the right unit given the grouped split. The paired AUC comparison uses the same draws for both models.
- A contact-budget table shows recall and unnecessary calls when the hotel contacts 10–50% of bookings. It replaces the earlier contradictory cost story.
- A five-field removal test checks how much the model relies on fields that may be recorded late.
- A time-ordered check trains on early arrivals and tests on 2017, with label-availability dates respected.
- Calibration bins with counts, permutation importance on validation, and recall by segment are all included.
- Claims are worded carefully and match slide 37 on confidence intervals.

**Deductions**
- As saved, none of these results are visible.
- The conclusion contains no numbers. It never names the selected model or states its test AUC, recall and precision. It only points to §3.6.
- The conclusion does not say plainly what worked and what didn't, which the course page asks for.
- The text is hedged so heavily that a reader may miss the recommendation. Keep the caveats, but state the result first.
- The related-work table does not set this project's score next to published results.

---

## 4. Coverage of the lecture checklists

| Checklist | Covered | Partial | Missing |
|---|---|---|---|
| 1 Frame | objective, intended use (ranked contact list), supervised batch framing, protocol, minimum performance (training prior) | related work (no numbers) | explicit assumption list |
| 2 Collect | source and link in notebook, size and type, test set put aside, citation | — | licence and anonymization note |
| 3 Explore | types, missing values, target, class balance, train-only plots, quality flags, availability audit | outliers (flagged only) | distributions, correlations |
| 4 Prepare | pipelines, missing values, feature engineering, scaling, encoding, leakage removal, transforms as options | feature selection (five-field test) | outlier cleaning |
| 5 Select model | four families, grouped CV mean and std, permutation importance, segment errors, shortlist | — | naive Bayes, SVM, k-NN and neural-network baselines |
| 6 Fine-tune | grouped CV search on ROC AUC, transforms as hyperparameters, single test use | grid breadth | ensembles |
| 7 Present | figures exported for slides via `plot_style` | — | slides PDF not in the repo |
| Challenges deck | duplicates, over- and underfitting with the depth/leaf plot, regularization, validation and CV, time shift | representativeness | learning curve |
| Landscape deck | four models with formulas, metrics including ROC AUC and average precision, parameters vs hyperparameters | — | SVM, naive Bayes, k-NN (optional) |

---

## 5. Improvement steps

### P0: before submission

1. **Restart the kernel and run every cell in one session.** Check that every code cell has an execution count, in order, and no error output. The searches take several minutes.
2. **Fix the README.** Either add `scripts/execute_notebook.py` and a `tests/` folder, or remove the lines that mention them. Correct the Python version, and align the numpy pin with the version you actually ran.
3. **Put numbers in the conclusion.** Generate them like cell 87 does: the selected model, test ROC AUC with its interval, and recall and precision at the reference threshold. Then add one sentence each on what worked, what didn't, and what you would do with more time.
4. **Add a "This project" row to the related-work table** (seed rows in §8), so the reader sees where your score sits.
5. **Rename the toy dictionary** to `toy_models` and drop its duplicate imports.
6. **Team task.** Make the slides PDF with all three names. Submit the notebook, `src/hotel_cancellation/` and the slides. Do not include data or models.

### P1: strengthen the analysis (changes results; decide on validation only)

1. **Exploration.** Add `describe()`, a few histograms and a correlation heatmap that includes the target, all on train. Look at the extreme `adr` and `adults` values.
2. **Wider grids where the best value lands on an edge.** Show the forest's `max_features` and tree-count choice inside the notebook rather than citing an unseen search.
3. **Country check.** Add a run without `country` next to the five-field removal, and write a short fairness note on using nationality.
4. **Boosting.** Try `HistGradientBoostingClassifier` with early stopping. It is faster and is usually the strongest tabular baseline.
5. **Ensemble.** Try a soft `VotingClassifier` of the tuned forest, boosting model and logistic regression, compared on validation.
6. **Duplicate-leakage check.** Score the same pipeline on a plain stratified random split to show how much ignoring duplicates inflates results. This supports your originality argument.

### P2: polish

1. Rename the modules to valid identifiers (`preprocessing.py`, `modeling.py`, `evaluation.py`, `plots.py`) and use normal imports.
2. Add the quick default baselines from slide 32: naive Bayes, k-NN on a sample, a linear SVM and an MLP.
3. Plot a learning curve for the final model (Challenges slide 5).
4. Add a licence and anonymization note for the dataset.

---

## 6. Oral presentation (individual, /20)

The talk lasts 10 minutes, uses slides rather than the notebook, and gives every member equal time.

| Block | Minutes | Content |
|---|---|---|
| Problem and related work | 1.5 | ranked contact list, protocol, related-work table with your score |
| Data and preprocessing | 2.5 | leakage removal, duplicate groups, quality flags, availability audit, preprocessing study |
| Models and tuning | 2.5 | four families, grouped CV, ROC AUC search, forest depth/leaf plot |
| Evaluation | 2.5 | test score with group-bootstrap intervals, contact budgets, five-field removal, time-ordered check |
| Conclusion | 1 | recommendation, what worked, what didn't, next steps |

Every member should be ready for these likely questions:
- Why group duplicates, and why bootstrap over groups instead of rows?
- Why select on ROC AUC but report an F1 threshold? Which contact budget would you recommend?
- What happens to the score without the five late fields, and without `country`?
- Why is the time-ordered result weaker than the shared split?
- Why are the grids so small, and why are 100 trees and `max_features=0.5` fixed?
- What does a 95% confidence interval mean (slide 37)?
- Basics: the Gini formula, log loss, how a forest averages probabilities, parameters vs hyperparameters.

---

## 7. Changes since the first review

| First-review issue | Status |
|---|---|
| No team names | ✅ Fixed |
| No data link in the notebook | ✅ Fixed, with dataset DOI |
| No state-of-the-art summary | ⚠️ Related-work table added, without a numeric comparison |
| Conclusion missing required parts | ⚠️ Section 4 added, but no numbers and no explicit "worked / didn't" |
| Dead links to `FULL_REPORT.md` and the archive | ✅ Removed |
| Stale outputs contradicting the text | ⚠️ Fixed where executed, but half the notebook now has no output |
| `project.py` out of sync | ✅ Fixed |
| Typos and heading levels | ✅ Fixed |
| Unpinned requirements | ✅ Pinned (numpy pin differs from the installed version) |
| Preprocessing findings unused | ✅ Now search parameters |
| Search on F1 at 0.5 | ✅ Now ROC AUC |
| Contradictory cost story | ✅ Replaced by contact budgets |
| CV folds ignored duplicate groups | ✅ Grouped folds |
| Boosting fixed at `max_features="sqrt"` | ✅ Now searched |
| Best values on grid edges | ❌ Grids are smaller, so likely still the case |
| No ablation of late fields | ✅ Five-field removal added |
| No time-based check | ✅ Exploratory time-ordered check added |
| Overstated confidence-interval wording | ✅ Fixed |
| Thin exploration (distributions, correlations) | ❌ Still missing |
| No ensemble or extra baselines | ❌ Still missing |
| Digit-prefixed module names | ❌ Unchanged |
| New: README references missing `scripts/` and `tests/` | ❌ New issue |

---

## 8. Related-work seed rows (numbers checked against the abstracts)

| Source | Data and protocol | Reported result |
|---|---|---|
| António, de Almeida & Nunes 2017, *Tourism & Management Studies* 13(2) | four resort hotels | accuracy above 90% |
| António, de Almeida & Nunes 2019, *Data Science Journal* 18(1):32 | automated system deployed in two hotels | accuracy above 84%, precision 82%, AUC 88% |
| Arasadina, Briandoko & Setiawan 2026, *MALCOM* 6(3) | same 119,390 bookings, stratified 70/15/15, leak-free pipeline | tuned random forest: AUC 0.9394, accuracy 0.8749, precision 0.8691, F1 0.8219 |
| This project | same data, duplicate-aware 60/20/20 split, group bootstrap | fill in from §3.6 after the full run |

Before the code changes, the tuned forest reached a test AUC of 0.941 on a row-level protocol. The new grouped protocol and smaller grids may change that, so quote only the fresh §3.6 output.

Links:
- António et al. 2017: https://ciencia.iscte-iul.pt/publications/predicting-hotel-booking-cancellations-to-decrease-uncertainty-and-increase-revenue/37158?lang=en
- António et al. 2019: https://ciencia.iscte-iul.pt/publications/an-automated-machine-learning-based-decision-support-system-to-predict-hotel-booking-cancellations/60634?lang=en
- Arasadina et al. 2026: https://journal.irpi.or.id/index.php/malcom/article/view/2674/1328 (DOI 10.57152/malcom.v6i3.2674). The numbers come from the abstract; check them against the PDF before using them on a slide.
- Dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

---

## 9. Verification before submission

- Every code cell has an execution count, the counts increase without gaps, and there are no error outputs.
- Sections 3.3–3.9 show tables and plots, and the conclusion's generated sentence shows the test numbers.
- `project.py` and the notebook have the same number of cells.
- Every file the README mentions exists.
- The notebook is still under 10 MB after all plots render.
- A fresh clone, set up by following the README in a new virtual environment, runs from top to bottom without errors.

---

## 10. Code cleanup and full run (third review)

### Score now

| Criterion | Second review, as saved | Now |
|---|---|---|
| Preprocessing (exploration, cleaning) | 4.0 | 4.0 |
| Model | 3.0 | 3.5 |
| Code readability | 3.0 | 4.0 |
| Performance evaluation and analysis | 1.5 | 4.5 |
| **Total /20** | **11.5** | **16.0** |

The presentation is not included in this score.

**Why it went up.** The notebook now runs from top to bottom in one kernel: all 48 code cells, in order, with no errors, in about 14 minutes; the file is 1.25 MB. Every table and plot is visible, including the test score. The code is shorter and less repetitive (below).

**Main results of the run**

| Item | Result |
|---|---|
| Selected model | Random forest: 100 trees, depth 24, leaf 5, `max_features` 0.5, rare grouping 50 |
| Validation ROC AUC | 0.946 (boosting 0.934, logistic regression 0.920, tree 0.916) |
| Test ROC AUC | 0.941 (95% group-bootstrap interval 0.933–0.948) |
| Test recall / precision at threshold 0.38 | 0.855 / 0.786 |
| AUC gain over logistic regression on test | 0.010–0.015 (95% interval) |
| Five late fields removed | validation ROC AUC 0.946 → 0.941 |
| Time-ordered check (2017 arrivals) | ROC AUC 0.866 |

**Why it is not higher**
- **Preprocessing:** no feature distributions or correlation matrix; outliers are flagged but not examined.
- **Model:** small grids, and several selected values sit on the edge of their range (forest depth 24 and leaf 5, boosting rate 0.2, logistic C 0.1), which slide 41 says to search past. No ensemble of the tuned models.
- **Code readability:** the README still mentions `scripts/execute_notebook.py` and `tests/`, which do not exist.
- **Evaluation:** the conclusion still states no numbers and does not say plainly what worked and what didn't.

### What the cleanup changed

The file structure is unchanged. Every refactored module function was checked to give exactly the same output as before.

| File | Cleanup |
|---|---|
| `0_preprocessing.py` | Shorter `load_data` and `prepare_features`; `split_data` builds the three parts from one dictionary; one-line splitters; the rare-category columns are a named constant; `build_preprocessor` lists its steps directly |
| `1_modeling.py` | Compact training functions; the random forest uses all CPU cores (`n_jobs=-1`, same forests, much faster searches); simpler search-space table and labels |
| `2_evaluation.py` | Removed three unused functions; accuracy computed once; list comprehensions for thresholds, bootstraps and segments; plot labels set in one call |
| `00_preprocessing_plots.py` | Shared helpers for bar labels and figure styling instead of three copies |
| Notebook | One import cell; no duplicate aliases (`prep_X_train` and similar); one-line training calls; a `tune()` helper replaces four copied search blocks; a `proba()` helper replaces repeated `predict_proba(...)[:, 1]`; compact toy example with one shared colormap; one-line cells merged; shorter temporal check |

`record_preprocessing_results` keeps its two feature parameters, so it can score other feature tables. It was restored after the run with identical behavior.

Earlier experiments are saved in the git refs `refs/backup/full-improvements` and `refs/backup/cleanup-with-renames`. Restore a file with `git checkout <ref> -- <path>`.

### Later cleanup: plumbing moved to `src`, simpler plots

| Moved from the notebook | To |
|---|---|
| Arrival dates, time-ordered split, quality flags | `0_preprocessing.py`: `arrival_dates`, `temporal_split`, `quality_flags` |
| Gradient boosting pipeline, grid search helper, search summary, tree and toy-boundary plots | `1_modeling.py`: `train_gradient_boosting`, `grouped_grid_search`, `search_summary`, `plot_tree_levels`, `plot_decision_boundaries` |
| Probability helper, CV summary | `2_evaluation.py`: `positive_proba`, `summarize_cv` |

Plot styling was reduced to matplotlib defaults: `plot_style.py` is now one settings dictionary and a PNG export, and the custom palette, axis restyling and per-element recolouring are gone. Moved functions were checked against the original inline code, and a full run (48 code cells, in order, no errors) reproduces the same results: test ROC AUC 0.941, recall 0.855, precision 0.786.
