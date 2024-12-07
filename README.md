# Replication of Kapoor et al. (2020)

---

This is the final project of the Microeconometrics course in the summer semester 2021 by Timo Haupt. 

---

## Project overview

My project is the replication of the paper ["The Price of Forced Attendance"](https://onlinelibrary.wiley.com/doi/10.1002/jae.2781) by S. Kapoor, M. Oosterveen and D. Webbink (Journal of Applied Econometrics, 2020, Volume 36, Issue 2, pp. 209-227).

The Jupyter Notebook of my project can be accessed here:

<a href="https://nbviewer.org/github/tihaup/Project-Microeconometrics/blob/main/Forced_Attendance_Notebook.ipynb" 
   target="_parent">
   <img align="center"
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png"
      width="109" height="20">
</a>

Kapoor et al. (2020) estimate the treatment effect of a forced tutorial policy within an undergraduate Economics program using the Regression Discontinuity Design. Students were assigned to treatment based on their first-year GPA. If a GPA value of 7 hasn't been reached within the first year, tutorials became mandatory for these below-7 students in the second-year of the program. 
Kapoor et al. (2020) find particularly large Local Average Treatment Effects in "voluntary" courses where untreated students themselves could choose whether to attend tutorials. In these courses the estimated effect on second-year grades has been significantly negative. The authors argue that the positive effect of tutorials, giving more structure to students, is outweighed by the negative effect of constraining independent study choices.  


## Reproducibility

To ensure full reproducibility of my project, [GitHub Actions CI](https://docs.github.com/en/actions) has been set up. Certain parts of the baseline results have been only possible to replicate using a robust regression discontinuity function, that hasn't been implemented in python yet. That's why I had to estimate some results in the "R_magic" notebook that can be found in the [cache-folder](https://github.com/OpenSourceEconomics/ose-data-science-course-projeect-tihaup/tree/master/cache). In order to get the Continuous Integration workflow running I had to import these results into the main notebook. In the main notebook it has been explained which results have been imported and what has been done in the "R_magic" notebook.

![Continuous Integration](https://github.com/OpenSourceEconomics/ose-template-course-project/workflows/Continuous%20Integration/badge.svg)


## Structure of notebook

1. Introduction
2. Data
    * Data Description
    * Descriptive Statistics
3. Identification Strategy & Empirical Approach
    * Regression Discontinuity Design
    * Discussion of continuity assumption  
    * Mass Points  
4. Replication of Baseline Results
    * Average Treatment Effect Estimates
    * Heterogeneous Average Treatment Effect Estimates
5. Robustness Checks
    * Comparison Abolition Cohort
    * Discussion of Mechanisms
    * Fake Cutoff Test 
6. Independent Contributions
    * Local Linear Regression Estimates
    * Local Polynomial Regression Estimates
    * Sensitivity Analysis of Bandwidth Choice
6. Conclusion
7. References


## References

* **Calconico, Cattaneo & Farrel (2017).** [rdrobust: Software for regression-discontinuity designs](https://journals.sagepub.com/doi/abs/10.1177/1536867X1701700208). The Stata Journal (17, Number 2, pp. 372–404).


* **Cunningham (2021).** [Causal Inference: The Mixtape](https://mixtape.scunning.com/index.html). Yale University Press.


* **Dobkin, Gil & Marion (2010).** Skipping Class in College and Exam Performance: Evidence from a Regression Discontinuity Classroom Experiment. Economics of Education Review (Volume 29, Issue 4, pp. 566-575).


* **Kapoor, Oosterveen & Webbink (2020).** [The Price of Forced Attendance](https://onlinelibrary.wiley.com/doi/10.1002/jae.2781). Journal of Applied Econometrics (Volume 36, Issue 2, pp. 209-227). 
