# Hospital Readmittance Predictive Model

### [Presentation](https://docs.google.com/presentation/d/10ZXe-AMFxUA1tyd-muZWlMH_uLtWfd_7NdCIsclehWs/edit?usp=sharing)

### Part 1: [Predicting Hospital Readmission Using XGBoost](https://github.com/CamH53/DiabetesDatasetTool/blob/5ee0981282c593ebfc26b6b345a5b11b8cb2621f/CleanSortAndPredict(Cameron))

In this file, I undertook a comprehensive analysis and modeling task on a health sciences dataset, demonstrating my proficiency in data wrangling, visualization, and predictive modeling. I began by loading and cleaning the dataset, addressing missing values, dropping unnecessary columns, and converting categorical age data into numeric values for better analysis. I visualized key relationships using bar plots and pie charts, providing insights into readmission rates for patients on various medications. I then built an XGBoost model to predict hospital readmission, incorporating both numerical and categorical features through preprocessing pipelines. Feature importance analysis revealed the most impactful factors influencing readmission, which I highlighted through a bar plot of the top contributing features.


### Part 2: [Diabetes Data Preprocessing and Feature Engineering](https://github.com/CamH53/DiabetesDatasetTool/blob/5ee0981282c593ebfc26b6b345a5b11b8cb2621f/CleanandWrangle(Both).ipynb)

In this second part of my project, I meticulously tackled the complexities of data preprocessing to enhance the quality and usability of our diabetes dataset. I began by addressing the missingness in crucial variables such as race, weight, and payer code, with a notable focus on resolving the 96.9% missingness in weight. I transformed the age variable from categorical ranges into numerical midpoints, ensuring a more nuanced analysis while acknowledging potential biases. I evaluated the impact of the self-pay option on readmission rates, ultimately determining it had negligible influence, leading to its exclusion from the final model. Furthermore, I refined the dataset by removing irrelevant columns and handling missing values, ensuring a clean and robust dataset for modeling. The result was a streamlined dataset, meticulously prepared and ready for insightful analysis, which I saved and exported for further use.

