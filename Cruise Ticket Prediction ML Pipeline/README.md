# Cruise Ticket Prediction Machine Learning Pipeline

### a. Objectives
Cruise Company A (CCA) is a prestigious cruise company that aims to provide our customers with the best experiences possible, vying to be the top choice for travellers worldwide. The company is dedicated to constantly improving our service, tailoring offerings to match guests' needs and tastes, hence ensuring an unforgettable cruising experience. As such, CCA create an interaction-rich ecosystem where they cherish customer involvement at every touchpoint of their journey synergizing the offline and online experiences.

To elevate the guest experience and meet evolving demands, CCA regularly undertakes pre-purchase surveys on their website, incentivising future customers with attractive vouchers
and upgrades. The survey requires potential guests to rate their preferences on a range of indicators critical in ensuring a memorable cruise journey - "Onboard Wifi Service", "Embarkation/Disembarkation time convenient", "Ease of Online booking", "Gate location", "Onboard Dining Service", "Online Check-in", "Cabin Comfort", "Onboard Entertainment", "Cabin service", "Baggage handling", "Port Check-in Service", "Onboard Service" as well as "Cleanliness". These preferences provide CCA with comprehensive insights into what potential guests value the most, ensuring that they can meticulously tailor offerings to guest desires. Simultaneously, after concluding each journey, CCA collects post-trip data such as "Cruise Name",
"Cruise Distance", "WiFi", "Dining", "Entertainment" travelled to cross-reference and contextualise guest preferences along with the realities of their chosen itineraries. This information shapes the foundations for the company's data repository further enriching our understanding of guest preferences and patterns. It proves to be invaluable as it empowers the company with insights necessary to formulate efficient and compelling marketing strategies, amplifying their appeal in the cruising market.

Harnessing the collective power of the pre-purchase and post-trip data, we predict the type of tickets potential customers are most likely to purchase. By predicting guests' preferred ticket type, the company aims to customise the experiences and amenities, masterfully aligning them with the guests' comfort and preferences to maximise potential revenue.

### b. Overview and Folder Structure
This repository contains a machine learning pipeline designed to preprocess, train, and evaluate a model on cruise customer data. The pipeline is structured to ensure modularity, scalability, and ease of use.


##### Folder Structure
![Folder structure](https://github.com/askavania/cruise_prediction/blob/main/Cruise%20Ticket%20Prediction%20ML%20Pipeline/Images/image-4.png) 


### c. Execution Instructions
Ensure all dependencies are installed using pip install -r requirements.txt.
Modify any parameters or configurations in config.json as needed.
Execute the pipeline using GitBash with the command: bash run.sh.

### d. Pipeline Flow
The ML pipeline follows these logical steps:

1. Data is fetched from SQLite databases.
2. Preprocessing is applied to clean and transform the data.
3. Features are scaled and encoded.
4. A machine learning model is trained.
5. The model is evaluated using test data.
6. Evaluation metrics are saved to a text file for reference.

### e. Key Findings from EDA
1. Customer Demographics:
- The majority of potential customers come from the 31-45 and 46-60 age groups.
- There's a balanced distribution between genders.
- Most sales come from Direct Sources of Traffic.

2. Customer Preferences:
- Features like "Onboard Wifi Service", "Baggage Handling", "Cabin Comfort", "Onboard Dining Service", and "Cleanliness" are highly valued across all age groups.
- The 40-60 age group has higher expectations across multiple features.
- Females have a higher minimum requirement for Cabin Comfort as compared to Males, while Males have a higher minimum requirement for Cabin Service as compared to Females.

3. Ticket Types and Services:
- Luxury tickets include WiFi, Dining, and Entertainment; Deluxe offers Dining & Entertainment; Standard offers only Dining.
- Luxury ticket holders have higher expectations, especially for features like Online Check-in, Cabin Comfort, and Onboard Entertainment.
- Most customers who purchased luxury tickets travel longer distances.
- Number of Deluxe tickets are significantly lower(class imbalance).

4. Marketing Insights:
- Younger customers (18-30) are more likely to be drawn from Indirect Sources, while older age groups prefer Direct Sources.
- Luxury tickets are predominantly purchased by the 31-45 and 46-60 age groups.

5. Data Quality and Missingness:
- Some customers might have provided inaccurate age data, leading to unrealistic age values. We made them null and will impute them later.
- Missing data patterns in "WiFi" and "Entertainment" are most likely due to the services offered by the different ticket types, while the missingness of the other features could influence ticket types. As such, we will impute missing values with the KNN imputer.

Based on these findings, several preprocessing steps were applied, such as converting mixed date formats and mapping the importance scale.

### f. Feature Processing
| Feature | Processing Applied |
|---------| -------------------|
|Date of Birth | Converted to age, removed day and month|
|Logging | Converted to datetime, used to calculate age |
|Onboard Services |	Mapped to consistent importance scale |
|Gender, Source of Traffic | One-hot encoded|
|Cruise Distance, WiFi, Dining, Entertainment |	Removed as they are post-trip features |
|||

All features were then scaled using RobustScaler, which was chosen as it is less sensitive to outliers. 

### g. Model Choices and Evaluation
As the target variable that we want to predict is a 'Ticket Type', a classification problem, we explore suitable models such as Logistic Regression, RandomForest and XGBoost.  

For a baseline model, we chose to use Logistic Regression, a commonly used model for classification tasks. The f1-score was 0.64.

We then try out 2 other models, Random Forest and XGBoost classifiers.

After hyperparameter tuning, Random Forest gave us a f1 score of 0.82.<br><br> 
![Alt text](https://github.com/askavania/cruise_prediction/blob/main/Cruise%20Ticket%20Prediction%20ML%20Pipeline/Images/image.png)

XGBoost gave us a f1-score of 0.82 as well.<br><br> 
![Alt text](https://github.com/askavania/cruise_prediction/blob/main/Cruise%20Ticket%20Prediction%20ML%20Pipeline/Images/image-1.png)
<br><br>
We decided to go with the RandomForest Classifier model in the end based on the precision score, where they outperform xgboost in the 'Deluxe' ticket type. Reason being, for our use case, the company is trying to predict the ticket type customers will purchase in order to customise the experiences and amenities, which will be costly in time, effort and money. We will want to try to avoid investing in false positives.   


### h. Deployment Considerations
Ensure the model is retrained periodically with fresh data.
Monitor the model's performance in a real-world setting and compare it with the evaluation metrics.

---
## Conclusion

Throughout this project, we've designed and implemented a comprehensive machine learning pipeline tailored to the specific needs of the cruise dataset. By leveraging advanced preprocessing techniques and the power of the RandomForest algorithm, we've aimed to create a model that is both accurate and interpretable.

The pipeline's modular design ensures that it can be easily adapted to new data or modified to incorporate different modeling techniques. The detailed documentation provided in this README, along with the accompanying Jupyter notebook, offers a clear overview of the entire process, from initial data exploration to final model evaluation.

Key findings from our exploratory data analysis informed many of the decisions made throughout the pipeline, ensuring that our approach was data-driven and aligned with the unique characteristics of the dataset. The feature processing summary table offers a quick reference to understand the transformations applied, while the model training and evaluation section provides insights into the performance of our chosen algorithm.

In choosing RandomForest, we leveraged an algorithm known for its versatility, ability to handle high-dimensional data, and its capability to provide feature importance metrics. The ensemble nature of RandomForest, which builds multiple decision trees and aggregates their results, offers a balance between bias and variance, leading to robust performance on diverse datasets.

In conclusion, this project serves as a testament to the power of systematic data analysis and the right choice of machine learning algorithms. We believe that the pipeline developed here can serve as a robust foundation for future enhancements and adaptations.
