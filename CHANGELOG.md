# CHANGELOG

## v0.1.0
All notable changes to this project will be documented in this file.

---

## v0.1.0 - Initial Release

### Added
- Project structure following ML lifecycle
- Data loading and preprocessing pipeline
- Logistic Regression model training
- Model serialization using joblib
- Compressed model storage
- Flask REST API for serving predictions
- /predict endpoint for inference
- Requirements.txt for dependency management
- README documentation with setup and usage instructions

### Data
- Integrated Kaggle Diabetes Prediction Dataset
- Implemented categorical encoding (OneHotEncoder)
- Implemented train/test split (80/20)

### Model
- Logistic Regression classifier
- Accuracy evaluation metric
- Model compression to reduce storage size

### Deployment
- Flask-based serving implementation
- curl command example for predictions
- Local testing instructions

---

## Future Improvements

- Add Docker containerization
- Implement CI/CD pipeline
- Integrate MLflow Model Registry
- Add automated testing
- Deploy to cloud environment
