from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the logistic regression model pipeline
model = joblib.load('logreg_model.joblib')

# Define the input data model
class CustomerData(BaseModel):
    tenure: int
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    Contract: str
    PaymentMethod: str

# Create FastAPI app
app = FastAPI()

# Define prediction endpoint
@app.post("/predict")
def predict(data: CustomerData):
    # Convert input data to a dictionary and then to a DataFrame
    input_data = {
        'tenure': [data.tenure],
        'InternetService': [data.InternetService],
        'OnlineSecurity': [data.OnlineSecurity],
        'TechSupport': [data.TechSupport],
        'Contract': [data.Contract],
        'PaymentMethod': [data.PaymentMethod]
    }
    
    import pandas as pd
    input_df = pd.DataFrame(input_data)
    
    # Make a prediction
    prediction = model.predict(input_df)
    
    # Return the prediction
    return {"prediction": int(prediction[0])}

# If running directly, use this part to run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
