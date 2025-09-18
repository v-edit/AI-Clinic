import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker with a specific locale
fake = Faker('en_US')

# Function to generate a single patient record
def generate_patient_record():
    medical_conditions = ["Hypertension", "Diabetes", "Asthma", "Chronic Kidney Disease", "Arthritis"]
    medications = ["Aspirin", "Lisinopril", "Metformin", "Albuterol"]
    
    # Generate demographic data
    first_name = fake.first_name()
    last_name = fake.last_name()
    gender = random.choice(['Male', 'Female'])
    age = random.randint(20, 80)
    
    # Generate medical data based on simple rules
    condition = random.choice(medical_conditions)
    
    # Simple rule: if condition is Diabetes, prescribe Metformin
    if condition == "Diabetes":
        medication = "Metformin"
    else:
        medication = random.choice(medications)
        
    admission_date = fake.date_between(start_date="-2y", end_date="today")
    
    return {
        "patient_id": fake.uuid4(),
        "first_name": first_name,
        "last_name": last_name,
        "age": age,
        "gender": gender,
        "medical_condition": condition,
        "medication": medication,
        "date_of_admission": admission_date.isoformat(),
        "notes": f"Patient presented with {condition} and was prescribed {medication}."
    }

# Generate a dataset of 100 patient records
records = [generate_patient_record() for _ in range(100)]
df = pd.DataFrame(records)
print(df.head())