from pdf_report import generate_health_report

report = generate_health_report(
    disease="Diabetes",
    prediction="Positive — elevated risk detected",
    inputs={
        "Age": 45,
        "Glucose": 160,
        "Blood Pressure": 80,
        "BMI": 29.5,
    },
    patient_name="Test Patient",
    shap_explanation=[
        "Glucose: Example feature contribution.",
        "BMI: Example feature contribution.",
    ],
)

with open("test_health_report.pdf", "wb") as file:
    file.write(report.getvalue())

print("SUCCESS: PDF report generated!")
print("File: test_health_report.pdf")