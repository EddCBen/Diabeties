import torch
from src.model_class import Diabeties_classifier

model = Diabeties_classifier()
model.load_state_dict(torch.load("./models/Diabeties.pt"),
                      strict=False)

model.eval()
def classify_medical_data(BloodPressure : int,
                   Glucose : int,
                   BMI : float,
                   age : int,
                   pregnancies : int
                   ):
    r"""
    Classify the Sensory Data into: S7i7, or mrid
        args: Sensory Data 
        return: Class_ 
    """
    input_ = torch.tensor([BloodPressure, Glucose, BMI, age, pregnancies])
    class_ = model(input_.reshape(1,-1).float())
    output = ["S7i7", "Mrid"][int(class_.squeeze().item())] 
    return output

if __name__ == "__main__":
    # Test, The model is crap, because the data is crap
    # Diabeties.pt -> The Data was manually agumented
    # Diabeties__1.pt -> Original Dataset (Better results)

    print(classify_medical_data(BloodPressure=120,
                                Glucose=148,
                                BMI=33.6,
                                age=50,
                                pregnancies=6))
