import pytest
from my_functions.evaluate import calculate_f1score
import csv

@pytest.fixture
def predictions_file(tmpdir):
    # Writes a prediction csv to test

    predictions_file_path = tmpdir.join("predictions.csv")

    with open(predictions_file_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["PassengerId", "Prediction", "True_Label", "split"])        
        writer.writerow(["1-training", "die", "die", "training"])
        writer.writerow(["2-training", "die", "die", "training"])
        writer.writerow(["3-training", "die", "die", "training"])
        writer.writerow(["4-training", "die", "die", "training"])
        writer.writerow(["5-training", "die", "die", "training"])
        writer.writerow(["6-training", "die", "die", "training"])
        writer.writerow(["7-training", "die", "die", "training"])
        writer.writerow(["8-training", "die", "die", "training"])
        writer.writerow(["9-training", "die", "live", "training"])
        writer.writerow(["10-training", "live", "live", "training"])       
        writer.writerow(["79-validation","live","live","validation"])
        writer.writerow(["80-validation","live","live","validation"])
        writer.writerow(["81-validation","die","die","validation"])
        writer.writerow(["82-validation","die","live","validation"])
        writer.writerow(["83-validation","live","live","validation"])
        writer.writerow(["84-validation","die","die","validation"])
        writer.writerow(["85-validation","die","live","validation"])
        writer.writerow(["86-validation","die","live","validation"])
        writer.writerow(["87-validation","die","die","validation"])
        writer.writerow(["88-validation","die","die","validation"])

    yield predictions_file_path



def test_on_csv_with_training_and_validation(predictions_file):
    expected_f1_train, expected_f1_val = 0.89 , 0.69

    predictions_file_path = predictions_file

    actual_f1_train, actual_f1_val = calculate_f1score(predictions_file_path)
    actual_f1_train = round(actual_f1_train, 2)
    actual_f1_val = round(actual_f1_val, 2)

    assert (actual_f1_train, actual_f1_val) == (pytest.approx(expected_f1_train), pytest.approx(expected_f1_val))