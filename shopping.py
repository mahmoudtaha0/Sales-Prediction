import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():
    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data("shopping.csv")
    # print(evidence)
    # print(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    # Label is revenue
    month_mapping = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    with open(filename) as f:
      reader = csv.reader(f)
      next(reader)

      evidence = []
      label = []

      for row in reader:
          evidence.append([
              int(row[0]),  # Administrative
              float(row[1]),  # Administrative_Duration
              int(row[2]),  # Informational
              float(row[3]),  # Informational_Duration
              int(row[4]),  # ProductRelated
              float(row[5]),  # ProductRelated_Duration
              float(row[6]),  # BounceRates
              float(row[7]),  # ExitRates
              float(row[8]),  # PageValues
              float(row[9]),  # SpecialDay
              month_mapping[row[10]],  # Month
              int(row[11]),  # OperatingSystems
              int(row[12]),  # Browser
              int(row[13]),  # Region
              int(row[14]),  # TrafficType
              1 if row[15] == "Returning_Visitor" else 0,  # VisitorType
              1 if row[16] == "TRUE" else 0  # Weekend
          ])
          label.append(1 if row[17] == "TRUE" else 0)
    return evidence, label
    raise NotImplementedError


def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model
    raise NotImplementedError


def evaluate(labels, predictions):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            true_positive += 1
        elif actual == 0 and predicted == 0:
            true_negative += 1
        elif actual == 1 and predicted == 0:
            false_negative += 1
        else: 
            false_positive += 1

    sensitivity = true_positive/(true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    specificity = true_negative/(true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

    return sensitivity,specificity
    raise NotImplementedError


if __name__ == "__main__":
    main()
