import numpy as np


def sigmoid(input_value):
    return 1 / (1 + np.exp(-input_value))


def sigmoidDerivative(output_value):
    return output_value * (1 - output_value)


with open("mnist_train.csv", "r") as trainingFile:
    trainingData = trainingFile.readlines()

with open("mnist_test.csv", "r") as testFile:
    testData = testFile.readlines()


number_inputs = 784
hiddenNodes = 128
number_outputs = 10

learningRate = float(input("Enter learning rate: "))
epochs = int(input("Enter number of epochs: "))

np.random.seed(1)


input_hidden_weights = np.random.randn(hiddenNodes, number_inputs) * np.sqrt(1 / number_inputs)
hiddenBias = np.zeros(hiddenNodes)

hidden_output_weights = np.random.randn(number_outputs, hiddenNodes) * np.sqrt(1 / hiddenNodes)
outputBias = np.zeros(number_outputs)


for epoch_number in range(epochs):

    np.random.shuffle(trainingData)

    for training_row in trainingData:

        row_values = training_row.strip().split(",")

        correctDigit = int(row_values[0])

        inputValues = np.array(row_values[1:], dtype=float)
        inputValues = (inputValues / 255.0 * 0.99) + 0.01

        targetOutput = np.zeros(number_outputs) + 0.01
        targetOutput[correctDigit] = 0.99

        hidden_input_values = np.dot(
            inputValues,
            np.array(input_hidden_weights).T
        ) + hiddenBias

        hiddenValues = sigmoid(hidden_input_values)

        output_input_values = np.dot(
            hiddenValues,
            np.array(hidden_output_weights).T
        ) + outputBias

        finalValues = sigmoid(output_input_values)

        outputError = targetOutput - finalValues

        hiddenError = np.dot(
            outputError,
            hidden_output_weights
        )

        outputGradient = (
            outputError
            * sigmoidDerivative(finalValues)
        )

        hiddenGradient = (
            hiddenError
            * sigmoidDerivative(hiddenValues)
        )

        hidden_output_weights += learningRate * np.outer(
            outputGradient,
            hiddenValues
        )

        outputBias += learningRate * outputGradient

        input_hidden_weights += learningRate * np.outer(
            hiddenGradient,
            inputValues
        )

        hiddenBias += learningRate * hiddenGradient

    amountCorrect = 0

    for test_row in testData:

        test_values = test_row.strip().split(",")

        actualDigit = int(test_values[0])

        testInputValues = np.array(test_values[1:], dtype=float)
        testInputValues = (testInputValues / 255.0 * 0.99) + 0.01

        hidden_test_inputs = np.dot(
            testInputValues,
            np.array(input_hidden_weights).T
        ) + hiddenBias

        hidden_test_values = sigmoid(hidden_test_inputs)

        final_test_inputs = np.dot(
            hidden_test_values,
            np.array(hidden_output_weights).T
        ) + outputBias

        testOutputValues = sigmoid(final_test_inputs)

        predictedDigit = np.argmax(testOutputValues)

        if predictedDigit == actualDigit:
            amountCorrect += 1

    accuracyPercentage = amountCorrect / len(testData) * 100

    print(
        "Epoch",
        epoch_number + 1,
        "-",
        amountCorrect,
        "out of",
        len(testData),
        "- Accuracy:",
        round(accuracyPercentage, 2),
        "%"
    )


print()
print("Finished training.")
print("Final accuracy:", round(accuracyPercentage, 2), "%")
