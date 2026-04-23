import pytest
import pickle
import os


def predict_label(text, predictor):
    vectorizer, model = predictor

    X = vectorizer.encode([text])
    return model.predict(X)[0]


@pytest.fixture(scope="session")
def predictor():
    import pickle

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return vectorizer, model

# 1. MINIMUM FUNCTIONALITY TESTS
@pytest.mark.parametrize(
    "text, expected",
    [
        # chiaramente reali
        ("Donald Trump is the new president of United States", 0),
        ("Scientists discovered a new species in the Amazon rainforest", 0),

        # chiaramente fake
        ("Aliens have landed in Rome and met the president", 1),
        ("Drinking bleach cures all diseases instantly", 1),
    ],
)
def test_minimum_functionality(text, expected, predictor):
    pred = predict_label(text, predictor)
    assert pred == expected


# 2. INVARIANCE TESTS
@pytest.mark.parametrize(
    "text1, text2",
    [
        (
            "The president signed a new economic policy",
            "The president has signed a new economic policy",
        ),
        (
            "COVID-19 vaccines are effective",
            "Covid 19 vaccines is a success",
        ),
        (
            "The stock market crashed yesterday",
            "Yesterday the stock market crashed",
        ),
    ],
)
def test_invariance(text1, text2, predictor):
    pred1 = predict_label(text1, predictor)
    pred2 = predict_label(text2, predictor)

    assert pred1 == pred2


# 3. DIRECTIONAL EXPECTATION TESTS

@pytest.mark.parametrize(
    "real_text, fake_text",
    [
        (
            "NASA launched a new satellite into orbit",
            "NASA admits the Earth is flat and covered by a dome",
        ),
        (
            "The vaccine passed all clinical trials",
            "The vaccine contains microchips to control people",
        ),
        (
            "The UK sign fo brexit ufficialy",
            "Secret elites manipulated the economy using mind control",
        ),
    ],
)
def test_directional(real_text, fake_text, predictor):
    pred_real = predict_label(real_text, predictor)
    pred_fake = predict_label(fake_text, predictor)

    assert pred_real == 0
    assert pred_fake == 1


# ======================================================
# 4. ROBUSTNESS (extra ma molto importante)
# Rumore nel testo → stesso output
@pytest.mark.parametrize(
    "clean, noisy",
    [
        (
            "The government passed a new law",
            "THE government  ree passed a yrtet new law!!!",
        ),
        (
            "Scientists discovered water on Mars",
            "Scientists discovered dadwdw water on Mars!!!",
        ),
    ],
)
def test_robustness_noise(clean, noisy, predictor):
    pred_clean = predict_label(clean, predictor)
    pred_noisy = predict_label(noisy, predictor)

    assert pred_clean == pred_noisy