import pickle


def sm_detection(features):
    pickled_model = pickle.load(open('smoke_detection.pkl', 'rb'))
    smoke = str(round(list(pickled_model.predict([features]))[0]))

    return str("smoke detection " + smoke)
test_features=[-4.129,
 56.03,
 314.0,
 400.0,
 13105.0,
 19952.0,
 939.413,
 0.25,
 0.26,
 1.75,
 0.273,
 0.006]
sm_detection(test_features)