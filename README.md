# claim Rank Demo
To debug the demo:
 1. Start with demoAPI.py : Flask environment is set to call the main page (index. html) when the app is run.
    The app gets the pages from the templates folder
 2. DemoAPI calls main procedure in data/main.py (this is the main interface that connects the flask environment with the underlying system
 3. In demoAPI.py , the call to the main interface is done via the statement : sentenceVec = obj.mainproc(text,True,False)
    first param is the text entered by the client, second param indicates whether we need to train the model first, third param
    indicates whether we need to get predictions for the entered text. all of the four debates are used as a training set, there is no testing at this stage.
 4. If training is set to True, the training starts with constructing a dataset from the four debates (in data/debates- method :prepare_train_data_for_demo)
    - then only features that are suitable for the demo are put into a pipeline (in features/feature_sets.py -method: get_demo_pipeline )
    - if you want to change the set of features just change this method call to the method you want in feature_sets.py
    - pipeline is a set of features selected, method that implements the different features are all under "features"
    - to select features you need to know dependencies of the methods that implement the feature extraction
    - two things to look at : the feature name(s) that each method extracts (sent.features['feature_name'] or FEATS ['feat1_name', 'feat2_name', ...])
      and the other thing is what member of the sentence object you will need in these methods, like sent.label, sent.speaker, sent.text
 5. After getting the pipeline the svm is run with the selected set of features to train the model
 6. The trained model is preserved/serialized ("pickled" as per python) via the "joblib" class form "scilearn" API
 7. If the prediction param in the main call is set to True, the prediction method is called, the trained pickled model is loaded and prediction starts
 8. finally, the predictions are sent back to demoAPI, and demoAPI redirects the client to the results.html template carrying the sentence list with predictions filled to be displayed
 Israa