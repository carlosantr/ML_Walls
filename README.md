# ML_Walls
This repository presents the trained ML models of the paper "Machine Learning based evaluation of roof drifts and accelerations of RC wall buildings during ground motions". Artificial Neural Netwroks (ANN) and Random Forest (RF) are available to make predictions of the maximum roof acceleration (PFA_max) and the maximum roof drift ratio (rDR_max) in reinforced concrete wall buildings. The next steps are required to properly use the prediction functions:

0. Download the repository as a zip file and extract the elements on your own PC.
1. Open the code called "Prediction_Function.py" in a Python IDE (Spyder 5.4.3 is recommended)
2. Installation of libraries: You have to install from your console in python the next libraries: Scikit-Learn (write in console: pip install --user scikit-learn==1.2.1), Joblib (write in console: pip install --user joblib==1.4.2), Keras (wite in console: pip install --user keras==3.5.0) and Tensorflow (write in console: pip install --user tensorflow==2.17.0).
3. Restart the kernel (or close and open again the IDE).
4. Once you have the libraries downloaded, you can freely use and call the available functions to make your predictions. Check the description of each function in its own code to avoid errors.
