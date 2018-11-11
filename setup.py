import os
if not os.path.exists("./Modules"):
  os.system("mkdir Modules")
  os.system("cd Modules")
if not os.path.exists("./DesignMat"):
  os.system("mkdir DesignMat")
  os.system("wget -P ./DesignMat https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/DesignMat/__init__.py -q -o /dev/null")
  os.system("wget -P ./DesignMat https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/DesignMat/gaussian.py -q -o /dev/null")
  os.system("wget -P ./DesignMat https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/DesignMat/polynomial.py -q -o /dev/null")
  os.system("wget -P ./DesignMat https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/DesignMat/sigmoidal.py -q -o /dev/null")
if not os.path.exists("./Regressor"):
  os.system("mkdir Regressor")
  os.system("wget -P ./Regressor https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Regressor/__init__.py -q -o /dev/null")
  os.system("wget -P ./Regressor https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Regressor/bayesian.py -q -o /dev/null")
  os.system("wget -P ./Regressor https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Regressor/empirical.py -q -o /dev/null")
  os.system("wget -P ./Regressor https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Regressor/least_squares.py -q -o /dev/null")
  os.system("wget -P ./Regressor https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Regressor/Regressor.py -q -o /dev/null")
if not os.path.exists("./Classifier"):
  os.system("mkdir Classifier")
  os.system("wget -P ./Classifier https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Classifier/__init__.py -q -o /dev/null")
  os.system("wget -P ./Classifier https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Classifier/logistic.py -q -o /dev/null")
  os.system("wget -P ./Classifier https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Classifier/least_squares.py -q -o /dev/null")
  os.system("wget -P ./Classifier https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Classifier/Classifier.py -q -o /dev/null")