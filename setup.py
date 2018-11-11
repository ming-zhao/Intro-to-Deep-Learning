import os
if not os.path.exists("./Modules/DesignMat"):
  os.makedirs('./Modules/DesignMat')
  os.system("wget -P ./Modules/DesignMat https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/DesignMat/__init__.py -q -o /dev/null")
  os.system("wget -P ./Modules/DesignMat https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/DesignMat/Gaussian.py -q -o /dev/null")
  os.system("wget -P ./Modules/DesignMat https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/DesignMat/Polynomial.py -q -o /dev/null")
  os.system("wget -P ./Modules/DesignMat https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/DesignMat/Sigmoidal.py -q -o /dev/null")
if not os.path.exists("./Modules/Regressor"):
  os.makedirs('./Modules/Regressor')
  os.system("wget -P ./Modules/Regressor https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Regressor/__init__.py -q -o /dev/null")
  os.system("wget -P ./Modules/Regressor https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Regressor/Bayesian.py -q -o /dev/null")
  os.system("wget -P ./Modules/Regressor https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Regressor/Empirical.py -q -o /dev/null")
  os.system("wget -P ./Modules/Regressor https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Regressor/LeastSquares.py -q -o /dev/null")
  os.system("wget -P ./Modules/Regressor https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Regressor/Regressor.py -q -o /dev/null")
if not os.path.exists("./Modules/Classifier"):
  os.makedirs('./Modules/Classifier')
  os.system("wget -P ./Modules/Classifier https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Classifier/__init__.py -q -o /dev/null")
  os.system("wget -P ./Modules/Classifier https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Classifier/Logistic.py -q -o /dev/null")
  os.system("wget -P ./Modules/Classifier https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Classifier/LeastSquares.py -q -o /dev/null")
  os.system("wget -P ./Modules/Classifier https://raw.githubusercontent.com/ming-zhao/Intro-to-Deep-Learning/master/Modules/Classifier/Classifier.py -q -o /dev/null")