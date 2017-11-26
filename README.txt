Installation notes
#################
#Get the package
git clone https://github.com/cthurau/pymf.git

#Try to download the dependencies
cd pymf
python setup.py install

#I ran into an error trying to install cvxopt, so I installed a binary version with pip and reran setup.py
pip install cvxopt
#Alternative ways to install can be found here: http://cvxopt.org/install/index.html
