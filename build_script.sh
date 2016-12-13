pip install --upgrade pip

pip install numpy
pip install scipy
pip install matplotlib

#for installing box2d and getting that running
#first you need to install swig globally and symlink to the venv
#change installation directory as necessary to where swig is installed

#brew install swig
#ln -s /usr/local/Cellar/swig/3.0.10_1/bin/swig PATH_TO_VENV/venv/bin/swig

#based on thread, https://github.com/openai/gym/issues/100
cd PATH_TO_VENV/venv
git clone https://github.com/pybox2d/pybox2d
cd ./pybox2d
python setup.py clean
python setup.py build
python setup.py install


