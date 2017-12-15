from setuptools import setup, find_packages

setup(name='safekit',
      version=1.0,
      description='Neural Network Anomaly Detection for Multivariate Sequences',
      url='http://aarontuor.site',
      author='Aaron Tuor, Ryan Baerwolf, Robin Cosbey, Nick Knowles',
      author_email='aaron.tuor@pnnl.gov',
      license='MIT',
      packages=find_packages(), # or list of package paths from this directory
      zip_safe=False,
      install_requires=['tensorflow', 'scipy', 'sklearn', 'numpy', 'matplotlib'],
      classifiers=['Programming Language :: Python'],
      keywords=['Deep Learning', 'Anomaly Detection', 'LSTM', 'RNN'])
