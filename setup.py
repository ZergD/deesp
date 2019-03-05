from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='deesp',
      version='0.1',
      description='The best Dispatcher in the universe',
      url='https://github.com/ZergD/deesp',
      author='MarcM',
      author_email='marc.mozgawa@gmail.com',
      license='LGP-LV3',
      packages=['deesp'],
      install_requires=[
          'markdown',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      entry_points={
          'console_scripts':['test=deesp.cmd_line:main'], 
      },
      zip_safe=False)
