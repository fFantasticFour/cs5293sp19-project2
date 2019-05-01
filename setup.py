from setuptools import setup, find_packages

setup(
        name='project2',
        version='1.0',
        author='Brandon Wolfe',
        authour_email='Brandon.E.Wolfe-1@ou.edu',
        packages=find_packages(exclude=('tests', 'redacted_docs')),
        setup_requires=['pytest-runner'],
        tests_require=['pytest']
)


