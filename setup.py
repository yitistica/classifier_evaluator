#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = []

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]


if __name__ == '__main__':
    setup(
        author="Yi Q",
        author_email='yitistica@outlook.com',
        python_requires='>=3.6',
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT License',
            'Natural Language :: English',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        description="a classification model evaluation package",
        install_requires=requirements,
        license="MIT license",
        long_description=readme,
        include_package_data=True,
        keywords='month',
        name='month',
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        setup_requires=setup_requirements,
        test_suite='tests',
        tests_require=test_requirements,
        url='https://github.com/yitistica/classifier_evaluator',
        version='0.1.0',
        zip_safe=False,
    )
