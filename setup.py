from setuptools import setup, find_packages


def main():
    setup(
        name='muv',
        version='0.1',
        license='3-clause BSD',
        url='https://github.com/skearnes/muv',
        description='Generate maximum unbiased validation (MUV) datasets for virtual screening',
        packages=find_packages(),
    )

if __name__ == '__main__':
    main()
