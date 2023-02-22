# Introduction

Repo to extract tables from pdf images with OCR.

# Requierements

Along with the python requirements that are listed in requirements.txt, there are a few external requirements for some of the modules.

I haven&rsquo;t looked into the minimum required versions of these dependencies, but I&rsquo;ll list the versions that I&rsquo;m using.

-   `pdfimages` 20.09.0 of [Poppler](https://poppler.freedesktop.org/)
-   `tesseract` 5.0.0 of [Tesseract](https://github.com/tesseract-ocr/tesseract)
-   `mogrify` 7.0.10 of [ImageMagick](https://imagemagick.org/index.php)

# Usage

Place your pdfs in a root directory named `pdf/`.

Then, change line 341 to 350 of `extract_tables.py` to the specificities of your files :

```python
my_csv_tb = [row.replace('\n', ' ').replace('"', '').split(' ')[0][2:] for row in my_csv.split('\r\n')]
if len(my_csv_tb) > 1:
    data = pd.DataFrame(data=my_csv_tb[1:], columns=['plaque'])
    data['zone'] = zone
    data['page'] = re.findall(r'Page_\d+', image)[0]
    data['mauvaise long.'] = (data.plaque.str.len() - len("{:02d}".format(int(zone))) > 6) | \
                            (data.plaque.str.len() - len("{:02d}".format(int(zone))) < 6)
    data = data[~data.plaque.isna() & (data.plaque.str.strip() != "")]
else: 
    data = pd.DataFrame(columns=['plaque', 'zone', 'page', 'à verifier'])
```

You might also need to changes the parameters from Tesseract (line 313). The command `tessarct --help-extra` 
might be handy.

Eventually, you can execute the program with the following command :

```sh
python extract_tables.py
```