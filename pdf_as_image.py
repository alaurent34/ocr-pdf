import os
from pdf2image import convert_from_path


def convert_pdf_as_im(path):
    pages = convert_from_path(path, dpi=200)
    _, filename = os.path.split(path)
    filename = filename.split('.')[0]
    save_path = f'images_pdf/{filename}/'

    os.makedirs(save_path, exist_ok=True)

    i = 1
    for page in pages:
        image_name = "Page_" + str(i) + ".jpg"  
        page.save(os.path.join(save_path, image_name), "JPEG")
        i = i+1  
    
if __name__ == '__main__':
    convert_pdf_as_im(r"./pdf/Secteur 4.pdf")