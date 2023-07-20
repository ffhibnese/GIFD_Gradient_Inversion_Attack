import fitz
import os
import glob
def png2pdf(path='.'):
     for name in glob.glob(os.path.join(path, '*.png')):
        imgdoc = fitz.open(name)
        pdfbytes = imgdoc.convert_to_pdf()    # 使用图片创建单页的 PDF
        imgpdf = fitz.open("pdf", pdfbytes)
        imgpdf.save(name[:-4] + '.pdf')
path = "ood_outputs/cartoon/ffhq/GILO"
png2pdf(path)