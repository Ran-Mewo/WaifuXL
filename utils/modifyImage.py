from PIL import Image

def centerAndCropImage(targetDimensions, inputFile, outputPath):
    fileName = inputFile.split('/')[-1]
    
    im = Image.open(inputFile)

    width, height = im.size

    widthDiff = width - targetDimensions[0]
    heightDiff = height - targetDimensions[1]

    newLeft = widthDiff/2
    newRight = width-(widthDiff/2)
    newTop = heightDiff/2
    newBottom = height-(heightDiff/2) 

    im1 = im.crop((newLeft,newTop,newRight,newBottom))
    im1.save(f"{outputPath}{fileName}")

centerAndCropImage((500,500),"../images/sampleOne.png","../cropped/")