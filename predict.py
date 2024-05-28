# Prediction interface for Cog ⚙️
# https://cog.run/python
import io
import tempfile

from PIL import Image
import numpy as np
from cog import BasePredictor, Input, Path
import onnxruntime as ort

class Predictor(BasePredictor):
    model: ort.InferenceSession

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = ort.InferenceSession("./public/models/superRes.onnx", providers=['CUDAExecutionProvider'])


    def predict(
            self,
            image: Path = Input(description="Input image"),
            scale: int = Input(
                description="Factor to scale the image by", ge=0, default=2
            ),
            format: str = Input(
                description="Output type of the image", default="PNG"
            )
    ) -> Path:
        """Run a single prediction on the model"""
        processed_input = preprocess(image)
        output = multiUpscale(self.model, processed_input, scale, format.upper())
        out_path = Path(tempfile.mkdtemp()) / f'out.{format.lower()}'
        out_path.write_bytes(output)
        return out_path


def preprocess(image: Path) -> np.ndarray:
    img = Image.open(str(image))
    array = np.array(img) # The image is already in the correct format by default (W, H, C)
    if array.ndim == 4:
        # animated gif with multiple frames
        N, W, H, C = array.shape
        numPixelsInFrame = W * H
        for i in range(N):
            currIndex = i * W * H * C
            prevIndex = (i - 1) * W * H * C
            for j in range(numPixelsInFrame):
                curr = currIndex + j * C
                if array.ravel()[curr + C - 1] == 0:
                    prev = prevIndex + j * C
                    for k in range(C):
                        array.ravel()[curr + k] = array.ravel()[prev + k]
    elif array.ndim == 2:
        # Grayscale image, add three dimensions for RGB and one for alpha
        array = np.stack((array,)*4, axis=-1)
    elif array.shape[-1] == 3:
        # RGB image, add one dimension for alpha
        array = np.concatenate((array, np.zeros((*array.shape[:-1], 1), dtype=array.dtype)), axis=-1)
    return array.astype(np.uint8)


def runSuperRes(model, imageArray):
    # Get the input name from the model description
    input_name = model.get_inputs()[0].name
    # Get the output name from the model description
    output_name = model.get_outputs()[0].name
    try:
        return model.run([output_name], {input_name: imageArray})[0]
    except Exception as e:
        print("Failed to run super resolution", e)
    return None


def multiUpscale(model, imageArray, upscaleFactor, outputType = 'PNG'):
    outArr = imageArray
    for s in range(upscaleFactor):
        outArr = upscaleFrame(model, outArr)
    return imageNDarrayToBytes(outArr, outputType)


def upscaleFrame(model, imageArray):
    CHUNK_SIZE = 1024
    PAD_SIZE = 32

    inImgW = imageArray.shape[0]
    inImgH = imageArray.shape[1]
    outImgW = inImgW * 2
    outImgH = inImgH * 2
    nChunksW = int(np.ceil(inImgW / CHUNK_SIZE))
    nChunksH = int(np.ceil(inImgH / CHUNK_SIZE))
    chunkW = int(np.floor(inImgW / nChunksW))
    chunkH = int(np.floor(inImgH / nChunksH))

    outArr = np.zeros((outImgW, outImgH, 4), dtype=np.uint8)

    for i in range(int(nChunksH)):
        for j in range(int(nChunksW)):
            x = j * chunkW
            y = i * chunkH

            yStart = max(0, y - PAD_SIZE)
            xStart = max(0, x - PAD_SIZE)
            inH = inImgH - yStart if yStart + chunkH + PAD_SIZE * 2 > inImgH else chunkH + PAD_SIZE * 2
            outH = 2 * (min(inImgH, y + chunkH) - y)
            inW = inImgW - xStart if xStart + chunkW + PAD_SIZE * 2 > inImgW else chunkW + PAD_SIZE * 2
            outW = 2 * (min(inImgW, x + chunkW) - x)

            inSlice = imageArray[xStart:xStart+inW, yStart:yStart+inH, :4]
            subArr = np.zeros(inSlice.shape, dtype=np.uint8)
            np.copyto(subArr, inSlice)

            chunkData = runSuperRes(model, subArr)
            chunkArr = np.array(chunkData.data).reshape(chunkData.shape)
            chunkSlice = chunkArr[(x - xStart) * 2 : (x - xStart) * 2 + outW, (y - yStart) * 2 : (y - yStart) * 2 + outH, :4]
            outSlice = outArr[x * 2 : x * 2 + outW, y * 2 : y * 2 + outH, :4]
            np.copyto(outSlice, chunkSlice)

    return outArr


def imageNDarrayToBytes(outArr, outputType = 'PNG'):
    pil_img = Image.fromarray(outArr.astype('uint8'))
    buff = io.BytesIO()
    pil_img.save(buff, format=outputType)
    return buff.getvalue()
