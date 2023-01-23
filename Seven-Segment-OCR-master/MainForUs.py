from frame_extractor import frameExtractor
from digits_cut import cutDigits

f = frameExtractor(image=None,
                    src_file_name='img/test.jpg',
                    dst_file_name='img/testCropeedOut.jpg',
                    return_image=True,
                    output_shape=(400, 100))
f.extractAndSaveFrame()

