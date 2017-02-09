import logging

from PIL import Image

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

import os, random

random_seed = 1
random.seed(random_seed)

count = 0
root ='./notMNIST_small/'
while count < 5:
    t0 = random.choice(os.listdir(root))
    t1 = root + t0
    if os.path.isdir(t1):
        t2  =  random.choice(os.listdir(t1 + '/'))
        count += 1
        logging.debug(t1 + '/' + t2)

# images = map(Image.open, ['Test1.jpg', 'Test2.jpg', 'Test3.jpg'])
# widths, heights = zip(*(i.size for i in images))
#
# total_width = sum(widths)
# max_height = max(heights)
#
# new_im = Image.new('RGB', (total_width, max_height))
#
# x_offset = 0
# for im in images:
#     new_im.paste(im, (x_offset, 0))
#     x_offset += im.size[0]
#
# new_im.save('test.jpg')
