import logging
import os
import random
import shutil
import time
from PIL import Image

start_time = time.time()

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

random_seed = 1
random.seed(random_seed)

output_folder = './concatenate_output/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    shutil.rmtree(output_folder)
    os.makedirs(output_folder)

count = 0
root = './notMNIST_small/'
for file_count in range(0, 10000):
    try:
        output_filename = str(file_count) + '_'
        current = random.randint(0, 99999)
        t0 = [c for c in map(int, str(current))]
        blanks_needed = 5 - len(t0)
        sources = list()
        for blanks in range(0, blanks_needed):
            sources.append('./blank.png')
        for c0 in t0:
            t1 = chr(c0 + ord('A'))
            t2 = root + t1
            t3 = random.choice(os.listdir(t2 + '/'))
            output_filename += t1
            file_name = t2 + '/' + t3
            logging.debug(file_name)
            sources.append(file_name)

        images = map(Image.open, sources)
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_image.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        output_full_filename = output_folder + output_filename + '.png'
        new_image.save(output_full_filename)
        logging.debug('%d %s' % (file_count, output_full_filename))
    except IOError as io_error:
        logging.warn(io_error)

# to get a reasonably stable estimate of elapsed time we want to stop the clock before we plot.
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logging.info("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
