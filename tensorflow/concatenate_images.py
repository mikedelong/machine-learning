import json
import logging
import os
import random
import shutil
import time
from PIL import Image

start_time = time.time()

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

with open('test-data-settings.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    blank_file = data['blank_file']
    files_to_generate = data['files_to_generate']
    input_folder = data['input_folder']
    output_folder = data['output_folder']
    random_seed = data['random_seed']

random.seed(random_seed)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    shutil.rmtree(output_folder)
    os.makedirs(output_folder)

count = 0

for file_count in range(0, files_to_generate):
    try:
        output_filename = str(file_count) + '_'
        current = random.randint(0, 99999)
        t0 = [c for c in map(int, str(current))]
        blanks_needed = 5 - len(t0)
        sources = list()
        for blanks in range(0, blanks_needed):
            sources.append(blank_file)
        for c0 in t0:
            t1 = chr(c0 + ord('A'))
            t2 = input_folder + t1
            t3 = random.choice(os.listdir(t2 + '/'))
            output_filename += t1
            file_name = t2 + '/' + t3
            logging.debug(file_name)
            sources.append(file_name)

        images = map(Image.open, sources)
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_image = Image.new('L', (total_width, max_height)) # mode was 'RGB'

        x_offset = 0
        for im in images:
            new_image.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        output_full_filename = output_folder + output_filename + '.png'
        new_image.save(output_full_filename)
        logging.debug('%d %s' % (file_count, output_full_filename))
    except IOError as io_error:
        logging.warn(io_error)

finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logging.info("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
