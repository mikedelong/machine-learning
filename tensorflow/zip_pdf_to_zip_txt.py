import json
import logging
import time
import zipfile
import textract

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)


def run():
    start_time = time.time()

    with open('zip_pdf_to_zip_txt-settings.json') as data_file:
        data = json.load(data_file)
        logging.debug(data)
        zip_input_file = data['zip_input_file']
        zip_output_file = data['zip_output_file']
        with zipfile.ZipFile(zip_output_file, 'w') as output_zip:
            with zipfile.ZipFile(zip_input_file, 'r') as input_zip:
                file_list = input_zip.filelist
                for item in file_list:
                    file_name = item.filename
                    output_file_name = str(file_name).replace('.pdf', '.txt')
                    logging.info(file_name)
                    try:
                        data =input_zip.read(file_name)
                        # todo remove the working file if it exists
                        # todo figure out how to do this without writing a PDF file
                        with open('working.pdf', 'wb') as output_pdf:
                            output_pdf.write(data)
                            output_pdf.close()
                        t = textract.process('working.pdf')
                        with open(output_file_name, 'wb') as output_txt:
                            output_txt.write(t)
                        output_zip.write(output_file_name)
                        logging.info(t)
                    except KeyError:
                        logging.error('ERROR: Did not find %s in zip file' % file_name)
                    # todo clean up the missing PDF file if it exists



    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))


if __name__ == '__main__':
    run()
