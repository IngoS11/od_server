from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time
import logging
import numpy as np

from annotation import Annotator
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from watchdog.observers.polling import PollingObserver
from watchdog.events import PatternMatchingEventHandler

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

interpreter = None
labels = None
threshold = None

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s;%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def on_created(event):
    logging.info(f"{event.src_path} was created")

    interpreter = Interpreter(model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    image = Image.open(event.src_path).convert('RGB').resize(
                 (input_width, input_height), Image.ANTIALIAS)
    start_time = time.monotonic()
    results = detect_objects(interpreter, image, threshold)
    elapsed_ms = (time.monotonic() - start_time) * 1000
#
#  annotator.clear()
#  annotate_objects(annotator, results, labels)
#  annotator.text([5, 0], '%.1fms' % (elapsed_ms))
#  annotator.update()

def on_deleted(event):
    logging.info(f"{event.src_path} was deleted")

def on_modified(event):
    logging.info(f"{event.src_path} has been modified")

def on_moved(event):
    logging.info(f"{event.src_path} was moved to {event.dest_path}")


def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results


def annotate_objects(annotator, results, labels):
  """Draws the bounding box and label for each object in the results."""
  for obj in results:
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * CAMERA_WIDTH)
    xmax = int(xmax * CAMERA_WIDTH)
    ymin = int(ymin * CAMERA_HEIGHT)
    ymax = int(ymax * CAMERA_HEIGHT)

    # Overlay the box, label, and score on the camera preview
    annotator.bounding_box([xmin, ymin, xmax, ymax])
    annotator.text([xmin, ymin],
                   '%s\n%.2f' % (labels[obj['class_id']], obj['score']))


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  parser.add_argument(
      '--snapshot_dir',
      help='Directory to scan for snapshots.',
      required=True)      
  parser.add_argument(
      '--threshold',
      help='Score threshold for detected objects.',
      required=False,
      type=float,
      default=0.4)

  args = parser.parse_args()

  global threshold, labels, model

  threshold = args.threshold
  labels = load_labels(args.labels)
  model = args.model

  patterns = "*"
  ignore_patterns = ""
  ignore_directories = True
  case_sensitive = False
  event_handler = PatternMatchingEventHandler(patterns,
                                              ignore_patterns,
                                              ignore_directories,
                                              case_sensitive)
  
  event_handler.on_created = on_created
  event_handler.on_deleted = on_deleted
  event_handler.on_modified = on_modified
  event_handler.on_moved = on_moved

  observer = PollingObserver()
  observer.schedule(event_handler, args.snapshot_dir, recursive=False)
  logging.info(f"Starting to Watch {args.snapshot_dir}...")
  observer.start()

  try:
      while True:
          time.sleep(1)
  except KeyboardInterrupt:
      observer.stop()
      observer.join()

if __name__ == '__main__':
  main()
