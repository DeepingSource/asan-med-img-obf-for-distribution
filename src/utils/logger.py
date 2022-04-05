import os
import logging

def get_logger(log_path, filename="log.txt"):
    logger = logging.getLogger('logbuch')
    logger.setLevel(level=logging.DEBUG)
    
    # Stream handler
    sh = logging.StreamHandler()
    sh.setLevel(level=logging.DEBUG)
    sh_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    sh.setFormatter(sh_formatter)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_path, filename))
    fh.setLevel(level=logging.DEBUG)
    fh_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    fh.setFormatter(fh_formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
