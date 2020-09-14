# https://github.com/KaiyangZhou/Dassl.pytorch
from collections import defaultdict
import torch

# to redirect evaluations
import sys
import io


class CsvStream:
    """ Redirect stdout output to a csv file. """

    def __init__(self, filepath: str, parser: callable):
        self.filepath = filepath
        self.file = None
        self.buffer = io.StringIO()
        self.parser = parser

    def write(self, s):
        self.buffer.write(s)

    def __enter__(self):
        self.file = open(self.filename, "w")
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        out = self.parser(self.buffer.getvalue())
        self.file.write(out)
        self.buffer.close()
        self.file.close()
        sys.stdout = sys.__stdout__


class AverageMeter:
    """Compute and store the average and current value.
    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter:
    """Store the average and current value for a set of metrics.
    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    """ def to_writer(self, writer, tag, n_iter):
        for name, meter in self.meters.items():
            writer.add_scalars(f"{tag}/{name}", {
                "val": meter.val,
                "avg": meter.avg,
            }, n_iter) """

    def to_writer(self, writer, tag, n_iter):
        for name, meter in self.meters.items():
            writer.add_scalar(f"{tag}/{name}", meter.avg, n_iter)

    def get_avg(self, tag):
        return self.meters[tag].avg

    def get_val(self, tag):
        return self.meters[tag].val

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                '{} {:.4f} ({:.4f})'.format(name, meter.val, meter.avg)
            )
        return self.delimiter.join(output_str)


class Config(object):

    def __init__(self, config_path):
        with open(config_path, "r") as fh:
            content = fh.read()
        self.parse(content)

    def parse(self, content):
        for line in content.split('\n'):
            if len(line) == 0 or line.startswith('#'):
                continue  # skip comments and empty lines
            k, v = line.split(': ')
            if '[' in v:  # parse lists
                is_float = True if '.' in v else False
                v = v.strip()[1:-1].split(',')
                v = list(map(float if is_float else int, v))
            elif '/' in v:  # parse paths
                pass
            else:  # parse integer, floating point or string values
                is_float = True if '.' in v else False
                try:
                    v = float(v) if is_float else int(v)
                except ValueError:
                    if "True" in v:
                        v = True
                    elif "False" in v:
                        v = False

            setattr(self, k, v)

    def __repr__(self):
        info = []
        for k, v in self.__dict__.items():
            info.append(f"{k}: {v}\n")
        return "".join(info)


if __name__ == "__main__":
    filepath = './evaluation/example.txt'

    parser = lamda x: x

    with CsvStream():
        print('hello world\nhy')
