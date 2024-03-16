import time
from pathlib import Path


class TimeTable(object):
    def __init__(self):
        self.csv = None
        self.reset_epoch()

    def reset_epoch(self):
        """Set all time records to zeros"""
        self.t_epoch = 0.0
        self.t_loop = 0.0
        self.t_eval = 0.0
        self.t_forward = 0.0
        self.t_backward = 0.0
        self.t_sample = 0.0
        self.t_prep_batch = 0.0
        self.t_prep_input = 0.0
        self.t_post_update = 0.0
        self.t_mem_update = 0.0
        self.t_time_zero = 0.0
        self.t_time_nbrs = 0.0
        self.t_self_attn = 0.0

    def start(self):
        # Uncomment for better breakdown timings
        #torch.cuda.synchronize()
        return time.perf_counter()

    def elapsed(self, start):
        # Uncomment for better breakdown timings
        #torch.cuda.synchronize()
        return time.perf_counter() - start

    def print_epoch(self, prefix='  '):
        """Print the timing breakdown of different components in an epoch"""
        lines = f'' \
            f'{prefix}epoch | total:{self.t_epoch:.2f}s loop:{self.t_loop:.2f}s eval:{self.t_eval:.2f}s\n' \
            f'{prefix} loop | forward:{self.t_forward:.2f}s backward:{self.t_backward:.2f}s sample:{self.t_sample:.2f}s prep_batch:{self.t_prep_batch:.2f}s prep_input:{self.t_prep_input:.2f}s post_update:{self.t_post_update:.2f}s\n' \
            f'{prefix} comp | mem_update:{self.t_mem_update:.2f}s time_zero:{self.t_time_zero:.2f}s time_nbrs:{self.t_time_nbrs:.2f}s self_attn:{self.t_self_attn:.2f}s\n'
        print(lines, end='')

    def csv_open(self, path):
        """Close the opened file (if any) and open a new file in write mode"""
        self.csv_close()
        self.csv = Path(path).open('w')

    def csv_close(self):
        """Close the opened file (if any)"""
        if self.csv is not None:
            self.csv.close()
            self.csv = None

    def csv_write_header(self):
        """Write the header line to the CSV file"""
        header = 'epoch,total,loop,eval,' \
            'forward,backward,sample,prep_batch,prep_input,post_update,' \
            'mem_update,time_zero,time_nbrs,self_attn'
        self.csv.write(header + '\n')

    def csv_write_line(self, epoch):
        """Write a line of timing information to the CSV file"""
        line = f'{epoch},{self.t_epoch},{self.t_loop},{self.t_eval},' \
            f'{self.t_forward},{self.t_backward},{self.t_sample},{self.t_prep_batch},{self.t_prep_input},{self.t_post_update},' \
            f'{self.t_mem_update},{self.t_time_zero},{self.t_time_nbrs},{self.t_self_attn}'
        self.csv.write(line + '\n')


# Global for accumulating timings.
tt = TimeTable()
