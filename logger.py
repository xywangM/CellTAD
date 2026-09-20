#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time


class Logger(object):
    def __init__(self, logfile, rank=0, verbose_threshold=0, threaded=False):
        self.threaded = threaded
        self.logfilename = logfile
        if not self.threaded:
            log_dir = os.path.dirname(os.path.abspath(logfile))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        self.log = open(self.logfilename, "a", encoding="utf-8") if not self.threaded else None
        self.terminal = sys.stdout if not self.threaded else None
        self.verbose_threshold = verbose_threshold
        self.rank = rank

    def write(self, message, append_time=True, verbose_level=0, allow_all_ranks=False):
        if append_time:
            message = f'{time.ctime()}: {message}'
        if message and message[-1] != '\n':
            message += '\n'
        if verbose_level <= self.verbose_threshold:
            if (self.rank == 0 or allow_all_ranks) and not self.threaded:
                self.terminal.write(message)
                self.log.write(message)
            elif (self.rank == 0 or allow_all_ranks) and self.threaded:
                with open(self.logfilename, "a", encoding="utf-8") as logfile:
                    logfile.write(message)
                sys.stdout.write(message)

    def flush(self):
        if not self.threaded:
            self.log.flush()
            self.terminal.flush()
        else:
            sys.stdout.flush()

    def set_rank(self, rank):
        self.rank = rank

    def dump_args(self, args):
        self.write("Running CellTAD with following arguments")
        for k, v in vars(args).items():
            self.write(f'\t{k}={v}', append_time=False)